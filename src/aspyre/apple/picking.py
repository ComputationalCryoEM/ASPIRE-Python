import os
import logging
from concurrent import futures

import mrcfile
import numpy as np
from tqdm import tqdm

from scipy import ndimage, misc, signal
from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation, center_of_mass
from sklearn import svm, preprocessing

from aspyre import config
from aspyre.apple.helper import PickerHelper
from aspyre.utils.numeric import xp

logger = logging.getLogger(__name__)


class Picker:
    """ This class does the actual picking with help from PickerHelper class. """

    def __init__(self, particle_size, max_size, min_size, query_size, tau1, tau2, moa,
                 container_size, filename, output_directory):

        self.particle_size = int(particle_size / 2)
        self.max_size = int(max_size / 2)
        self.min_size = int(min_size / 2)
        self.query_size = int(query_size / 2)
        self.query_size -= self.query_size % 2
        self.tau1 = tau1
        self.tau2 = tau2
        self.moa = int(moa / 2)
        self.container_size = int(container_size / 2)
        self.filename = filename
        self.output_directory = output_directory

        self.original_im = None  # populated in read_mrc()
        self.im = self.read_mrc()

    def read_mrc(self):
        """Gets and preprocesses micrograph.

        Reads the micrograph, applies binning and a low-pass filter.

        Returns:
            Micrograph image.
        """

        with mrcfile.open(self.filename, mode='r+', permissive=True) as mrc:
            im = mrc.data.astype('float')
        self.original_im = im

        # Discard outer pixels
        im = im[
            config.apple.mrc_margin_top: -config.apple.mrc_margin_bottom,
            config.apple.mrc_margin_left: -config.apple.mrc_margin_right
        ]

        # Make square
        side_length = min(im.shape)
        im = im[:side_length, :side_length]

        im = misc.imresize(im, 1/config.apple.mrc_shrink_factor, mode='F', interp='cubic')
        im = signal.correlate(
            im,
            PickerHelper.gaussian_filter(
                config.apple.mrc_gauss_filter_size,
                config.apple.mrc_gauss_filter_sigma
            ),
            'same'
        )

        return im.astype('double')

    def query_score(self, show_progress=True):
        """Calculates score for each query image.

        Extracts query images and reference windows. Computes the cross-correlation between these
        windows, and applies a threshold to compute a score for each query image.

        Args:
            show_progress: Whether to show a progress bar

        Returns:
            Matrix containing a score for each query image.
        """

        micro_img = xp.asarray(self.im)
        logger.info('Extracting query images')
        query_box = PickerHelper.extract_query(micro_img, self.query_size // 2)
        logger.info('Extracting query images complete')

        query_box = xp.conj(xp.fft2(query_box, axes=(2, 3)))

        reference_box = PickerHelper.extract_references(micro_img, self.query_size, self.container_size)

        reference_size = PickerHelper.reference_size(micro_img, self.container_size)
        conv_map = xp.zeros((reference_size, query_box.shape[0], query_box.shape[1]))

        def _work(index):
            reference_box_i = xp.fft2(reference_box[index], axes=(0, 1))
            window_t = xp.multiply(reference_box_i, query_box)
            cc = xp.ifft2(window_t, axes=(2, 3))
            return index, cc.real.max((2, 3)) - cc.real.mean((2, 3))

        n_works = reference_size
        n_threads = config.apple.conv_map_nthreads
        pbar = tqdm(total=reference_size, disable=not show_progress)

        # Ideally we'd like something like 'SerialExecutor' to enable easy debugging
        # but for now do an if-else
        if n_threads > 1:
            with futures.ThreadPoolExecutor(n_threads) as executor:
                to_do = [executor.submit(_work, i) for i in range(n_works)]

                for future in futures.as_completed(to_do):
                    i, res = future.result()
                    conv_map[i, :, :] = res
                    pbar.update(1)
        else:
            for i in range(n_works):
                _, conv_map[i, :, :] = _work(i)
                pbar.update(1)

        pbar.close()

        conv_map = xp.transpose(conv_map, (1, 2, 0))

        min_val = xp.min(conv_map)
        max_val = xp.max(conv_map)
        thresh = min_val + (max_val - min_val) / config.apple.response_thresh_norm_factor
        return xp.asnumpy(xp.sum(conv_map >= thresh, axis=2))

    def run_svm(self, score):
        """
        Trains and uses an SVM classifier.

        Trains an SVM classifier to distinguish between noise and particle projections based on
        mean intensity and variance. Every possible window in the micrograph is then classified
        as either noise or particle, resulting in a segmentation of the micrograph.

        Args:

            score: Matrix containing a score for each query image.

        Returns:
            Segmentation of the micrograph into noise and particle projections.
        """

        micro_img = xp.asarray(self.im)
        particle_windows = np.floor(self.tau1)
        non_noise_windows = np.ceil(self.tau2)
        bw_mask_p, bw_mask_n = Picker.get_maps(self, score, micro_img, particle_windows, non_noise_windows)

        x, y = PickerHelper.get_training_set(micro_img, bw_mask_p, bw_mask_n, self.query_size)
        x = xp.asnumpy(x)
        y = xp.asnumpy(y)

        scaler = preprocessing.StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)
        classify = svm.SVC(C=1, kernel=config.apple.svm_kernel, gamma=config.apple.svm_gamma, class_weight='balanced')
        classify.fit(x, y)

        mean_all, std_all = PickerHelper.moments(micro_img, self.query_size)
        mean_all = xp.asnumpy(mean_all)
        std_all = xp.asnumpy(std_all)

        mean_all = mean_all[self.query_size - 1:-(self.query_size - 1),
                            self.query_size - 1:-(self.query_size - 1)]

        std_all = std_all[self.query_size - 1:-(self.query_size - 1),
                          self.query_size - 1:-(self.query_size - 1)]

        mean_all = np.reshape(mean_all, (np.prod(mean_all.shape), 1), 'F')
        std_all = np.reshape(std_all, (np.prod(std_all.shape), 1), 'F')
        cls_input = np.concatenate((mean_all, std_all), axis=1)
        cls_input = scaler.transform(cls_input)

        # compute classification for all possible windows in micrograph
        segmentation = classify.predict(cls_input)

        _segmentation_shape = int(np.sqrt(segmentation.shape[0]))
        segmentation = np.reshape(segmentation, (_segmentation_shape, _segmentation_shape), 'F')

        return segmentation.copy()

    def morphology_ops(self, segmentation):
        """
        Discards suspected artifacts from segmentation.

        Args:
            segmentation: Segmentation of the micrograph into noise and particle projections.

        Returns:
            Segmentation of the micrograph into noise and particle projections.
        """

        if (binary_fill_holes(segmentation) == np.ones(segmentation.shape)).all():
            segmentation[0:100, 0:100] = np.zeros((100, 100))

        segmentation = binary_fill_holes(segmentation)
        y, x = np.ogrid[-self.min_size:self.min_size+1, -self.min_size:self.min_size+1]
        element = x*x+y*y <= self.min_size * self.min_size
        segmentation_e = binary_erosion(segmentation, element)

        y, x = np.ogrid[-self.max_size:self.max_size+1, -self.max_size:self.max_size+1]
        element = x*x+y*y <= self.max_size * self.max_size
        segmentation_o = binary_erosion(segmentation, element)
        segmentation_o = np.reshape(segmentation_o,
                                    (segmentation_o.shape[0], segmentation_o.shape[1], 1), 'F')

        size_const, _ = ndimage.label(segmentation_e, np.ones((3, 3)))
        size_const = np.reshape(size_const, (size_const.shape[0], size_const.shape[1], 1), 'F')
        labels = np.unique(size_const*segmentation_o)
        idx = np.where(labels != 0)
        labels = np.take(labels, idx)
        labels = np.reshape(labels, (1, 1, np.prod(labels.shape)), 'F')

        matrix1 = np.repeat(size_const, labels.shape[2], 2)
        matrix2 = np.repeat(labels, matrix1.shape[0], 0)
        matrix2 = np.repeat(matrix2, matrix1.shape[1], 1)

        matrix3 = np.equal(matrix1, matrix2)
        matrix4 = np.sum(matrix3, 2)

        segmentation_e[np.where(matrix4 == 1)] = 0

        return segmentation_e

    def extract_particles(self, segmentation):
        """
        Saves particle centers into output .star file, after dismissing regions
        that are too big to contain a particle.

        Args:
            segmentation: Segmentation of the micrograph into noise and particle projections.
        """
        segmentation = segmentation[self.query_size // 2 - 1:-self.query_size // 2,
                                    self.query_size // 2 - 1:-self.query_size // 2]
        labeled_segments, _ = ndimage.label(segmentation, np.ones((3, 3)))
        values, repeats = np.unique(labeled_segments, return_counts=True)

        values_to_remove = np.where(repeats > self.max_size ** 2)
        values = np.take(values, values_to_remove)
        values = np.reshape(values, (1, 1, np.prod(values.shape)), 'F')

        labeled_segments = np.reshape(labeled_segments, (labeled_segments.shape[0],
                                                         labeled_segments.shape[1], 1), 'F')
        matrix1 = np.repeat(labeled_segments, values.shape[2], 2)
        matrix2 = np.repeat(values, matrix1.shape[0], 0)
        matrix2 = np.repeat(matrix2, matrix1.shape[1], 1)

        matrix3 = np.equal(matrix1, matrix2)
        matrix4 = np.sum(matrix3, 2)

        segmentation[np.where(matrix4 == 1)] = 0
        labeled_segments, _ = ndimage.label(segmentation, np.ones((3, 3)))

        max_val = np.amax(np.reshape(labeled_segments, (np.prod(labeled_segments.shape))))
        center = center_of_mass(segmentation, labeled_segments, np.arange(1, max_val))
        center = np.rint(center)

        img = np.zeros((segmentation.shape[0], segmentation.shape[1]))
        img[center[:, 0].astype(int), center[:, 1].astype(int)] = 1
        y, x = np.ogrid[-self.moa:self.moa+1, -self.moa:self.moa+1]
        element = x*x+y*y <= self.moa * self.moa
        img = binary_dilation(img, structure=element)
        labeled_img, _ = ndimage.label(img, np.ones((3, 3)))
        values, repeats = np.unique(labeled_img, return_counts=True)
        y = np.where(repeats == np.count_nonzero(element))
        y = np.array(y)
        y = y.astype(int)
        y = np.reshape(y, (np.prod(y.shape)), 'F')
        y -= 1
        center = center[y, :]

        center = center + (self.query_size // 2 - 1) * np.ones(center.shape)
        center = center + (self.query_size // 2 - 1) * np.ones(center.shape)
        center = center + np.ones(center.shape)

        center = config.apple.mrc_shrink_factor * center

        # swap columns to align with Relion
        center = center[:, [1, 0]]

        # first column is x; second column is y - offset by margins that were discarded from the image
        center[:, 0] += config.apple.mrc_margin_left
        center[:, 1] += config.apple.mrc_margin_top

        if self.output_directory is not None:
            basename = os.path.basename(self.filename)
            name_str, ext = os.path.splitext(basename)

            applepick_path = os.path.join(self.output_directory, "{}_applepick.star".format(name_str))
            with open(applepick_path, "w") as f:
                np.savetxt(f, ["data_root\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2"], fmt='%s')
                np.savetxt(f, center, fmt='%d %d')

        return center

    def get_maps(self, score, micro_img, particle_windows, non_noise_windows):
        """
        Gets maps of regions from which to extract particle training for the SVM classifier.

        Args:
            score: Matrix containing a score for each query image.
            micro_img: Micrograph image.
            particle_windows: Number of windows that must contain a particle.
            non_noise_windows: Number of windows that contain neither noise nor particles.
        """
        particles = particle_windows.astype(int)
        non_noise = non_noise_windows.astype(int)

        idx = np.argsort(-np.reshape(score, (np.prod(score.shape)), 'F'))
        x, y = np.unravel_index(idx, score.shape)
        bw_mask_p = np.zeros((micro_img.shape[0], micro_img.shape[1]))
        qs = self.query_size // 2

        begin_row_idx = y * qs
        end_row_idx = np.minimum(begin_row_idx + self.query_size, bw_mask_p.shape[0])
        begin_col_idx = x * qs
        end_col_idx = np.minimum(begin_col_idx + self.query_size, bw_mask_p.shape[1])

        for j in range(particles):
            bw_mask_p[begin_row_idx[j]:end_row_idx[j], begin_col_idx[j]:end_col_idx[j]] = 1

        bw_mask_n = np.copy(bw_mask_p)
        for j in range(particles, non_noise):
            bw_mask_n[begin_row_idx[j]:end_row_idx[j], begin_col_idx[j]:end_col_idx[j]] = 1

        return bw_mask_p, bw_mask_n
