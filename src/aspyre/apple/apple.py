import logging
import glob
import os
import numpy as np
from concurrent import futures
from tqdm import tqdm
from scipy import misc

from aspyre.apple.picking import Picker
from aspyre import config
from aspyre.utils import ensure

logger = logging.getLogger(__name__)


class Apple:
    def __init__(self, output_dir=None):

        self.particle_size = config.apple.particle_size
        self.query_image_size = config.apple.query_image_size
        self.max_particle_size = config.apple.max_particle_size or self.particle_size * 2
        self.min_particle_size = config.apple.min_particle_size or self.particle_size // 4
        self.minimum_overlap_amount = config.apple.minimum_overlap_amount or self.particle_size // 10
        self.container_size = config.apple.container_size
        self.n_processes = config.apple.n_processes
        self.output_dir = output_dir

        if self.query_image_size is None:
            query_image_size = np.round(self.particle_size * 2 / 3)
            query_image_size -= query_image_size % 4
            query_image_size = int(query_image_size)
    
            self.query_image_size = query_image_size

        q_box = (4000 ** 2) / (self.query_image_size ** 2) * 4
        self.tau1 = config.apple.tau1 or int(q_box * 0.03)
        self.tau2 = config.apple.tau2 or int(q_box * 0.3)

        if self.output_dir is not None and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.verify_input_values()

    def verify_input_values(self):
        ensure(1 <= self.max_particle_size <= 3000, "Max particle size must be in range [1, 3000]!")
        ensure(1 <= self.query_image_size <= 3000, "Query image size must be in range [1, 3000]!")
        ensure(5 <= self.particle_size < 3000, "Particle size must be in range [5, 3000]!")
        ensure(1 <= self.min_particle_size < 3000, "Min particle size must be in range [1, 3000]!")

        max_tau1_value = (4000 / self.query_image_size * 2) ** 2
        ensure(0 <= self.tau1 <= max_tau1_value, f"tau1 must be a in range [0, {max_tau1_value}]!")

        max_tau2_value = max_tau1_value
        ensure(0 <= self.tau2 <= max_tau2_value, f"tau2 must be in range [0, {max_tau2_value}]!")

        ensure(0 <= self.minimum_overlap_amount <= 3000, "overlap must be in range [0, 3000]!")

        # max container_size condition is (container_size_max * 2 + 200 > 4000), which is 1900
        ensure(self.particle_size <= self.container_size <= 1900,
               f"Container size must be within range [{self.particle_size}, 1900]!")

        ensure(self.particle_size >= self.query_image_size,
               f"Particle size ({self.particle_size}) must exceed query image size ({self.query_image_size})!")

    def process_folder(self, folder, create_jpg=False):
        filenames = glob.glob('{}/*.mrc'.format(folder))
        logger.info(f"converting {len(filenames)} mrc files")
        logger.info(f"launching {self.n_processes} processes")

        pbar = tqdm(total=len(filenames))
        with futures.ProcessPoolExecutor(self.n_processes) as executor:
            to_do = []
            for filename in filenames:
                future = executor.submit(self.process_micrograph, filename, False, False, False, create_jpg)
                to_do.append(future)

            for future in futures.as_completed(to_do):
                # Retrieve (None) result, since this operation re-raises Exceptions, if any.
                _ = future.result()
                pbar.update(1)
        pbar.close()

    def process_micrograph(self, filepath, return_centers=True, return_img=False, show_progress=True, create_jpg=False):
        ensure(not all([return_centers, return_img]), "Cannot specify both return_centers and return_img")
        ensure(filepath.endswith('.mrc'), f"Input file doesn't seem to be in MRC format! ({filepath})")

        picker = Picker(self.particle_size, self.max_particle_size, self.min_particle_size, self.query_image_size,
                        self.tau1, self.tau2, self.minimum_overlap_amount, self.container_size, filepath,
                        self.output_dir)

        logger.info('Computing scores for query images')
        score = picker.query_score(show_progress=show_progress)  # compute score using normalized cross-correlations

        while True:
            logger.info(f'Running svm with tau1={picker.tau1}, tau2={picker.tau2}')
            # train SVM classifier and classify all windows in micrograph
            segmentation = picker.run_svm(score)

            # If all windows are classified identically, update tau_1 or tau_2
            if np.all(segmentation):
                picker.tau2 += 500
            elif not np.any(segmentation):
                picker.tau1 += 500
            else:
                break

        logger.info('Discarding suspected artifacts')
        segmentation = picker.morphology_ops(segmentation)

        logger.info('Getting particle centers')
        centers = picker.extract_particles(segmentation)

        particle_image = None
        if create_jpg and self.output_dir is not None:
            particle_image = self.particle_image(picker.original_im, picker.particle_size, centers)
            misc.imsave(
                os.path.join(self.output_dir, os.path.splitext(os.path.basename(picker.filename))[0] + '_result.jpg'),
                particle_image
            )

        if return_centers:
            return centers
        elif return_img:
            if particle_image is not None:
                return particle_image
            else:
                return self.particle_image(picker.original_im, picker.particle_size, centers)

    def particle_image(self, micro_img, particle_size, centers):
        """
        Return a numpy array representing the picked centers on a micrograph, suitable for rendering in a jupyter
            notebook or saving as a jpg etc.
        :param micro_img: The micrograph image as a numpy array
        :param particle_size: Particle size of picked particles
        :param centers: Picked centers for micrograph.
        :return: A numpy array with picked centers displayed as rectangles
        """
        micro_img = micro_img - np.amin(np.reshape(micro_img, (np.prod(micro_img.shape))))
        picks = np.ones(micro_img.shape)
        for i in range(0, centers.shape[0]):
            y = int(centers[i, 1])
            x = int(centers[i, 0])
            d = int(np.floor(particle_size))
            picks[y-d:y-d+5, x-d:x+d] = 0
            picks[y+d:y+d+5, x-d:x+d] = 0
            picks[y-d:y+d, x-d:x-d+5] = 0
            picks[y-d:y+d, x+d:x+d+5] = 0

        return np.multiply(micro_img, picks)
