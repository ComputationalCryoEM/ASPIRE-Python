import logging
import os
from concurrent import futures
from glob import glob

import numpy as np
from PIL import Image

from aspire.apple.picking import Picker
from aspire.utils import tqdm

logger = logging.getLogger(__name__)


class Apple:
    def __init__(
        self,
        particle_size,
        output_dir="",
        min_particle_size=None,
        max_particle_size=None,
        query_image_size=None,
        minimum_overlap_amount=None,
        tau1=None,
        tau2=None,
        container_size=450,
        model="svm",
        model_opts=None,
        mrc_margin_left=99,
        mrc_margin_right=100,
        mrc_margin_top=99,
        mrc_margin_bottom=100,
        mrc_shrink_factor=2,
        mrc_gauss_filter_size=15,
        mrc_gauss_filter_sigma=0.5,
        response_thresh_norm_factor=20,
        conv_map_nthreads=4,
        n_processes=1,
    ):
        """
        Instantiate an Apple instance with a given configuration.

        Some discussion of parameters can be found in:

            APPLE picker : Automatic particle picking, a low-effort cryo-EM framework.
            Heimowitz, Ayelet; Andén, Joakim; Singer, Amit.
            Journal of Structural Biology, Vol. 204, No. 2, 11.2018, p. 215-227.

        :param particle_size: Particle size (pixels) is required.
            Remaining parameters generally have defaults based on particle size.
        :param min_particle_size:
        :param max_particle_size:
        :param query_image_size:
        :param minimum_overlap_amount:
        :param tau1: SVM parameter
        :param tau2: SVM parameter
        :param container_size: Defaults 450
        :param output_dir: Optionally specify output_dir, defaults to local dir.
        :param model: One of svm/gaussian_mixture/gaussian_naive_bayes/xgboost/thunder_svm
        :param model_opts: Optional dictionary of svm model options. Defaults to None.
        :param n_processes: Concurrent processes to spawn.
            May improve performance on very large machines.
            Otherwise use default of 1.
        """

        self.particle_size = particle_size

        self.max_particle_size = max_particle_size or self.particle_size * 2

        self.min_particle_size = min_particle_size or self.particle_size // 4

        if self.max_particle_size < self.min_particle_size:
            raise RuntimeError("max_particle_size must be >= min_particle_size")

        self.minimum_overlap_amount = minimum_overlap_amount or self.particle_size // 10

        self.container_size = container_size
        self.n_processes = int(n_processes)
        self.output_dir = output_dir

        if query_image_size is None:
            query_image_size = np.round(self.particle_size * 2 / 3)
            query_image_size -= query_image_size % 4
            query_image_size = int(query_image_size)

        self.query_image_size = query_image_size

        # q_box is the query box (or "container") size
        self.q_box = (4000**2) / (self.query_image_size**2) * 4
        # tau1 and tau2 correspond to particle and noise windows respectively
        self.tau1 = tau1 or int(self.q_box * 0.03)
        self.tau2 = tau2 or int(self.q_box * 0.3)

        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.verify_input_values()

        self.model = model
        self.model_opts = model_opts

        # MRC processing config
        # Margins to discard from any processed .mrc file
        # TODO: Margins are asymmetrical to conform to old behavior - fix going forward
        self.mrc_margin_left = mrc_margin_left
        self.mrc_margin_right = mrc_margin_right
        self.mrc_margin_top = mrc_margin_top
        self.mrc_margin_bottom = mrc_margin_bottom
        self.mrc_shrink_factor = mrc_shrink_factor
        self.mrc_gauss_filter_size = mrc_gauss_filter_size
        self.mrc_gauss_filter_sigma = mrc_gauss_filter_sigma
        self.response_thresh_norm_factor = response_thresh_norm_factor
        self.conv_map_nthreads = conv_map_nthreads

        # Assigned when process_micrograph_centers is run
        self.picker = dict()

    def verify_input_values(self):
        assert (
            1 <= self.max_particle_size <= 3000
        ), "Max particle size must be in range [1, 3000]!"

        assert (
            1 <= self.query_image_size <= 3000
        ), "Query image size must be in range [1, 3000]!"

        assert (
            5 <= self.particle_size < 3000
        ), "Particle size must be in range [5, 3000]!"

        assert (
            1 <= self.min_particle_size < 3000
        ), "Min particle size must be in range [1, 3000]!"

        max_tau1_value = self.q_box
        assert (
            0 <= self.tau1 <= max_tau1_value
        ), f"tau1 must be a in range [0, {max_tau1_value}]!"

        max_tau2_value = self.q_box
        assert (
            0 <= self.tau2 <= max_tau2_value
        ), f"tau2 must be in range [0, {max_tau2_value}]!"

        assert (
            0 <= self.minimum_overlap_amount <= 3000
        ), "overlap must be in range [0, 3000]!"

        # max container_size condition is (container_size_max * 2 + 200 > 4000), which is 1900
        assert (
            self.particle_size <= self.container_size <= 1900
        ), f"Container size must be within range [{self.particle_size}, 1900]!"

        assert (
            self.particle_size >= self.query_image_size
        ), f"Particle size ({self.particle_size}) must exceed query image size ({self.query_image_size})!"

    def process_folder(self, folder, create_jpg=False):
        # Gather matches for multiple possible file types.
        filenames = []
        for ext in Image.extensions:
            filenames.extend(glob(f"{folder}/*{ext}"))
        logger.info(f"converting {len(filenames)} input files")
        logger.info(f"launching {self.n_processes} processes")

        pbar = tqdm(total=len(filenames))
        with futures.ProcessPoolExecutor(self.n_processes) as executor:
            to_do = []
            for filename in filenames:
                future = executor.submit(
                    self.process_micrograph, filename, False, create_jpg
                )
                to_do.append(future)

            for future in futures.as_completed(to_do):
                # Retrieve (None) result, since this operation re-raises Exceptions, if any.
                _ = future.result()
                pbar.update(1)
        pbar.close()

    def process_micrograph_centers(
        self,
        filepath,
    ):
        """
        Process micrograph at `filepath`, returning `centers`.

        :param filepath: MRC filepath
        :return: `centers`
        """

        # Note we assign picker now so we can access it from process_micrograph_plots
        self.picker[filepath] = picker = Picker(
            self.particle_size,
            self.max_particle_size,
            self.min_particle_size,
            self.query_image_size,
            self.tau1,
            self.tau2,
            self.minimum_overlap_amount,
            self.container_size,
            filepath,
            self.output_dir,
            model=self.model,
            model_opts=self.model_opts,
            mrc_margin_left=self.mrc_margin_left,
            mrc_margin_right=self.mrc_margin_right,
            mrc_margin_top=self.mrc_margin_top,
            mrc_margin_bottom=self.mrc_margin_bottom,
            mrc_shrink_factor=self.mrc_shrink_factor,
            mrc_gauss_filter_size=self.mrc_gauss_filter_size,
            mrc_gauss_filter_sigma=self.mrc_gauss_filter_sigma,
            response_thresh_norm_factor=self.response_thresh_norm_factor,
            conv_map_nthreads=self.conv_map_nthreads,
        )

        logger.info("Computing scores for query images")
        score = (
            picker.query_score()
        )  # compute score using normalized cross-correlations

        while True:
            logger.info(f"Running svm with tau1={picker.tau1}, tau2={picker.tau2}")
            # train SVM classifier and classify all windows in micrograph
            segmentation = picker.run_svm(score)

            # If all windows are classified identically, update tau_1 or tau_2
            if np.all(segmentation):
                picker.tau2 += 500
            elif not np.any(segmentation):
                picker.tau1 += 500
            else:
                break

        logger.info("Discarding suspected artifacts")
        segmentation = picker.morphology_ops(segmentation)

        logger.info("Getting particle centers")
        centers = picker.extract_particles(segmentation)

        return centers

    def process_micrograph_plots(self, filepath, centers, create_jpg=False):
        """
        Takes in `centers`, returns corresponding `particle_image`.

        Optionally writes jpg to disk.

        :param filepath: mrc filepath
        :param centers: Particle centers, typically from `process_micrograph_centers`.
        :param create_jpg: Optionally writes JPG file with picked particles.
        :return: `particle_image`
        """

        # Note that we only use filepath to identify the specific picker,
        #   since Apple instance may be shared.
        picker = self.picker.get(filepath)
        if picker is None:
            raise RuntimeError(
                "Must run `process_micrograph_centers` for this filepath first"
            )

        particle_image = self.particle_image(
            picker.original_im, picker.particle_size, centers
        )

        output_dir = self.output_dir

        # If user wants to create_jpg, but has None output_dir,
        #  warn them and use local dir instead of crashing.
        if create_jpg and output_dir is None:
            output_dir = "."
            logger.warn(
                "`create_jpg` called with (default) output_dir=None."
                f'  Using output_dir = "{output_dir}".'
                "  To avoid this warning, set `output_dir` when creating JPG files."
            )

        if create_jpg:
            # Create the image
            image_out = Image.fromarray((particle_image).astype(np.uint8))

            # Construct filename
            base = os.path.splitext(os.path.basename(picker.filename))[0]
            filename_out = os.path.join(output_dir, f"{base}_result.jpg")

            # Save the image
            image_out.save(filename_out)

        return particle_image

    def process_micrograph(self, filepath, create_jpg=False):
        """
        Process micrograph at `filepath`, returning (`centers`, `particle_image`).

        :param filepath: mrc filepath
        :param create_jpg: Optionally writes JPG file identifying picked particles.
        :return: (`centers`, `particle_image`)
        """

        centers = self.process_micrograph_centers(filepath)
        particle_image = self.process_micrograph_plots(filepath, centers, create_jpg)
        return centers, particle_image

    def particle_image(self, micro_img, particle_size, centers):
        """
        Return a numpy array representing the picked centers on a micrograph,
        suitable for rendering in a jupyter notebook or saving as a jpg etc.

        :param micro_img: The micrograph image as a numpy array
        :param particle_size: Particle size of picked particles
        :param centers: Picked centers for micrograph.
        :return: A numpy array with picked centers displayed as rectangles
        """

        micro_img = micro_img - np.amin(
            np.reshape(micro_img, (np.prod(micro_img.shape)))
        )
        picks = np.ones(micro_img.shape)
        for i in range(0, centers.shape[0]):
            y = int(centers[i, 1])
            x = int(centers[i, 0])
            d = int(np.floor(particle_size))
            picks[y - d : y - d + 5, x - d : x + d] = 0
            picks[y + d : y + d + 5, x - d : x + d] = 0
            picks[y - d : y + d, x - d : x - d + 5] = 0
            picks[y - d : y + d, x + d : x + d + 5] = 0

        return np.multiply(micro_img, picks)
