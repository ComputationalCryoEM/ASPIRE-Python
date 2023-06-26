import numpy as np

from aspire.image import Image
from aspire.source import Simulation
from aspire.source.image import _ImageAccessor


class MicrographSource:
    def __init__(
        self,
        micrograph_size=200,
        L=300,
        n=1,
        ppm=100,
        unique_filters=None,
        dtype=None,
        seed=0,
        noise_adder=None,
        collisions=False,
        boundaries=True,
    ):
        """
        A cryo-EM MicrographSource object that supplies micrographs.
        :param micrograph_size: Size of micrograph
        :param L: Image size for picked particle images
        :param n: Number of micrographs to generate (integer)
        :param ppm: Particles per micrograph
        :param unique_filters: A list of Filter objects to be applied to micrograph
        :param dtype: dtype for picked particle Simulation
        :param seed: Random seed
        :param noise_adder: Optionally append instance of NoiseAdder to generation pipeline
        :param collisions: Optionally allow collisions
        :param boundaries: Optionally allow boundaries
        :return: A MicrographSource object
        """
        self.micrograph_size = micrograph_size
        self.n = n
        self.L = L
        self.ppm = ppm
        self.unique_filters = unique_filters
        self.dtype = dtype
        self.seed = seed
        self.noise_adder = noise_adder
        self.collisions = collisions
        self.boundaries = boundaries
        self.simulation = Simulation(
            L=self.L,
            n=self.n * self.ppm,
            offsets=0,
            C=1,
            dtype=self.dtype,
            seed=self.seed,
        )

        self.centers = np.zeros((self.n, self.ppm, 2), dtype=int)
        for i in range(n):
            self.centers[i] = self._create_centers()

        self._clean_micrographs_accessor = _ImageAccessor(
            self._clean_micrographs, self.n
        )
        self._micrographs_accessor = _ImageAccessor(self._micrographs, self.n)
        self.clean_micrographs = self.clean_micrographs()
        self.micrographs = self.micrographs()

        self.images = _ImageAccessor(self._images, self.n * self.ppm)
        # np.zeros((self.n, self.ppm, self.L, self.L))

    def not_colliding(self, x1, y1, x2, y2, distance):
        return np.sqrt(np.square(x1 - x2) + np.square(y1 - y2)) > distance

    def _create_centers(self) -> list((int, int)):
        # initilize root2 for calculating sqrt(2) for Euclidean distance, and max_counts for attempts at randomizing points
        root2 = np.sqrt(2)
        collision_distance = self.L * root2
        if self.collisions:
            collision_distance = 0
        max_counts = 2500

        centers = np.zeros((self.ppm, 2))
        for i in range(self.ppm):
            # Initialize center coordinates and attempt count
            center_x, center_y, count = 0, 0, 0
            while count < max_counts:
                # Generate random coordinate within bounds
                x_bound = (
                    self.micrograph_size - (2 * self.L)
                ) * np.random.rand() + self.L
                y_bound = (
                    self.micrograph_size - (2 * self.L)
                ) * np.random.rand() + self.L
                if self.boundaries is False:
                    x_bound = (
                        self.micrograph_size + (self.L)
                    ) * np.random.rand() - self.L
                    y_bound = (
                        self.micrograph_size + (self.L)
                    ) * np.random.rand() - self.L
                center_x, center_y = int(x_bound), int(y_bound)

                collisions = False
                # Check if new center is in the radial bounds of an existing center, make collisions var true if so.
                for j in range(i):
                    if (
                        self.not_colliding(
                            centers[j][0],
                            centers[j][1],
                            center_x,
                            center_y,
                            collision_distance,
                        )
                        is False
                    ):
                        collisions = True

                # If there are no collisions, add new center and increase center count
                if collisions is False:
                    centers[i] = np.array([center_x, center_y])
                    count += max_counts
                count += 1
            # if count == max_counts:
            # raise RuntimeError('Error: Too many particles requested')
        return centers

    def __len__(self):
        """ """
        return self.n

    def clean_micrographs(self):
        return self._clean_micrographs_accessor

    def micrographs(self):
        return self._micrographs_accessor

    def _micrographs(self, indices):
        micrograph = self._clean_micrographs(indices)
        if self.noise_adder:
            micrograph = self.noise_adder.forward(micrograph)
        return micrograph

    def _clean_micrographs(self, indices):
        # Initialize empty micrograph
        clean_micrograph = np.zeros((self.micrograph_size, self.micrograph_size))
        # Get centers
        centers = self.centers[indices][0]
        for i in range(centers.shape[0]):
            image = self.simulation.clean_images[self.ppm * indices + i].asnumpy()
            clean_micrograph[
                centers[i][0] - self.L // 2 : centers[i][0] + self.L // 2,
                centers[i][1] - self.L // 2 : centers[i][1] + self.L // 2,
            ] = (
                clean_micrograph[
                    centers[i][0] - self.L // 2 : centers[i][0] + self.L // 2,
                    centers[i][1] - self.L // 2 : centers[i][1] + self.L // 2,
                ]
                + image
            )
        return Image(clean_micrograph)

    def _images(self, indices):
        return self.simulation.images[indices].asnumpy()[0]

    def get_micrograph(self, id):
        """
        :param id: Global ID of the particle
        """
        micrograph_id = id // self.ppm
        particle_id = id % self.ppm
        return (micrograph_id, particle_id)

    def get_particle(self, micrograph_id, particle_id):
        """
        :param micrograph_id: ID of the microgram
        :param particle_id: Local ID of the particle
        """
        return micrograph_id * self.ppm + particle_id
