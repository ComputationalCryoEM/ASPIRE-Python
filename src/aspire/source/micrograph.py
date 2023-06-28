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
        boundary=None,
        interparticle_distance=None,
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
        :param boundary: Optionally set boundaries
        :param interparticle_distance: Optionally set distance between particles
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
        if boundary is None:
            self.boundary = self.L // 2
        else:
            if boundary < (0 - L // 2) or boundary > self.micrograph_size // 2:
                raise RuntimeError("Illegal boundary value.")
            self.boundary = boundary
        if interparticle_distance is None:
            self.interparticle_distance = np.sqrt(2) * self.L
        else:
            self.interparticle_distance = interparticle_distance
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

    def not_colliding(self, x1, y1, x2, y2, distance):
        return np.sqrt(np.square(x1 - x2) + np.square(y1 - y2)) > distance

    def _create_centers(self):
        # initilize root2 for calculating sqrt(2) for Euclidean distance, and max_counts for attempts at randomizing points
        max_counts = 2500

        centers = np.ones((self.ppm, 2)) * -9999
        for i in range(self.ppm):
            # Initialize center coordinates and attempt count
            center_x, center_y, count = 0, 0, 0
            while count < max_counts:
                # Generate random coordinate within bounds
                center_x, center_y = self._generate_center()

                good_center = True
                # Check if new center is in the radial bounds of an existing center, make collisions var true if so.
                for j in range(i):
                    if not self.not_colliding(
                        centers[j][0],
                        centers[j][1],
                        center_x,
                        center_y,
                        self.interparticle_distance,
                    ) or not self._in_boundary(center_x, center_y):
                        good_center = False

                # If there are no collisions or collisions are allowed, add new center and increase center count
                if self.collisions or good_center:
                    centers[i] = np.array([center_x, center_y])
                    count += max_counts
                count += 1
        # Check for zeroes
        zero_count = 0
        for center in centers:
            if center[0] == -9999 and center[1] == -9999:
                zero_count += 1
        if zero_count > 0:
            raise RuntimeError("Not enough centers generated.")
        return centers

    def _generate_center(self):
        parity = (self.boundary + 1) % 2
        x = (
            (self.micrograph_size - 2 * self.boundary - parity) * np.random.rand()
            + self.boundary
            + parity
        )
        y = (
            (self.micrograph_size - 2 * self.boundary - parity) * np.random.rand()
            + self.boundary
            + parity
        )
        return (int(x), int(y))

    def _in_boundary(self, x, y):
        return (
            x - self.L // 2 > self.boundary
            and x + self.L // 2 < self.micrograph_size - self.boundary
            and y - self.L // 2 > self.boundary
            and y + self.L // 2 < self.micrograph_size - self.boundary
        )

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
        pad = 0
        if self.boundary < 0:
            pad = self.L
            clean_micrograph = np.pad(
                clean_micrograph, pad, "constant", constant_values=(0)
            )
        # Get centers
        centers = self.centers[indices][0]
        parity = self.L % 2
        for i in range(centers.shape[0]):
            image = self.simulation.clean_images[self.ppm * indices + i].asnumpy()
            x_left = centers[i][0] - self.L // 2 + pad
            x_right = centers[i][0] + self.L // 2 + parity + pad
            y_left = centers[i][1] - self.L // 2 + pad
            y_right = centers[i][1] + self.L // 2 + parity + pad
            clean_micrograph[x_left:x_right, y_left:y_right] = (
                clean_micrograph[x_left:x_right, y_left:y_right] - image
            )

        if self.boundary < 0:
            clean_micrograph = clean_micrograph[
                pad : self.micrograph_size + parity + pad,
                pad : self.micrograph_size + parity + pad,
            ]
        return Image(clean_micrograph)

    def _images(self, indices):
        return self.simulation.images[indices].asnumpy()[0]

    def get_micrograph(self, id):
        """
        :param id: Global ID of the particle
        """
        if id >= self.ppm * self.n:
            raise RuntimeError("ID out of bounds")
        micrograph_id = id // self.ppm
        particle_id = id % self.ppm
        return (micrograph_id, particle_id)

    def get_particle(self, micrograph_id, particle_id):
        """
        :param micrograph_id: ID of the microgram
        :param particle_id: Local ID of the particle
        """
        if micrograph_id >= self.n or micrograph_id < 0:
            raise RuntimeError("Out of bounds for micrograph")
        if particle_id >= self.ppm or particle_id < 0:
            raise RuntimeError("Out of bounds for particle")
        return micrograph_id * self.ppm + particle_id
