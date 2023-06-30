import numpy as np

from aspire.image import Image
from aspire.source import Simulation
from aspire.source.image import _ImageAccessor


class MicrographSource:
    def __init__(
        self,
        micrograph_size=4096,
        particle_box_size=300,
        micrograph_count=1,
        particles_per_micrograph=10,
        unique_filters=None,
        filter_indices=None,
        dtype=None,
        seed=0,
        noise_adder=None,
        boundary=None,
        interparticle_distance=None,
    ):
        """
        A cryo-EM MicrographSource object that supplies micrographs.
        :param micrograph_size: Size of micrograph
        :param particle_box_size: Image size for picked particle images
        :param micrograph_count: Number of micrographs to generate (integer)
        :param particles_per_micrograph: The amount of particles generated for each micrograph
        :param unique_filters: A list of Filter objects to be applied to micrograph
        :param dtype: dtype for picked particle Simulation
        :param seed: Random seed
        :param noise_adder: Optionally append instance of NoiseAdder to generation pipeline
        :param boundary: Optionally set boundaries for particle centers. Defaults to particle_box_size // 2, positive values move the boundary inward
        :param interparticle_distance: Optionally set minimum distance between particle centers. Defaults to particle_box_size * sqrt(2) to avoid collisions
        :return: A MicrographSource object
        """
        self.seed = seed
        np.random.seed(self.seed)

        self.micrograph_size = micrograph_size
        self.micrograph_count = micrograph_count
        self.particle_box_size = particle_box_size
        self.particles_per_micrograph = particles_per_micrograph
        self.total_particle_count = (
            self.micrograph_count * self.particles_per_micrograph
        )
        if unique_filters is None:
            unique_filters = []
        else:
            self.unique_filters = unique_filters
        self.dtype = dtype

        self.noise_adder = noise_adder

        if boundary is None:
            self.boundary = self.particle_box_size // 2
        else:
            if (
                boundary < (0 - particle_box_size // 2)
                or boundary > self.micrograph_size // 2
            ):
                raise RuntimeError("Illegal boundary value.")
            self.boundary = boundary

        if interparticle_distance is None:
            self.interparticle_distance = np.sqrt(2) * self.particle_box_size
        else:
            self.interparticle_distance = interparticle_distance

        self.simulation = Simulation(
            L=self.particle_box_size,
            n=self.total_particle_count,
            offsets=0,
            C=1,
            dtype=self.dtype,
            seed=self.seed,
            unique_filters=self.unique_filters,
        )

        self.centers = np.zeros(
            (self.micrograph_count, self.particles_per_micrograph, 2), dtype=int
        )
        for i in range(self.micrograph_count):
            self.centers[i] = self._create_centers()

        self._clean_micrographs_accessor = _ImageAccessor(
            self._clean_micrographs, self.micrograph_count
        )
        self._micrographs_accessor = _ImageAccessor(
            self._micrographs, self.micrograph_count
        )

        self.images = _ImageAccessor(self._images, self.total_particle_count)

    def not_colliding(self, x1, y1, x2, y2, distance):
        return np.hypot(x1 - x2, y1 - y2) > distance

    def _create_centers(self):
        # initilize root2 for calculating sqrt(2) for Euclidean distance, and max_counts for attempts at randomizing points
        max_counts = 2500

        centers = np.ones((self.particles_per_micrograph, 2)) * -9999
        for i in range(self.particles_per_micrograph):
            # Initialize center coordinates and attempt count
            center_x, center_y, count = 0, 0, 0
            while count < max_counts:
                # Generate random coordinate within bounds
                center_x, center_y = self._generate_center()

                good_center = True
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
                if good_center:
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
            x - self.particle_box_size // 2 > self.boundary
            and x + self.particle_box_size // 2 < self.micrograph_size - self.boundary
            and y - self.particle_box_size // 2 > self.boundary
            and y + self.particle_box_size // 2 < self.micrograph_size - self.boundary
        )

    def __len__(self):
        """ """
        return self.micrograph_count

    @property
    def clean_micrographs(self):
        return self._clean_micrographs_accessor

    @property
    def micrographs(self):
        return self._micrographs_accessor

    def _micrographs(self, indices):
        micrographs = self._clean_micrographs(indices)
        if self.noise_adder:
            micrographs = self.noise_adder.forward(micrographs)
        return micrographs

    def _clean_micrographs(self, indices):
        # Initialize empty micrograph
        clean_micrograph = np.zeros((self.micrograph_size, self.micrograph_size))
        pad = self.particle_box_size
        clean_micrograph = np.pad(
            clean_micrograph, pad, "constant", constant_values=(0)
        )
        # Get centers
        centers = self.centers[indices][0]
        parity = self.particle_box_size % 2
        for i in range(centers.shape[0]):
            image = self.simulation.clean_images[
                self.particles_per_micrograph * indices + i
            ].asnumpy()
            x_left = centers[i][0] - self.particle_box_size // 2 + pad
            x_right = centers[i][0] + self.particle_box_size // 2 + parity + pad
            y_left = centers[i][1] - self.particle_box_size // 2 + pad
            y_right = centers[i][1] + self.particle_box_size // 2 + parity + pad
            clean_micrograph[x_left:x_right, y_left:y_right] = (
                clean_micrograph[x_left:x_right, y_left:y_right] - image
            )

        clean_micrograph = clean_micrograph[
            pad : self.micrograph_size + parity + pad,
            pad : self.micrograph_size + parity + pad,
        ]
        return Image(clean_micrograph)

    def _images(self, indices):
        return self.simulation.images[indices].asnumpy()[0]

    def get_micrograph(self, particle_id):
        """
        :param particle_id: Global ID of the particle
        """
        if particle_id >= self.total_particle_count or particle_id < 0:
            raise RuntimeError("ID out of bounds")
        return divmod(particle_id, self.particles_per_micrograph)

    def get_particle(self, micrograph_id, particle_id):
        """
        :param micrograph_id: ID of the microgram
        :param particle_id: Local ID of the particle
        """
        if micrograph_id >= self.micrograph_count or micrograph_id < 0:
            raise RuntimeError("Out of bounds for micrograph")
        if particle_id >= self.particles_per_micrograph or particle_id < 0:
            raise RuntimeError("Out of bounds for particle")
        return micrograph_id * self.particles_per_micrograph + particle_id
