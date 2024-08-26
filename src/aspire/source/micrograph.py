import logging
import os
from abc import ABC, abstractmethod
from glob import glob
from pathlib import Path

import numpy as np

from aspire.image import Image
from aspire.source import Simulation
from aspire.source.image import _ImageAccessor
from aspire.storage import StarFile
from aspire.utils import Random, grid_2d
from aspire.volume import Volume

logger = logging.getLogger(__name__)


class MicrographSource(ABC):
    def __init__(self, micrograph_count, micrograph_size, dtype, pixel_size=None):
        """ """
        self.micrograph_count = int(micrograph_count)
        self.micrograph_size = int(micrograph_size)
        self.dtype = np.dtype(dtype)
        if pixel_size is not None:
            pixel_size = float(pixel_size)
        self.pixel_size = pixel_size

        self._images_accessor = _ImageAccessor(self._images, self.micrograph_count)

    def __repr__(self):
        """
        String representation.

        :return: Returns a string description of instance.
        """
        return f"{self.__class__.__name__} with {self.micrograph_count} {self.dtype.name} micrographs of size {self.micrograph_size}x{self.micrograph_size}"

    def __len__(self):
        """
        Returns the number of micrographs.

        :return: Returns the micrograph count
        """
        return self.micrograph_count

    def save(self, path, name_prefix="micrograph", overwrite=True):
        """
        Save micrographs to `path`.

        `path` should be a directory. If it does not exist, ASPIRE will attempt to create it.

        Currently saves micrographs to `.mrc`.

        :param path: Directory to save data.
        :param name_prefix: Optional, name prefix string for micrograph files.
        :param overwrite: Optional, bool. Allow writing to existing directory,
            and overwriting existing files.
        :return: List of saved `.mrc` files.
        """

        # Make dir if does not exist.
        Path(path).mkdir(parents=True, exist_ok=overwrite)

        output_mrc_filenames = []
        for m in range(self.micrograph_count):
            prefix = f"{name_prefix}_{m}"

            # Get the micrograph and save it as an `mrc`
            micrograph = self.images[m].asnumpy()

            mrc_file_path = os.path.join(path, f"{prefix}.mrc")
            Image(micrograph).save(mrc_file_path, overwrite=overwrite)

            output_mrc_filenames.append(mrc_file_path)

        return output_mrc_filenames

    def asnumpy(self):
        """
        Return micrograph data as dense Numpy array.

        :return: Numpy array.
        """
        return self.images[:]._data

    def show(self, *args, **kwargs):
        """
        Helper function to display micrograph. See Image.show().
        """
        Image(self.asnumpy(), pixel_size=self.pixel_size).show(*args, **kwargs)

    @property
    def images(self):
        """
        Returns micrographs.

        :return: An `ImageAccessor` for the noisy micrographs.
        """
        return self._images_accessor

    @abstractmethod
    def _images(self, indices):
        """
        Accesses and returns micrographs.

        :param indices: A 1-D Numpy array of integer indices.
        :return: An `Image` object representing the micrographs for `indices`.
        """


class ArrayMicrographSource(MicrographSource):
    def __init__(self, micrographs, dtype=None, pixel_size=None):
        """
        Instantiate a `MicrographSource` with `micrographs`.

        :param micrographs: Numpy array (count, L, L)
            where  `L` represents the size in pixels.
        :param dtype: Data type of returned `MicrographSource`.
            Default of `None` will infer dtype from the array.
            Explicitly setting dtype will attempt a cast.
            Currently only `float32` and `float64` are supported.
            Note, due to limitations of common MRC implementations,
            saving is limited to single precision.
        :param pixel_size: Pixel size of the images in angstroms, default `None`.
        """

        # Check micrographs is an array
        if not isinstance(micrographs, np.ndarray):
            raise NotImplementedError(
                f"{self.__class__.__name__} not implemented for {type(micrographs)}."
            )

        # Ensure dimensions
        if micrographs.ndim == 2:
            micrographs = micrographs[None, :, :]

        if micrographs.ndim != 3 or (micrographs.shape[-2] != micrographs.shape[-1]):
            raise NotImplementedError(
                f"Incompatible `micrographs` shape {micrographs.shape}, expects (count, L, L)"
            )

        super().__init__(
            micrograph_count=micrographs.shape[0],
            micrograph_size=micrographs.shape[-1],
            dtype=dtype or micrographs.dtype,
            pixel_size=pixel_size,
        )

        # We're already backed by an array, access it directly.
        self._data = micrographs.astype(self.dtype, copy=False)

    def _images(self, indices):
        """
        Accesses and returns micrographs from a Numpy array.

        :param indices: A 1-D Numpy array of integer indices.
        :return: An array backed `MicrographSource` object representing the micrographs for `indices`.
        """
        return Image(self._data[indices], pixel_size=self.pixel_size)


class DiskMicrographSource(MicrographSource):
    def __init__(self, micrographs_path, dtype=None, pixel_size=None):
        """
        Instantiate a `MicrographSource` with `micrographs_path`.

        :param micrographs_path: Path string representing directory of file,
            or list of file path strings.
        :param dtype: Data type of returned `MicrographSource`.
            Default of `None` will attempt to infer dtype from the first micrograph.
            Explicitly setting dtype will attempt a cast when returning micrographs.
            Currently only `float32` and `float64` are supported.
            Note, due to limitations of common MRC implementations,
            saving is limited to single precision.
        """

        if not self._is_path_like_input(micrographs_path):
            raise NotImplementedError(
                f"{self.__class__.__name__} not implemented for {type(micrographs_path)}."
            )

        if hasattr(micrographs_path, "__len__") and len(micrographs_path) == 0:
            raise RuntimeError(
                f"Must supply non-empty `micrographs` argument, received {micrographs_path}."
            )

        if isinstance(micrographs_path, list):
            # Explicit list
            self.micrograph_files = micrographs_path
        else:
            # `str` cast will accommodate string and Path objects
            self.micrograph_files = self._glob_files(str(micrographs_path))

        # Load the first micrograph to infer shape/type
        # Size will be checked during on-the-fly loading of subsequent micrographs.
        micrograph0 = Image.load(self.micrograph_files[0])
        if micrograph0.pixel_size is not None and micrograph0.pixel_size != pixel_size:
            raise ValueError(
                f"Mismatched pixel size. {micrograph0.pixel_size} angstroms defined in {self.micrograph_files[0]}, but provided {pixel_size} angstroms."
            )

        super().__init__(
            micrograph_count=len(self.micrograph_files),
            micrograph_size=micrograph0.resolution,
            dtype=dtype or micrograph0.dtype,
            pixel_size=pixel_size,
        )

        # Prepare accessor to load files from disk on the fly.
        self._images_accessor = _ImageAccessor(self._images, self.micrograph_count)

    @staticmethod
    def _is_path_like_input(p):
        """
        Utility to return whether `p` should be treated as a path to micrograph(s).

        :param p: Item to test.
        :return: Boolean.
        """
        return any([isinstance(p, list), isinstance(p, str), isinstance(p, Path)])

    def _glob_files(self, file_path):
        """
        Helper utility to glob for matching micrograph files at `file_path`.

        Will match on any image matching `Image.extensions` in `file_path`.

        :return: List of files.
        """

        files = []
        # Determine if directory or single file.
        if os.path.isdir(file_path):
            # Add all loadable images in the directory.
            for ext in Image.extensions:
                files.extend(sorted(glob(os.path.join(file_path, f"*{ext}"))))
            # Ensure we have not failed to find any micrographs in the provided directory.
            if len(files) == 0:
                raise RuntimeError(f"No suitable files were found at {file_path}.")
        elif os.path.isfile(file_path):
            # Add this file.
            files.append(file_path)
        else:
            raise RuntimeError(
                f"File path {file_path} does not appear to be a directory or a file"
            )

        return files

    def _images(self, indices):
        """
        Accesses and returns micrographs from disk.

        :param indices: A 1-D Numpy array of integer indices.
        :return: An `Image` object representing the micrographs for `indices`.
        """
        # Initialize empty result
        n_micrographs = len(indices)
        micrographs = np.empty(
            (n_micrographs, self.micrograph_size, self.micrograph_size),
            dtype=self.dtype,
        )
        for i, ind in enumerate(indices):
            # Load the micrograph image from file
            micrograph = Image.load(self.micrograph_files[ind])
            # Assert size
            if micrograph.resolution != self.micrograph_size:
                raise NotImplementedError(
                    f"Micrograph {ind} has inconsistent shape {micrograph.shape},"
                    f" expected {(self.micrograph_size, self.micrograph_size)}."
                )
            # Assign to array, implicitly performs casting to dtype
            micrographs[i] = micrograph.asnumpy()
            # Assert pixel_size
            if (
                micrograph.pixel_size is not None
                and micrograph.pixel_size != self.pixel_size
            ):
                raise ValueError(
                    f"Mismatched pixel size. {micrograph.pixel_size} angstroms defined in {self.micrograph_files[ind]}, but provided {self.pixel_size} angstroms."
                )

        return Image(micrographs, pixel_size=self.pixel_size)


class MicrographSimulation(MicrographSource):
    def __init__(
        self,
        volume,
        micrograph_size=4096,
        micrograph_count=1,
        particles_per_micrograph=100,
        particle_amplitudes=None,
        projection_angles=None,
        seed=None,
        ctf_filters=None,
        noise_adder=None,
        boundary=None,
        interparticle_distance=None,
    ):
        """
        A cryo-EM MicrographSimulation object that supplies micrographs.

        `dtype` and `particle_box_size` are inferred from `volume`, where `dtype` is the data type of the micrographs and `particle_box_size` is the size of the particle images.

        :param volume: `Volume` instance to be used in `Simulation`.
             An `(L,L,L)` `Volume` will generate `(L,L)` particle images.
        :param micrograph_size: Size of micrograph in pixels, defaults to 4096.
        :param micrograph_count: Number of micrographs to generate (integer). Defaults to 1.
        :param particles_per_micrograph: The amount of particles generated for each micrograph. Defaults to 10.
        :param particle_amplitudes: Optional, amplitudes to pass to `Simulation`.
             Default `None` uses `Simulation` defaults.
             When provided must be array with size `particles_per_micrograph * micrograph_count`.
        :param projection_angles: Optional, projection rotation angles to pass to `Simulation`.
             Default `None` uses `Simulation` defaults.
             When provided must have shape `(particles_per_micrograph * micrograph_count, 3)`.

        :param seed: Random seed.
        :param noise_adder: Append instance of NoiseAdder to generation pipeline.
        :param ctf_filters: Optional list of `Filter` objects to apply to particles.
            This list should be 1, n_micrographs, or particles_per_micrograph * micrograph_count.
            These will apply filters to all particles, per-micrograph, or per-particle respectively.
            Default `None` will not apply any additional filters.
        :param boundary: Set boundaries for particle centers, positive values move the boundary inward from the edge of the micrograph. Defaults to half of the particle size (particle_box_size // 2).
        :param interparticle_distance: Set minimum distance between particle centers, in pixels. Defaults to particle_box_size.
        :return: A MicrographSimulation object.
        """
        if not isinstance(volume, Volume):
            raise TypeError("`volume` should be of type `Volume`.")
        self.volume = volume

        self.seed = seed

        super().__init__(
            micrograph_count=micrograph_count,
            micrograph_size=micrograph_size,
            dtype=self.volume.dtype,
        )

        self.particle_box_size = volume.resolution  # L
        self.particles_per_micrograph = particles_per_micrograph
        self.total_particle_count = (
            self.micrograph_count * self.particles_per_micrograph
        )
        self.dtype = volume.dtype

        self.noise_adder = noise_adder

        if self.particle_box_size > micrograph_size:
            raise ValueError(
                "The micrograph size must be larger or equal to the `particle_box_size`."
            )

        if particle_amplitudes is not None:
            if (
                not isinstance(particle_amplitudes, int)
                and len(particle_amplitudes) != self.total_particle_count
            ):
                raise RuntimeError(
                    f"`particle_amplitudes` must be an `int` or length {self.total_particle_count}."
                )
        self.particle_amplitudes = particle_amplitudes

        if projection_angles is not None and projection_angles.shape != (
            self.total_particle_count,
            3,
        ):
            raise RuntimeError(
                f"`projection_angles.shape` {projection_angles.shape} != (total_particle_count,3) {self.total_particle_count, 3}"
            )
        self.projection_angles = projection_angles

        self.filter_indices = None
        if ctf_filters is not None:
            acceptable_lens = [1, self.micrograph_count, self.total_particle_count]
            if (
                not isinstance(ctf_filters, list)
                or len(ctf_filters) not in acceptable_lens
            ):
                raise TypeError(
                    f"`ctf_filters` expects a list of len {acceptable_lens[0]},"
                    f" {acceptable_lens[1]}, or {acceptable_lens[2]}."
                )

            # Generate explicit filter indices (zero-indexed).
            self.filter_indices = np.arange(self.total_particle_count) % len(
                ctf_filters
            )

        self.ctf_filters = ctf_filters

        self.simulation = Simulation(
            n=self.total_particle_count,
            vols=self.volume,
            L=self.particle_box_size,
            offsets=0,
            amplitudes=self.particle_amplitudes,
            angles=self.projection_angles,
            unique_filters=ctf_filters,
            filter_indices=self.filter_indices,
            dtype=self.dtype,
            seed=self.seed,
        )

        if boundary is None:
            self.boundary = self.particle_box_size // 2
        else:
            if (
                boundary < (-self.particle_box_size // 2)
                or boundary > self.micrograph_size // 2
            ):
                raise ValueError("Illegal boundary value.")
            self.boundary = boundary

        if interparticle_distance is None:
            self.interparticle_distance = self.particle_box_size
        else:
            self.interparticle_distance = interparticle_distance

        # Create the radial mask for each center
        g2d = grid_2d(int(2 * self.interparticle_distance), normalized=False)
        radial_mask = g2d["r"] <= self.interparticle_distance
        self.grid_x = g2d["x"].astype(int)[radial_mask]
        self.grid_y = g2d["y"].astype(int)[radial_mask]

        # Calculate the proper padding for the micrograph borders and create the micrograph mask
        self.pad = int(max(self.particle_box_size, self.interparticle_distance))
        self._mask_boundary = self.pad + self.boundary
        self._set_mask()

        # Create the centers
        self.centers = np.zeros(
            (self.micrograph_count, self.particles_per_micrograph, 2), dtype=int
        )

        # Set pass threshold here
        self._fail_limit = 0.1 * self.micrograph_count
        self._fail_count = 0

        with Random(seed=self.seed) as _:
            for i in range(self.micrograph_count):
                self.centers[i] = self._create_centers(i)

        self._clean_images_accessor = _ImageAccessor(
            self._clean_images, self.micrograph_count
        )

        self._images_accessor = _ImageAccessor(self._images, self.micrograph_count)

    def _create_centers(self, micrograph_index):
        """
        Creates centers for the given micrograph if the fail threshold isn't met.

        param micrograph_index: The ID of the micrograph.
        return: A Numpy array containing the generated centers.
        """
        while self._fail_count < self._fail_limit:
            try:
                self._set_mask()
                centers = np.zeros((self.particles_per_micrograph, 2))
                for i in range(self.particles_per_micrograph):
                    x, y = self._generate_center()
                    centers[i] = np.array([x, y])
                return centers
            except RuntimeError:
                self._fail_count += 1
        else:
            raise RuntimeError(
                "Micrograph generation failures exceeded limit. This"
                "can happen if constraints are too strict. Consider"
                "adjusting micrograph_size, particles_per_micrograph,"
                " or interparticle_distance."
            )

    def _generate_center(self):
        """
        Helper method to generate centers using the mask.

        return: The x-y coordinate values of the generated center.
        """
        available_centers = np.transpose(np.where(self._mask))

        if available_centers.shape[0] == 0:
            self._fail_count += 1
            raise RuntimeError("Not enough centers generated.")

        random_index = np.random.choice(available_centers.shape[0])
        x, y = available_centers[random_index]
        x_vals = self.grid_x + x
        y_vals = self.grid_y + y
        self._mask[x_vals, y_vals] = False
        return x - self.pad, y - self.pad

    def _set_mask(self):
        """
        Helper method to set the mask.
        """
        self._mask = np.full(
            (
                int(self.micrograph_size + 2 * self.pad),
                int(self.micrograph_size + 2 * self.pad),
            ),
            False,
            dtype=bool,
        )
        self._mask[
            self._mask_boundary : -self._mask_boundary,
            self._mask_boundary : -self._mask_boundary,
        ] = True

    @property
    def clean_images(self):
        """
        Returns the micrographs without any noise.

        :return: An `ImageAccessor` for the unnoisy micrographs.
        """
        return self._clean_images_accessor

    def _images(self, indices):
        """
        Accesses and returns micrographs with any added noise.

        :param indices: A 1-D Numpy array of integer indices.
        :return: An `Image` object representing the noisy micrograph.
        """
        micrographs = self._clean_images(indices)
        if self.noise_adder:
            micrographs = self.noise_adder.forward(micrographs)
        return micrographs

    def _clean_images(self, indices):
        """
        Accesses and returns micrographs without any added noise.

        :param indices: A 1-D Numpy array of integer indices.
        :return: An `Image` object representing the clean micrograph
        """
        # Initialize empty micrograph
        n_micrographs = len(indices)
        clean_micrograph = np.zeros(
            (n_micrographs, self.micrograph_size, self.micrograph_size),
            dtype=self.dtype,
        )
        # Pad the micrograph
        clean_micrograph = np.pad(
            clean_micrograph,
            ((0, 0), (self.pad, self.pad), (self.pad, self.pad)),
            "constant",
            constant_values=(0),
        )
        # Get centers
        parity = self.particle_box_size % 2
        for m in range(n_micrographs):
            global_id = indices[m]
            images = self.simulation.clean_images[self.get_particle_indices(global_id)]
            centers = self.centers[global_id]
            x_lefts = centers[:, 0] - self.particle_box_size // 2 + self.pad
            x_rights = centers[:, 0] + self.particle_box_size // 2 + parity + self.pad
            y_lefts = centers[:, 1] - self.particle_box_size // 2 + self.pad
            y_rights = centers[:, 1] + self.particle_box_size // 2 + parity + self.pad
            # Subtract image from the micrograph using the particle's bounding box and center
            for p in range(self.particles_per_micrograph):
                clean_micrograph[m][
                    x_lefts[p] : x_rights[p], y_lefts[p] : y_rights[p]
                ] = (
                    clean_micrograph[m][
                        x_lefts[p] : x_rights[p], y_lefts[p] : y_rights[p]
                    ]
                    + images[p]
                )
        clean_micrograph = clean_micrograph[
            :,
            self.pad : self.micrograph_size + self.pad,
            self.pad : self.micrograph_size + self.pad,
        ]
        return Image(clean_micrograph, pixel_size=self.pixel_size)

    def get_micrograph_index(self, particle_index):
        """
        Using the global ID of the particle, returns the micrograph ID and the local particle ID.

        :param particle_id: Global ID of the particle.
        :return: The micrograph ID and the local ID of the particle.
        """
        if particle_index >= self.total_particle_count or particle_index < 0:
            raise RuntimeError("Index out of bounds.")
        return divmod(particle_index, self.particles_per_micrograph)

    def get_particle_indices(self, micrograph_index, particle_index=None):
        """
        Using the micrograph ID, returns every global particle ID from
        that micrograph. Returns specific global IDs if the local IDs
        are given.

        :param micrograph_index: ID of the micrograph.
        :param particle_index: Local ID of the particle.
        :return: The global ID of the particle.
        """
        if micrograph_index >= self.micrograph_count or micrograph_index < 0:
            raise RuntimeError("Index out of bounds for micrograph.")
        if particle_index is None:
            return np.arange(
                micrograph_index * self.particles_per_micrograph,
                (micrograph_index + 1) * self.particles_per_micrograph,
            )
        if particle_index >= self.particles_per_micrograph or particle_index < 0:
            raise RuntimeError("Index out of bounds for particle.")
        return micrograph_index * self.particles_per_micrograph + particle_index

    def save(self, path, name_prefix="micrograph", overwrite=True):
        """
        Save micrograph simulation to `path`.

        Currently saves micrographs to `.mrc`.
        Saves simulated centers, projection rotations, and CTF filter parameters
        when applicable.

        :param path: Directory to save data.
        :param name_prefix: Optional, name prefix string for micrograph files.
        :param overwrite: Optional, bool. Allow writing to existing directory,
            and overwriting existing files.
        :return: List of tuples [(`.mrc`, `.star`)..], compatible with CentersCoordinateSource.
        """

        _meta_fields = {
            "ctf": [
                "_rlnVoltage",
                "_rlnDefocusU",
                "_rlnDefocusV",
                "_rlnDefocusAngle",
                "_rlnSphericalAberration",
                "_rlnAmplitudeContrast",
            ],
            "rotations": ["_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"],
        }

        output_mrc_filenames = super().save(
            path=path, name_prefix=name_prefix, overwrite=overwrite
        )

        output_star_filenames = []
        for m in range(self.micrograph_count):
            prefix = f"{name_prefix}_{m}"

            d = dict()

            # Particle to micrograph
            d["_rlnImageName"] = np.full(
                self.particles_per_micrograph, fill_value=""
            ).astype("object")
            d["_rlnImageName"][:] = [
                f"{j + 1:06}@{prefix}" for j in range(self.particles_per_micrograph)
            ]
            d["_rlnImageSize"] = np.full(
                self.particles_per_micrograph, fill_value=self.particle_box_size
            )

            # Particle centers
            y, x = zip(*self.centers[m])  # unzips
            d["_rlnCoordinateX"] = x
            d["_rlnCoordinateY"] = y

            # Projection rotations
            rots_metadata = self.simulation.get_metadata(
                metadata_fields=_meta_fields["rotations"],
                indices=self.get_particle_indices(m),
                as_dict=True,
            )

            # CTF
            ctf_metadata = dict()
            if self.simulation.unique_filters:
                ctf_metadata = self.simulation.get_metadata(
                    metadata_fields=_meta_fields["ctf"],
                    indices=self.get_particle_indices(m),
                    as_dict=True,
                )

            # Union dictionaries
            d = {**d, **rots_metadata, **ctf_metadata}

            # Write STAR file for this micrograph
            star_file_path = os.path.join(path, f"{prefix}.star")
            StarFile(blocks={"": d}).write(star_file_path)
            output_star_filenames.append(star_file_path)

        return list(zip(output_mrc_filenames, output_star_filenames))
