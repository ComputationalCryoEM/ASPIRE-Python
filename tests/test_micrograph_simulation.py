import itertools
import logging
import os
import tempfile
from datetime import datetime
from unittest import mock

import numpy as np
import pytest

from aspire.noise import WhiteNoiseAdder
from aspire.operators import RadialCTFFilter
from aspire.source import (
    CentersCoordinateSource,
    DiskMicrographSource,
    MicrographSimulation,
)
from aspire.volume import AsymmetricVolume

logger = logging.getLogger(__name__)
IMG_SIZES = [12, 13]
DTYPES = [np.float32, np.float64]
PARTICLES_PER_MICROGRAPHS = [7, 10]
MICROGRAPH_COUNTS = [1, 2]
MICROGRAPH_SIZES = [101, 100]
SIM_VOLUMES = [1, 2]
BOUNDARIES = [-1, 0, 20]


def vol_fixture_id(params):
    sim_volumes = params[0]
    img_size = params[1]
    dtype = params[2]
    return f"number of volumes={sim_volumes}, image size={img_size}, dtype={dtype.__name__}"


@pytest.fixture(
    params=itertools.product(SIM_VOLUMES, IMG_SIZES, DTYPES), ids=vol_fixture_id
)
def vol_fixture(request):
    sim_volumes, img_size, dtype = request.param
    return AsymmetricVolume(L=img_size, C=sim_volumes, dtype=dtype).generate()


def micrograph_fixture_id(params):
    particles_per_micrograph = params[0]
    micrograph_count = params[1]
    micrograph_size = params[2]
    boundary = params[3]
    return f"particles per micrograph={particles_per_micrograph}, micrograph count={micrograph_count}, micrograph size={micrograph_size}, boundary={boundary}"


@pytest.fixture(
    params=itertools.product(
        PARTICLES_PER_MICROGRAPHS, MICROGRAPH_COUNTS, MICROGRAPH_SIZES, BOUNDARIES
    ),
    ids=micrograph_fixture_id,
)
def micrograph_fixture(vol_fixture, request):
    """
    Construct a MicrographSimulation.
    """
    (
        particles_per_micrograph,
        micrograph_count,
        micrograph_size,
        boundary,
    ) = request.param
    return MicrographSimulation(
        volume=vol_fixture,
        interparticle_distance=0,
        particles_per_micrograph=particles_per_micrograph,
        micrograph_count=micrograph_count,
        micrograph_size=micrograph_size,
        boundary=boundary,
    )


def test_micrograph_source_has_correct_values(vol_fixture, micrograph_fixture):
    """
    Test the MicrographSimulation has the correct values from arguments.
    """
    v = vol_fixture
    m = micrograph_fixture
    assert v.resolution == m.particle_box_size
    assert v == m.simulation.vols
    assert len(m) == m.micrograph_count
    assert m.clean_images[0].shape[1] == m.micrograph_size[0]
    assert m.clean_images[0].shape[2] == m.micrograph_size[1]
    assert (
        repr(m)
        == f"{m.__class__.__name__} with {m.micrograph_count} {m.dtype.name} micrographs of size {m.micrograph_size}"
    )
    _ = m.clean_images[:]
    _ = m.images[:]


def test_micrograph_raises_error_simulation():
    """
    Test that MicrographSimulation raises error when simulation argument is not a Simulation
    """
    with pytest.raises(Exception) as e_info:
        _ = MicrographSimulation(
            "Simulation",
            micrograph_size=100,
            particles_per_micrograph=20,
            interparticle_distance=10,
        )
    assert str(e_info.value) == "`volume` should be of type `Volume`."


def test_micrograph_simulation_pixel_size():
    """
    Test for various cases of (vol_px_sz, user_px_sz)
    """
    vol_px_sz = 1.23
    user_px_sz = 2.34

    L = 10
    # Case (None, None): None
    vol = AsymmetricVolume(L=L, C=1).generate()
    micrograph_sim = MicrographSimulation(
        vol,
        micrograph_size=50,
        micrograph_count=1,
        particles_per_micrograph=2,
    )
    assert micrograph_sim.pixel_size is None

    # Case (vol_px_sz, None): vol_px_sz
    vol = AsymmetricVolume(L=L, C=1, pixel_size=vol_px_sz).generate()
    micrograph_sim = MicrographSimulation(
        vol,
        micrograph_size=50,
        micrograph_count=1,
        particles_per_micrograph=2,
    )
    np.testing.assert_allclose(micrograph_sim.pixel_size, vol_px_sz)

    # Case (None, user_px_sz): user_px_sz
    vol = AsymmetricVolume(L=L, C=1).generate()
    micrograph_sim = MicrographSimulation(
        vol,
        micrograph_size=50,
        micrograph_count=1,
        particles_per_micrograph=2,
        pixel_size=user_px_sz,
    )
    np.testing.assert_allclose(micrograph_sim.pixel_size, user_px_sz)

    # Case (vol_px_sz, user_px_sz): user_px_sz w/ warning
    vol = AsymmetricVolume(L=L, C=1, pixel_size=vol_px_sz).generate()
    with pytest.warns(UserWarning, match="does not match pixel_size"):
        micrograph_sim = MicrographSimulation(
            vol,
            micrograph_size=50,
            micrograph_count=1,
            particles_per_micrograph=2,
            pixel_size=user_px_sz,
        )
        np.testing.assert_allclose(micrograph_sim.pixel_size, user_px_sz)


def test_micrograph_raises_error_image_size(vol_fixture):
    """
    Test the MicrographSimulation class raises errors when the image size is larger than micrograph size.
    """
    with pytest.raises(ValueError) as e_info:
        v = vol_fixture
        _ = MicrographSimulation(
            v,
            micrograph_size=v.resolution - 1,
            particles_per_micrograph=10,
            interparticle_distance=0,
        )
    assert (
        str(e_info.value)
        == "The micrograph size must be larger or equal to the `particle_box_size`."
    )


def test_micrograph_centers_match(micrograph_fixture):
    """
    Test that the Micrograph's centers are forming at generated points.
    """
    m = micrograph_fixture
    centers = np.reshape(m.centers, (m.total_particle_count, 2))
    for i, center in enumerate(centers):
        if (
            center[0] >= 0
            and center[0] < m.micrograph_size[0]
            and center[1] >= 0
            and center[1] < m.micrograph_size[1]
        ):
            assert m.clean_images[i // m.particles_per_micrograph].asnumpy()[0][
                tuple(center)
            ] != np.min(m.clean_images[i // m.particles_per_micrograph].asnumpy()[0])


def test_micrograph_raises_error_when_out_of_bounds(vol_fixture):
    """
    Test that the Micrograph raises an error when illegal boundary values are given.
    """

    for boundary_value in [-100, 1000]:
        with pytest.raises(ValueError) as e_info:
            _ = MicrographSimulation(
                vol_fixture,
                micrograph_size=500,
                particles_per_micrograph=20,
                micrograph_count=1,
                interparticle_distance=10,
                boundary=boundary_value,
            )
        assert str(e_info.value) == "Illegal boundary value."


def test_micrograph_raises_error_when_too_dense(vol_fixture):
    """
    Tests that the micrograph fails when the fail limit is met.
    """

    with pytest.raises(RuntimeError, match="failures exceeded limit") as _:
        _ = MicrographSimulation(
            vol_fixture,
            micrograph_size=100,
            particles_per_micrograph=400,
            micrograph_count=1,
        )


def test_index_returns_correct_values(vol_fixture):
    """
    Test index methods return expected values
    """

    m = MicrographSimulation(
        vol_fixture,
        micrograph_size=500,
        particles_per_micrograph=10,
        micrograph_count=1,
    )
    particle_id = 5
    assert m.get_micrograph_index(particle_id) == (0, particle_id)
    assert m.get_particle_indices(0, particle_id) == particle_id
    assert np.array_equal(
        m.get_particle_indices(0), np.arange(m.particles_per_micrograph)
    )


def test_index_functions_raise_errors(vol_fixture):
    """
    Test errors for index method bounds
    """

    m = MicrographSimulation(
        vol_fixture,
        micrograph_size=500,
        particles_per_micrograph=10,
        micrograph_count=1,
    )
    with pytest.raises(RuntimeError) as e_info:
        m.get_particle_indices(1)
    assert str(e_info.value) == "Index out of bounds for micrograph."
    with pytest.raises(RuntimeError) as e_info:
        m.get_particle_indices(-1)
    assert str(e_info.value) == "Index out of bounds for micrograph."
    with pytest.raises(RuntimeError) as e_info:
        m.get_micrograph_index(11)
    assert str(e_info.value) == "Index out of bounds."
    with pytest.raises(RuntimeError) as e_info:
        m.get_micrograph_index(-1)
    assert str(e_info.value) == "Index out of bounds."
    with pytest.raises(RuntimeError) as e_info:
        m.get_particle_indices(0, 500)
    assert str(e_info.value) == "Index out of bounds for particle."
    with pytest.raises(RuntimeError) as e_info:
        m.get_particle_indices(0, -1)
    assert str(e_info.value) == "Index out of bounds for particle."


def test_noise_works(vol_fixture):
    """
    Tests that adding noise works by comparing to a micrograph with noise manually applied.
    """

    noise = WhiteNoiseAdder(1e-3)
    m = MicrographSimulation(
        vol_fixture,
        noise_adder=noise,
        micrograph_count=1,
        particles_per_micrograph=4,
        micrograph_size=200,
    )
    noisy_micrograph = noise.forward(m.clean_images[:], [0])
    assert np.array_equal(m.images[0], noisy_micrograph[0])


def test_sim_save():
    """
    Tests MicrographSimulation.save functionality.

    Specifically tests interoperability with CentersCoordinateSource
    """

    v = AsymmetricVolume(L=16, C=1, pixel_size=4, dtype=np.float64).generate()
    ctfs = [RadialCTFFilter(voltage=200, defocus=15000, Cs=2.26, alpha=0.07, B=0)]

    mg_sim = MicrographSimulation(
        volume=v,
        particles_per_micrograph=3,
        interparticle_distance=v.resolution,
        micrograph_count=2,
        micrograph_size=512,
        ctf_filters=ctfs,
    )

    with tempfile.TemporaryDirectory() as tmp_output_dir:
        path = os.path.join(tmp_output_dir, "test")

        # Write MRC and STAR files
        results = mg_sim.save(path)

        # Test we can load from dir `path`
        mg_src = DiskMicrographSource(path)
        np.testing.assert_allclose(mg_src.asnumpy(), mg_sim.asnumpy())

        # Test we can load via CentersCoordinateSource (STAR files)
        img_src = CentersCoordinateSource(
            results,
            v.pixel_size,
            mg_sim.particle_box_size,
        )
        np.testing.assert_allclose(
            img_src.images[:].asnumpy(),  # loaded image stack
            mg_sim.simulation.images[:].asnumpy(),  # simulated image stack
        )

        # TODO, Issue #1006
        # The following tests should pass, but are a different project.
        # Only the basic image stack behavior above.
        pytest.xfail(reason="CoordinateSource implementations are incomplete.")

        # Test the rotations match (auto load metadata from STAR).
        np.testing.assert_allclose(img_src.rotations, mg_sim.simulation.rotations)

        # Test a CTF param matches (auto load metadata from STAR).
        np.testing.assert_allclose(
            img_src.get_metadata("_rlnDefocusU"),
            mg_sim.simulation.get_metadata("_rlnDefocusU"),
        )
        # Alternatively, manually import CTF using the provided function, fails.
        img_src.import_relion_ctf([r[1] for r in results])
        np.testing.assert_allclose(
            img_src.get_metadata("_rlnDefocusU"),
            mg_sim.simulation.get_metadata("_rlnDefocusU"),
        )


def test_save_overwrite(caplog):
    """
    Tests MicrographSimulation.save functionality.

    Specifically tests interoperability with CentersCoordinateSource
    """

    v = AsymmetricVolume(L=16, C=1, pixel_size=4, dtype=np.float64).generate()
    ctfs = [RadialCTFFilter(voltage=200, defocus=15000, Cs=2.26, alpha=0.07, B=0)]

    mg_sim = MicrographSimulation(
        volume=v,
        particles_per_micrograph=3,
        interparticle_distance=v.resolution,
        micrograph_count=2,
        micrograph_size=512,
        ctf_filters=ctfs,
    )

    mg_sim_new = MicrographSimulation(
        volume=v,
        particles_per_micrograph=4,
        interparticle_distance=v.resolution,
        micrograph_count=3,
        micrograph_size=512,
        ctf_filters=ctfs,
    )

    with tempfile.TemporaryDirectory() as tmp_output_dir:
        path = os.path.join(tmp_output_dir, "test")

        # Write MRC and STAR files
        save_paths_1 = mg_sim.save(path, overwrite=True)

        # Case 1: overwrite=True (should overwrite the existing file)
        save_paths_2 = mg_sim.save(path, overwrite=True)
        np.testing.assert_array_equal(save_paths_1, save_paths_2)

        # Case2: overwrite=False (should raise error)
        with pytest.raises(FileExistsError):
            _ = mg_sim.save(path, overwrite=False)

        # Case 3: overwrite=None (should rename the existing directory)
        mock_datetime_value = datetime(2024, 10, 18, 12, 0, 0)
        with mock.patch("aspire.utils.misc.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_datetime_value
            mock_datetime.strftime = datetime.strftime

            with caplog.at_level(logging.INFO):
                _ = mg_sim_new.save(path, overwrite=None)

                # Check that the existing directory was renamed and logged
                assert f"Renaming {path}" in caplog.text
                assert os.path.exists(path), "Directory not found"

                # Construct the expected renamed directory using the mock timestamp
                mock_timestamp = mock_datetime_value.strftime("%y%m%d_%H%M%S")
                renamed_dir = f"{path}_{mock_timestamp}"

                # Assert that the renamed file exists
                assert os.path.exists(renamed_dir), "Renamed directory not found"

                # Load renamed directory and check images against orignal sim.
                mg_src = DiskMicrographSource(renamed_dir)
                np.testing.assert_allclose(mg_src.asnumpy(), mg_sim.asnumpy())

                # Load new directory and check images against orignal sim.
                mg_src_new = DiskMicrographSource(path)
                np.testing.assert_allclose(mg_src_new.asnumpy(), mg_sim_new.asnumpy())


def test_bad_amplitudes(vol_fixture):
    """
    Test incorrect `particle_amplitudes` argument raises.
    """
    with pytest.raises(RuntimeError, match=r".*particle_amplitudes.*"):
        _ = MicrographSimulation(
            volume=vol_fixture,
            particles_per_micrograph=1,
            micrograph_count=1,
            micrograph_size=512,
            particle_amplitudes=np.empty(2),  # total particles == 1
        )


def test_bad_angles(vol_fixture):
    """
    Test incorrect `projection_angles` argument raises.
    """
    with pytest.raises(RuntimeError, match=r".*projection_angles.shape.*"):
        _ = MicrographSimulation(
            volume=vol_fixture,
            particles_per_micrograph=1,
            micrograph_count=1,
            micrograph_size=512,
            projection_angles=np.empty((2, 3)),  # total particles == 1
        )


def test_bad_ctf(vol_fixture):
    """
    Test incorrect `ctf_filters` argument raises.
    """
    with pytest.raises(TypeError, match=r".*expects a list of len.*"):
        _ = MicrographSimulation(
            volume=vol_fixture,
            particles_per_micrograph=1,
            micrograph_count=1,
            micrograph_size=512,
            ctf_filters=[
                RadialCTFFilter(),
            ]
            * 2,  # total particles == 1
        )
