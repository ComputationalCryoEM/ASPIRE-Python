import logging
import os.path
import tempfile
from unittest import TestCase

import numpy as np
import pytest

from aspire.noise import WhiteNoiseAdder
from aspire.operators import RadialCTFFilter
from aspire.source import ImageSource, RelionSource, Simulation, _LegacySimulation
from aspire.utils import RelionStarFile, utest_tolerance
from aspire.volume import LegacyVolume, SymmetryGroup, Volume

from .test_utils import matplotlib_dry_run

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class SingleSimTestCase(TestCase):
    """Test we can construct a length 1 Sim."""

    def setUp(self):
        self._pixel_size = 1.23  # Test value

        self.sim = Simulation(n=1, L=8, pixel_size=self._pixel_size)

    def testImage(self):
        """Test we can get an Image from a length 1 Sim."""
        _ = self.sim.images[0]

    def testPixelSize(self):
        """Test pixel_size is passing through Simulation."""
        self.assertTrue(self.sim.pixel_size == self._pixel_size)
        self.assertTrue(self.sim.pixel_size == self.sim.vols.pixel_size)

    @matplotlib_dry_run
    def testImageShow(self):
        self.sim.images[:].show()

    @matplotlib_dry_run
    def testCleanImagesShow(self):
        self.sim.clean_images[:].show()

    @matplotlib_dry_run
    def testProjectionsShow(self):
        self.sim.projections[:].show()


class SimVolTestCase(TestCase):
    """Test Simulation with Volume provided."""

    def setUp(self):
        self.dtype = np.float32
        self.vol_res = 10
        self.vol_arr = np.ones((self.vol_res,) * 3, dtype=self.dtype)
        self.vol = Volume(self.vol_arr)

    def tearDown(self):
        pass

    def testResolutionMismatch(self):
        # Test we raise with expected error message with Volume/Simulation mismatch.
        with pytest.raises(
            RuntimeError, match=r"Simulation must have the same resolution*"
        ):
            _ = Simulation(L=8, vols=self.vol)

    def testNonVolumeError(self):
        # Test we raise with expected error if vols is not a Volume instance.
        with pytest.raises(RuntimeError, match=r"`vols` should be a Volume instance*"):
            _ = Simulation(L=self.vol_res, vols=self.vol_arr)

    def testDtypeMismatch(self):
        """
        Test we raise when the volume dtype does not match explicit Simulation dtype.
        """
        with pytest.raises(
            RuntimeError, match=r".*does not match provided vols.dtype.*"
        ):
            _ = Simulation(vols=self.vol.astype(np.float16), dtype=self.dtype)

    def testPassthroughFromVol(self):
        """
        Test we do not crash when passing a volume to Simulation,
        without an explcit Simulation dtype.
        """
        for dtype in (np.float32, np.float64):
            sim = Simulation(vols=self.vol.astype(dtype, copy=False))
            # Did we assign the right type?
            self.assertTrue(sim.dtype == dtype)

            # Is the Volume the intended type?
            self.assertTrue(sim.vols.dtype == dtype)

    def testPassthroughFromSim(self):
        """
        Test we do not crash when passing a volume to Simulation,
        with out an explcit Volume.
        """
        for dtype in (np.float32, np.float64):
            # Create a minimal sim
            sim = Simulation(dtype=dtype)

            # Did we assign the right type?
            self.assertTrue(sim.dtype == dtype)

            # Is the Volume the intended type?
            self.assertTrue(sim.vols.dtype == dtype)


class SimTestCase(TestCase):
    def setUp(self):
        self.n = 1024
        self.L = 8
        self.dtype = np.float32
        # Set legacy pixel_size
        self._pixel_size = 10

        self.vols = LegacyVolume(
            L=self.L,
            pixel_size=self._pixel_size,
            dtype=self.dtype,
        ).generate()

        self.sim = _LegacySimulation(
            n=self.n,
            L=self.L,
            vols=self.vols,
            unique_filters=[
                RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)
            ],
            noise_adder=WhiteNoiseAdder(var=1),
            dtype=self.dtype,
        )

        # Keep hardcoded tests passing after fixing swapped offsets.
        # See github issue #1146.
        self.sim.sim_offsets = self.sim.offsets[:, [1, 0]]
        self.sim = self.sim.update(offsets=self.sim.offsets[:, [1, 0]])

    def tearDown(self):
        pass

    def testGaussianBlob(self):
        blobs = self.sim.vols.asnumpy()
        ref = np.load(os.path.join(DATA_DIR, "sim_blobs.npy"))
        np.testing.assert_allclose(blobs, ref, rtol=1e-05, atol=1e-08)

    def testSimulationRots(self):
        np.testing.assert_allclose(
            self.sim.rots_zyx_to_legacy_aspire(self.sim.rotations[0, :, :]),
            np.array(
                [
                    [0.91675498, 0.2587233, 0.30433956],
                    [0.39941773, -0.58404652, -0.70665065],
                    [-0.00507853, 0.76938412, -0.63876622],
                ]
            ),
            atol=utest_tolerance(self.dtype),
        )

    def testSimulationImages(self):
        images = self.sim.clean_images[:512].asnumpy()
        np.testing.assert_allclose(
            images,
            np.load(os.path.join(DATA_DIR, "sim_clean_images.npy")),
            rtol=1e-2,
            atol=utest_tolerance(self.sim.dtype),
        )

    def testSimulationCached(self):
        sim_cached = _LegacySimulation(
            n=self.n,
            L=self.L,
            vols=self.vols,
            offsets=self.sim.offsets,
            unique_filters=[
                RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)
            ],
            noise_adder=WhiteNoiseAdder(var=1),
            dtype=self.dtype,
        )
        sim_cached = sim_cached.cache()
        np.testing.assert_allclose(
            sim_cached.images[:].asnumpy(), self.sim.images[:].asnumpy(), atol=1e-6
        )

    def testSimulationImagesNoisy(self):
        images = self.sim.images[:512].asnumpy()
        np.testing.assert_allclose(
            images,
            np.load(os.path.join(DATA_DIR, "sim_images_with_noise.npy")),
            rtol=1e-2,
            atol=utest_tolerance(self.sim.dtype),
        )

    def testSimulationImagesDownsample(self):
        # The simulation already generates images of size 8 x 8; Downsampling to resolution 8 should thus have no effect
        self.sim = self.sim.downsample(8)
        images = self.sim.clean_images[:512].asnumpy()
        np.testing.assert_allclose(
            images,
            np.load(os.path.join(DATA_DIR, "sim_clean_images.npy")),
            rtol=1e-2,
            atol=utest_tolerance(self.sim.dtype),
        )

    def testSimulationImagesShape(self):
        # The 'images' method should be tolerant of bounds - here we ask for 1000 images starting at index 1000,
        # so we'll get back 25 images in return instead
        images = self.sim.images[1000:2000]
        self.assertTrue(images.shape, (8, 8, 25))

    def testSimulationImagesDownsampleShape(self):
        self.sim = self.sim.downsample(6)
        first_image = self.sim.images[0].asnumpy()[0]
        self.assertEqual(first_image.shape, (6, 6))

    def testSimulationEigen(self):
        eigs_true, lambdas_true = self.sim.eigs()
        np.testing.assert_allclose(
            eigs_true.asnumpy()[0, :, :, 2],
            np.array(
                [
                    [
                        -1.67666201e-07,
                        -7.95741380e-06,
                        -1.49160041e-04,
                        -1.10151654e-03,
                        -3.11287888e-03,
                        -3.09157884e-03,
                        -9.91418026e-04,
                        -1.31673165e-04,
                    ],
                    [
                        -1.15402077e-06,
                        -2.49849709e-05,
                        -3.51658906e-04,
                        -2.21575261e-03,
                        -7.83315487e-03,
                        -9.44795180e-03,
                        -4.07636259e-03,
                        -9.02186439e-04,
                    ],
                    [
                        -1.88737249e-05,
                        -1.91418396e-04,
                        -1.09021540e-03,
                        -1.02020288e-03,
                        1.39411855e-02,
                        8.58035963e-03,
                        -5.54619730e-03,
                        -3.86377703e-03,
                    ],
                    [
                        -1.21280536e-04,
                        -9.51461843e-04,
                        -3.22565017e-03,
                        -1.05731178e-03,
                        2.61375736e-02,
                        3.11595201e-02,
                        6.40814053e-03,
                        -2.31698658e-02,
                    ],
                    [
                        -2.44067283e-04,
                        -1.40560151e-03,
                        -6.73082832e-05,
                        1.44160679e-02,
                        2.99893934e-02,
                        5.92632964e-02,
                        7.75623545e-02,
                        3.06570008e-02,
                    ],
                    [
                        -1.53507499e-04,
                        -7.21709803e-04,
                        8.54929152e-04,
                        -1.27235036e-02,
                        -5.34382043e-03,
                        2.18879692e-02,
                        6.22706190e-02,
                        4.51998860e-02,
                    ],
                    [
                        -3.00595184e-05,
                        -1.43038429e-04,
                        -2.15870258e-03,
                        -9.99002904e-02,
                        -7.79077187e-02,
                        -1.53395887e-02,
                        1.88777559e-02,
                        1.68759506e-02,
                    ],
                    [
                        3.22692649e-05,
                        4.07977635e-03,
                        1.63959339e-02,
                        -8.68835449e-02,
                        -7.86240026e-02,
                        -1.75694861e-02,
                        3.24984640e-03,
                        1.95389288e-03,
                    ],
                ]
            ),
            rtol=1e-05,
            atol=1e-08,
        )

    def testSimulationMean(self):
        mean_vol = self.sim.mean_true()
        np.testing.assert_allclose(
            [
                [
                    0.00000930,
                    0.00033866,
                    0.00490734,
                    0.01998369,
                    0.03874487,
                    0.04617764,
                    0.02970645,
                    0.00967604,
                ],
                [
                    0.00003904,
                    0.00247391,
                    0.03818476,
                    0.12325402,
                    0.22278425,
                    0.25246665,
                    0.14093882,
                    0.03683474,
                ],
                [
                    0.00014177,
                    0.01191146,
                    0.14421064,
                    0.38428235,
                    0.78645319,
                    0.86522675,
                    0.44862473,
                    0.16382280,
                ],
                [
                    0.00066036,
                    0.03137806,
                    0.29226971,
                    0.97105378,
                    2.39410496,
                    2.17099857,
                    1.23595858,
                    0.49233940,
                ],
                [
                    0.00271748,
                    0.05491289,
                    0.49955708,
                    2.05356097,
                    3.70941424,
                    3.01578689,
                    1.51441932,
                    0.52054572,
                ],
                [
                    0.00584845,
                    0.06962635,
                    0.50568032,
                    1.99643707,
                    3.77415895,
                    2.76039767,
                    1.04602003,
                    0.20633197,
                ],
                [
                    0.00539583,
                    0.06068972,
                    0.47008955,
                    1.17128026,
                    1.82821035,
                    1.18743944,
                    0.30667788,
                    0.04851476,
                ],
                [
                    0.00246362,
                    0.04867788,
                    0.65284950,
                    0.65238875,
                    0.65745538,
                    0.37955678,
                    0.08053055,
                    0.01210055,
                ],
            ],
            mean_vol.asnumpy()[0, :, :, 4],
            rtol=1e-05,
            atol=1e-08,
        )

    def testSimulationVolCoords(self):
        coords, norms, inners = self.sim.vol_coords()
        np.testing.assert_allclose([4.72837704, -4.72837709], coords, atol=1e-4)
        np.testing.assert_allclose([8.20515764e-07, 1.17550184e-06], norms, atol=1e-4)
        np.testing.assert_allclose(
            [[3.78030562e-06, -4.20475816e-06]], inners, atol=1e-4
        )

    def testSimulationCovar(self):
        covar = self.sim.covar_true()
        result = [
            [
                -0.00000289,
                -0.00005839,
                -0.00018998,
                -0.00124722,
                -0.00003155,
                +0.00743356,
                +0.00798143,
                +0.00303416,
            ],
            [
                -0.00000776,
                +0.00018371,
                +0.00448675,
                -0.00794970,
                -0.02988000,
                -0.00185446,
                +0.01786612,
                +0.00685990,
            ],
            [
                +0.00001144,
                +0.00324029,
                +0.03364052,
                -0.00272520,
                -0.08976389,
                -0.05404807,
                +0.00268740,
                -0.03081760,
            ],
            [
                +0.00003204,
                +0.00909853,
                +0.07859941,
                +0.07254293,
                -0.19365733,
                -0.09007251,
                -0.15731451,
                -0.15690306,
            ],
            [
                -0.00040561,
                +0.00685139,
                +0.11074986,
                +0.35207557,
                +0.17264650,
                -0.16662873,
                -0.15010859,
                -0.14292650,
            ],
            [
                -0.00107461,
                -0.00497393,
                +0.04630126,
                +0.38048555,
                +0.47915877,
                +0.05379957,
                -0.11833663,
                -0.03372971,
            ],
            [
                -0.00029630,
                -0.00485664,
                -0.00640120,
                +0.22068169,
                +0.15419035,
                +0.08281200,
                +0.03373241,
                +0.00103902,
            ],
            [
                +0.00044323,
                +0.00850533,
                +0.09683860,
                +0.16959519,
                +0.03629097,
                +0.03740599,
                +0.02212356,
                +0.00318127,
            ],
        ]

        np.testing.assert_allclose(result, covar[:, :, 4, 4, 4, 4], atol=1e-4)

    def testSimulationEvalMean(self):
        mean_est = Volume(
            np.load(os.path.join(DATA_DIR, "mean_8_8_8.npy")), dtype=self.dtype
        )
        result = self.sim.eval_mean(mean_est)

        np.testing.assert_allclose(result["err"], 2.664116055950763, atol=1e-4)
        np.testing.assert_allclose(result["rel_err"], 0.1765943704851626, atol=1e-4)
        np.testing.assert_allclose(result["corr"], 0.9849211540734224, atol=1e-4)

    def testSimulationEvalCovar(self):
        covar_est = np.load(os.path.join(DATA_DIR, "covar_8_8_8_8_8_8.npy"))
        result = self.sim.eval_covar(covar_est)

        np.testing.assert_allclose(result["err"], 13.322721549011165, atol=1e-4)
        np.testing.assert_allclose(result["rel_err"], 0.5958936073938558, atol=1e-4)
        np.testing.assert_allclose(result["corr"], 0.8405347287741631, atol=1e-4)

    def testSimulationEvalCoords(self):
        mean_est = Volume(
            np.load(os.path.join(DATA_DIR, "mean_8_8_8.npy")), dtype=self.dtype
        )
        eigs_est = Volume(
            np.load(os.path.join(DATA_DIR, "eigs_est_8_8_8_1.npy"))[..., 0],
            dtype=self.dtype,
        )

        clustered_coords_est = np.load(
            os.path.join(DATA_DIR, "clustered_coords_est.npy")
        ).astype(dtype=self.dtype)

        result = self.sim.eval_coords(mean_est, eigs_est, clustered_coords_est)

        np.testing.assert_allclose(
            result["err"][0, :10],
            [
                1.58382394,
                1.58382394,
                3.72076112,
                1.58382394,
                1.58382394,
                3.72076112,
                3.72076112,
                1.58382394,
                1.58382394,
                1.58382394,
            ],
        )

        np.testing.assert_allclose(
            result["rel_err"][0, :10],
            [
                0.11048937,
                0.11048937,
                0.21684697,
                0.11048937,
                0.11048937,
                0.21684697,
                0.21684697,
                0.11048937,
                0.11048937,
                0.11048937,
            ],
            rtol=1e-05,
            atol=1e-08,
        )

        np.testing.assert_allclose(
            result["corr"][0, :10],
            [
                0.99390133,
                0.99390133,
                0.97658719,
                0.99390133,
                0.99390133,
                0.97658719,
                0.97658719,
                0.99390133,
                0.99390133,
                0.99390133,
            ],
            rtol=1e-05,
            atol=1e-08,
        )

    def testSimulationSaveFile(self):
        # Create a tmpdir in a context. It will be cleaned up on exit.
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save the simulation object into STAR and MRCS files
            star_filepath = os.path.join(tmpdir, "save_test.star")
            # Save images into one single MRCS file
            info = self.sim.save(
                star_filepath, batch_size=512, save_mode="single", overwrite=False
            )
            # check info output by save()
            self.assertEqual(
                info,
                {
                    "starfile": star_filepath,
                    "mrcs": [f"save_test_0_{self.sim.n-1}.mrcs"],
                },
            )
            imgs_org = self.sim.images[:1024]
            # Input saved images into Relion object
            relion_src = RelionSource(star_filepath, tmpdir, max_rows=1024)
            imgs_sav = relion_src.images[:1024]
            # Compare original images with saved images
            np.testing.assert_allclose(
                imgs_org.asnumpy(), imgs_sav.asnumpy(), atol=1e-6
            )
            # Save images into multiple MRCS files based on batch size
            batch_size = 512
            info = self.sim.save(star_filepath, batch_size=batch_size, overwrite=False)
            # check info output by save()
            self.assertEqual(
                info,
                {
                    "starfile": star_filepath,
                    "mrcs": [
                        f"save_test_{i}_{i+batch_size-1}.mrcs"
                        for i in range(0, self.sim.n, batch_size)
                    ],
                },
            )
            # Input saved images into Relion object
            relion_src = RelionSource(star_filepath, tmpdir, max_rows=1024)
            imgs_sav = relion_src.images[:1024]
            # Compare original images with saved images
            np.testing.assert_allclose(
                imgs_org.asnumpy(), imgs_sav.asnumpy(), atol=1e-6
            )


def test_simulation_save_optics_block(tmp_path):
    res = 32

    # Radial CTF Filters. Should make 3 distinct optics blocks
    kv_min, kv_max, kv_ct = 200, 300, 3
    voltages = np.linspace(kv_min, kv_max, kv_ct)
    ctf_filters = [RadialCTFFilter(voltage=kv) for kv in voltages]

    # Generate and save Simulation
    sim = Simulation(
        n=9, L=res, C=1, unique_filters=ctf_filters, pixel_size=1.34
    ).cache()
    starpath = tmp_path / "sim.star"
    sim.save(starpath, overwrite=True)

    star = RelionStarFile(str(starpath))
    assert star.relion_version == "3.1"
    assert star.blocks.keys() == {"optics", "particles"}

    optics = star["optics"]
    expected_optics_fields = [
        "_rlnOpticsGroup",
        "_rlnOpticsGroupName",
        "_rlnImagePixelSize",
        "_rlnSphericalAberration",
        "_rlnVoltage",
        "_rlnAmplitudeContrast",
        "_rlnImageSize",
        "_rlnImageDimensionality",
    ]

    # Check all required fields are present
    for field in expected_optics_fields:
        assert field in optics

    # Optics group and group name should 1-indexed
    np.testing.assert_array_equal(
        optics["_rlnOpticsGroup"], np.arange(1, kv_ct + 1, dtype=int)
    )
    np.testing.assert_array_equal(
        optics["_rlnOpticsGroupName"],
        np.array([f"opticsGroup{i}" for i in range(1, kv_ct + 1)]),
    )

    # Check image size (res) and image dimensionality (2)
    np.testing.assert_array_equal(optics["_rlnImageSize"], np.full(kv_ct, res))
    np.testing.assert_array_equal(optics["_rlnImageDimensionality"], np.full(kv_ct, 2))

    # Due to Simulation random indexing, voltages will be unordered
    np.testing.assert_allclose(np.sort(optics["_rlnVoltage"]), voltages)

    # Check that each row of the data_particles block has an associated optics group
    particles = star["particles"]
    assert "_rlnOpticsGroup" in particles
    assert len(particles["_rlnOpticsGroup"]) == sim.n
    np.testing.assert_array_equal(
        np.sort(np.unique(particles["_rlnOpticsGroup"])),
        np.arange(1, kv_ct + 1, dtype=int),
    )

    # Test phase_flip after save/load round trip to ensure correct optics group mapping
    rln_src = RelionSource(starpath)
    np.testing.assert_allclose(
        sim.phase_flip().images[:], rln_src.phase_flip().images[:]
    )


def test_default_symmetry_group():
    # Check that default is "C1".
    sim = Simulation()
    assert isinstance(sim.symmetry_group, SymmetryGroup)
    assert str(sim.symmetry_group) == "C1"


def test_pixel_size(caplog):
    data = np.arange(8**3, dtype=np.float32).reshape(8, 8, 8)

    # Default to 1 angstrom when not provided.
    sim = Simulation()
    np.testing.assert_array_equal(sim.pixel_size, 1.0)

    # Check pixel_size inhereted from volume.
    vol = Volume(data, pixel_size=1.23)
    sim = Simulation(vols=vol)
    np.testing.assert_array_equal(sim.pixel_size, vol.pixel_size)

    # Check pixel_size passes from sim to default volume.
    sim = Simulation(pixel_size=2.34)
    np.testing.assert_array_equal(sim.pixel_size, sim.vols.pixel_size)

    # Check mismatched pixel_size warns and uses provided pixel_size.
    user_px_sz = vol.pixel_size / 2
    with pytest.warns(UserWarning, match="does not match pixel_size"):
        sim = Simulation(vols=vol, pixel_size=user_px_sz)
        np.testing.assert_allclose(sim.pixel_size, user_px_sz)


def test_symmetry_group_inheritence():
    # Check SymmetryGroup inheritence from Volume.
    data = np.arange(8**3, dtype=np.float32).reshape(8, 8, 8)
    vol = Volume(data, symmetry_group="T")
    sim = Simulation(vols=vol)
    assert isinstance(sim.symmetry_group, SymmetryGroup)
    assert str(sim.symmetry_group) == "T"


def test_symmetry_group_errors(caplog):
    # Check that providing a symmetry different than vols logs warning.
    caplog.clear()
    msg = "Overriding C1 symmetry group inherited"
    caplog.set_level(logging.WARN)
    assert msg not in caplog.text
    _ = Simulation(symmetry_group="D11")
    assert msg in caplog.text


def test_cached_image_accessors():
    """
    Test the behavior of image caching.
    """
    # Create a CTF
    ctf = [RadialCTFFilter()]
    # Create a Simulation with noise and `ctf`
    src = Simulation(
        L=32,
        n=3,
        C=1,
        noise_adder=WhiteNoiseAdder(var=0.123),
        unique_filters=ctf,
        pixel_size=5,
    )
    # Cache the simulation
    cached_src = src.cache()

    # Compare the cached vs dynamic image sets.
    np.testing.assert_allclose(cached_src.projections[:], src.projections[:], atol=1e-6)
    np.testing.assert_allclose(cached_src.images[:], src.images[:], atol=1e-6)
    np.testing.assert_allclose(
        cached_src.clean_images[:], src.clean_images[:], atol=1e-6
    )


def test_projections_and_clean_images_downsample():
    """
    Test `projections` and `clean_images` post downsample.
    `projections` should remain unaltered and `clean_images` should
    be resized with adjusted pixel_size.
    """
    n = 10
    L = 32
    L_ds = 21
    px_sz = 1.23
    ctf = [RadialCTFFilter(1.5e4)]

    src = Simulation(
        L=L,
        n=n,
        C=1,
        noise_adder=WhiteNoiseAdder(var=0.123),
        unique_filters=ctf,
        pixel_size=px_sz,
    )

    src_ds = src.downsample(L_ds)

    # Check pixel_size
    np.testing.assert_allclose(src_ds.projections[:].pixel_size, px_sz)
    np.testing.assert_allclose(src_ds.clean_images[:].pixel_size, px_sz * L / L_ds)

    # Check image size
    np.testing.assert_allclose(src_ds.projections[:].shape[-1], L)
    np.testing.assert_allclose(src_ds.clean_images[:].shape[-1], L_ds)


def test_save_overwrite(caplog):
    """
    Test that the overwrite flag behaves as expected.
    - overwrite=True: Overwrites the existing file.
    - overwrite=False: Raises an error if the file exists.
    - overwrite=None: Renames the existing file and saves the new one.
    """
    sim1 = Simulation(seed=1)
    sim2 = Simulation(seed=2)
    sim3 = Simulation(seed=3)

    # Create a tmp dir for this test output
    with tempfile.TemporaryDirectory() as tmpdir_name:
        # tmp filename
        starfile = os.path.join(tmpdir_name, "og.star")
        base, ext = os.path.splitext(starfile)

        sim1.save(starfile, overwrite=True)

        # Case 1: overwrite=True (should overwrite the existing file)
        sim2.save(starfile, overwrite=True)

        # Load and check if sim2 has overwritten sim1
        sim2_loaded = RelionSource(starfile)
        np.testing.assert_allclose(
            sim2.images[:].asnumpy(),
            sim2_loaded.images[:].asnumpy(),
            atol=utest_tolerance(sim2.dtype),
        )

        # Check that metadata is unchanged.
        check_metadata(sim2, sim2_loaded)

        # Case 2: overwrite=False (should raise an overwrite error)
        with pytest.raises(
            ValueError,
            match="File '.*' already exists; set overwrite=True to overwrite it",
        ):
            sim2.save(starfile, overwrite=False)

        # case 3: overwrite=None (should rename the existing file and save im3 with original filename)
        with caplog.at_level(logging.INFO):
            sim3.save(starfile, overwrite=None)

            # Check that the existing file was renamed and logged
            assert f"Renaming {starfile}" in caplog.text

            # Find the renamed file by checking the directory contents
            renamed_file = None
            for filename in os.listdir(tmpdir_name):
                if filename.startswith("og_") and filename.endswith(".star"):
                    renamed_file = os.path.join(tmpdir_name, filename)
                    break

            assert renamed_file is not None, "Renamed file not found"

        # Load and check that sim3 was saved to the original path
        sim3_loaded = RelionSource(starfile)
        np.testing.assert_allclose(
            sim3.images[:].asnumpy(),
            sim3_loaded.images[:].asnumpy(),
            atol=utest_tolerance(sim3.dtype),
        )
        check_metadata(sim3, sim3_loaded)

        # Also check that the renamed file still contains sim2's data
        sim2_loaded_renamed = RelionSource(renamed_file)
        np.testing.assert_allclose(
            sim2.images[:].asnumpy(),
            sim2_loaded_renamed.images[:].asnumpy(),
            atol=utest_tolerance(sim2.dtype),
        )
        check_metadata(sim2, sim2_loaded_renamed)


def check_metadata(sim_src, relion_src):
    """
    Helper function to test if metadata fields in a Simulation match
    those in a RelionSource.
    """
    for k, v in sim_src._metadata.items():
        try:
            np.testing.assert_array_equal(v, relion_src._metadata[k])
        except AssertionError:
            # Loaded metadata might be strings so recast.
            np.testing.assert_allclose(
                v, np.array(relion_src._metadata[k]).astype(type(v[0]))
            )
