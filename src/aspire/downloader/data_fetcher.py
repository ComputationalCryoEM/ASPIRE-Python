import shutil

import numpy as np
import pooch

from aspire import config
from aspire.downloader import file_to_method_map, registry, registry_urls
from aspire.image import Image
from aspire.source import _LegacySimulation
from aspire.utils import Rotation
from aspire.volume import Volume

# Initialize pooch data fetcher instance.
_data_fetcher = pooch.create(
    # Set the cache path defined in the config. By default, the cache
    # folder operating system dependent, set by `pooch.os_cache`.
    # Pooch uses appdirs (https://github.com/ActiveState/appdirs) to
    # select an appropriate directory for the cache on each platform.
    path=config["common"]["cache_dir"].as_filename(),
    # The remote data is on Zenodo, `base_url` is a required param,
    # even though we override using individual urls in the registry.
    base_url="https://zenodo.org/communities/computationalcryoem/",
    registry=registry,
    urls=registry_urls,
)


def fetch_data(dataset_name):
    """
    The `fetch_data` method returns the full path to the downloaded data file.

    If it is not in the local storage, it will be downloaded. If the hash of the
    file in local storage doesn’t match the one in the registry, will download a
    new copy of the file. This is considered a sign that the file was updated in
    the remote storage. If the hash of the downloaded file still doesn’t match the
    one in the registry, will raise an exception to warn of possible file corruption.

    :param dataset_name: The file name (as appears in the registry) to
        fetch from local storage.
    :return: The absolute path (including the file name) of the file in
        local storage.
    """
    return _data_fetcher.fetch(dataset_name)


def download_all():
    """
    Download all ASPIRE example data and return a dictionary of filepaths.

    :return: A dictionary of method names and associated file paths.
    """

    file_paths = {}
    for data_set in registry:
        path = fetch_data(data_set)
        name = file_to_method_map[data_set]
        file_paths[name] = path

    return file_paths


def remove_downloads():
    """
    Remove the downloads directory.
    """
    shutil.rmtree(_data_fetcher.abspath)


def available_downloads():
    """
    List all available downloads.

    :return: A list of method names for downloadable files.
    """
    return list(file_to_method_map.values())


def emdb_2660():
    """
    Downloads the EMDB-2660 volume map and returns a `Volume` instance.

    Cryo-EM structure of the Plasmodium falciparum 80S ribosome
    bound to the anti-protozoan drug emetine.

    :return: A `Volume` instance.
    """
    file_path = fetch_data("emdb_2660.map")
    vol = Volume.load(file_path)

    return vol


def emdb_8012():
    """
    Downloads the EMDB-8012 volume map and returns the file path.

    The overall structure of the yeast spliceosomal U4/U6.U5 tri-snRNP at 3.7 Angstrom.

    :return: A 'Volume' instance.
    """
    file_path = fetch_data("emdb_8012.map")
    vol = Volume.load(file_path)

    return vol


def emdb_2984():
    """
    Downloads the EMDB-2984 volume map and returns the file path.

    2.2 A resolution cryo-EM structure of beta-galactosidase in complex with a cell-permeant inhibitor.
    This molecule exhibits D2 symmetry.

    :return: A 'Volume' instance.
    """
    file_path = fetch_data("emdb_2984.map")
    vol = Volume.load(file_path, symmetry_group="D2")

    return vol


def emdb_8511():
    """
    Downloads the EMDB-8511 volume map and returns the file path.

    Structure of the human HCN1 hyperpolarization-activated cyclic nucleotide-gated ion channel.
    This molecule exhibits C4 symmetry.

    :return: A 'Volume' instance.
    """
    file_path = fetch_data("emdb_8511.map")
    vol = Volume.load(file_path, symmetry_group="C4")

    return vol


def emdb_3645():
    """
    Downloads the EMDB-3645 volume map and returns the file path.

    CryoEM density of TcdA1 in prepore state (SPHIRE tutorial).
    This molecule exhibits C5 symmetry.

    :return: A 'Volume' instance.
    """
    file_path = fetch_data("emdb_3645.map")
    vol = Volume.load(file_path, symmetry_group="C5")

    return vol


def emdb_4905():
    """
    Downloads the EMDB-4905 volume map and returns the file path.

    3D structure of horse spleen apoferritin determined using multifunctional
    graphene supports for electron cryomicroscopy. This molecule exhibits octahedral symmetry.

    :return: A 'Volume' instance.
    """
    file_path = fetch_data("emdb_4905.map")
    vol = Volume.load(file_path, symmetry_group="O")

    return vol


def emdb_10835():
    """
    Downloads the EMDB-10835 volume map and returns the file path.

    High resolution cryo-EM structure of urease from the pathogen Yersinia enterocolitica.
    This molecule exhibits tetrahedral symmetry.

    :return: A 'Volume' instance.
    """
    file_path = fetch_data("emdb_10835.map")
    vol = Volume.load(file_path, symmetry_group="T")

    return vol


def emdb_5778():
    """
    Downloads the EMDB-5778 volume map and returns the file path.

    Structure of the capsaicin receptor, TRPV1, determined by single particle electron cryo-microscopy.
    This molecule exhibits C4 symmetry.

    :return: A 'Volume' instance.
    """
    file_path = fetch_data("emdb_5778.map")
    vol = Volume.load(file_path, symmetry_group="C4")

    return vol


def emdb_6287():
    """
    Downloads the EMDB-6287 volume map and returns the file path.

    2.8 Angstrom resolution reconstruction of the T20S proteasome.
    This molecule exhibits D7 symmetry.

    :return: A 'Volume' instance.
    """
    file_path = fetch_data("emdb_6287.map")
    vol = Volume.load(file_path, symmetry_group="D7")

    return vol


def emdb_2824():
    """
    Downloads the EMDB-2824 volume map and returns the file path.

    Beta-galactosidase reconstruction.
    This molecule exhibits C2 symmetry.

    :return: A 'Volume' instance.
    """
    file_path = fetch_data("emdb_2824.map")
    vol = Volume.load(file_path, symmetry_group="C2")

    return vol


def emdb_14621():
    """
    Downloads the EMDB-14621 volume map and returns the file path.

    Map of SARSCoV2 spike protein.
    This molecule exhibits C3 symmetry.

    :return: A 'Volume' instance.
    """
    file_path = fetch_data("emdb_14621.map")
    vol = Volume.load(file_path, symmetry_group="C3")

    return vol


def emdb_2484():
    """
    Downloads the EMDB-2484 volume map and returns the file path.

    Pre-fusion structure of trimeric HIV-1 envelope glycoprotein determined by cryo-electron microscopy.
    This molecule exhibits C3 symmetry.

    :return: A 'Volume' instance.
    """
    file_path = fetch_data("emdb_2484.map")
    vol = Volume.load(file_path, symmetry_group="C3")

    return vol


def emdb_6458():
    """
    Downloads the EMDB-6458 volume map and returns the file path.

    Cryo-EM Structure of the Activated NAIP2/NLRC4 Inflammasome Reveals Nucleated Polymerization.
    This molecule exhibits C11 symmetry.

    :return: A 'Volume' instance.
    """
    file_path = fetch_data("emdb_6458.map")
    vol = Volume.load(file_path, symmetry_group="C11")

    return vol


def simulated_channelspin():
    """
    Downloads the Simulated ChannelSpin dataset and returns the file path.

    This dataset includes a stack of 54 volumes sized (54,54,54)
    and a corresponding stack of 10000 projection images (54,54).

    :return: Dictionary containing  Volume and Image instances,
        along with associated metadata fields in Numpy arrays.
    """
    file_path = fetch_data("simulated_channelspin.npz")
    # Use context manager so the file handle closes.
    with np.load(file_path) as data:
        # Convert to dict so that the entries can be modified
        data = dict(data)

    # Instantiate ASPIRE objects where appropriate
    data["vols"] = Volume(data["vols"])
    data["images"] = Image(data["images"])
    data["rots"] = Rotation(_LegacySimulation.rots_zyx_to_legacy_aspire(data["rots"]))

    return data
