import os
import shutil

import numpy as np
import pooch

from aspire import config
from aspire.downloader import file_to_method_map, registry, registry_urls
from aspire.volume import Volume

# Initialize pooch data fetcher instance.
data_fetcher = pooch.create(
    # Use the default cache folder for the operating system.
    # Pooch uses appdirs (https://github.com/ActiveState/appdirs) to
    # select an appropriate directory for the cache on each platform.
    path=config["common"]["cache_dir"].as_filename(),
    # The remote data is on Zenodo base_url is a required param,
    # even though we override using individual urls in the registry.
    base_url="https://zenodo.org/communities/computationalcryoem/",
    registry=registry,
    urls=registry_urls,
)


def fetch_data(dataset_name):
    """
    The "fetch" method returns the full path to the downloaded data file.
    """
    return data_fetcher.fetch(dataset_name)


def download_all():
    """
    Download all ASPIRE example data and return a dictionary of filepaths.
    """

    file_paths = {}
    for data_set in registry:
        path = fetch_data(data_set)
        name = file_to_method_map[data_set]
        file_paths[name] = path

    return file_paths


def clear_downloads():
    """
    Purge the downloads directory.
    """
    shutil.rmtree(data_fetcher.abspath)


def available_downloads():
    """
    List all available downloads
    """
    return list(file_to_method_map.values())


def show_downloads():
    """
    List all currently downloaded datasets.
    """

    data_dir = data_fetcher.abspath
    datasets = [
        f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))
    ]

    return datasets


def emdb_2660(dtype=None):
    """
    Downloads the EMDB-2660 volume map and returns a `Volume` instance.

    Cryo-EM structure of the Plasmodium falciparum 80S ribosome
    bound to the anti-protozoan drug emetine.

    :param dtype: Optionally set dtype for the `Volume`. Defaults to dtype of the volume map.
    :return: A `Volume` instance.
    """
    file_path = fetch_data("emdb_2660.map")
    vol = Volume.load(file_path, dtype=dtype)

    return vol


def emdb_8012():
    """
    Downloads the EMDB-8012 volume map and returns the file path.

    The overall structure of the yeast spliceosomal U4/U6.U5 tri-snRNP at 3.7 Angstrom.
    """
    return fetch_data("emdb_8012.map")


def emdb_2984():
    """
    Downloads the EMDB-2984 volume map and returns the file path.

    2.2 A resolution cryo-EM structure of beta-galactosidase in complex with a cell-permeant inhibitor.
    This molecule exhibits D2 symmetry.
    """
    return fetch_data("emdb_2984.map")


def emdb_8511():
    """
    Downloads the EMDB-8511 volume map and returns the file path.

    Structure of the human HCN1 hyperpolarization-activated cyclic nucleotide-gated ion channel.
    This molecule exhibits C4 symmetry.
    """
    return fetch_data("emdb_8511.map")


def emdb_3645():
    """
    Downloads the EMDB-3645 volume map and returns the file path.

    CryoEM density of TcdA1 in prepore state (SPHIRE tutorial).
    This molecule exhibits C5 symmetry.
    """
    return fetch_data("emdb_3645.map")


def emdb_4905():
    """
    Downloads the EMDB-4905 volume map and returns the file path.

    3D structure of horse spleen apoferritin determined using multifunctional
    graphene supports for electron cryomicroscopy. This molecule exhibits octahedral symmetry.
    """
    return fetch_data("emdb_4905.map")


def emdb_10835():
    """
    Downloads the EMDB-10835 volume map and returns the file path.

    High resolution cryo-EM structure of urease from the pathogen Yersinia enterocolitica.
    This molecule exhibits tetrahedral symmetry.
    """
    return fetch_data("emdb_10835.map")


def emdb_5778():
    """
    Downloads the EMDB-5778 volume map and returns the file path.

    Structure of the capsaicin receptor, TRPV1, determined by single particle electron cryo-microscopy.
    This molecule exhibits C4 symmetry.
    """
    return fetch_data("emdb_5778.map")


def emdb_6287():
    """
    Downloads the EMDB-6287 volume map and returns the file path.

    2.8 Angstrom resolution reconstruction of the T20S proteasome.
    This molecule exhibits D7 symmetry.
    """
    return fetch_data("emdb_6287.map")


def emdb_2824():
    """
    Downloads the EMDB-2824 volume map and returns the file path.

    Beta-galactosidase reconstruction.
    This molecule exhibits C2 symmetry.
    """
    return fetch_data("emdb_2824.map")


def emdb_14621():
    """
    Downloads the EMDB-14621 volume map and returns the file path.

    Map of SARSCoV2 spike protein.
    This molecule exhibits C3 symmetry.
    """
    return fetch_data("emdb_14621.map")


def emdb_2484():
    """
    Downloads the EMDB-2484 volume map and returns the file path.

    Pre-fusion structure of trimeric HIV-1 envelope glycoprotein determined by cryo-electron microscopy.
    This molecule exhibits C3 symmetry.
    """
    return fetch_data("emdb_2484.map")


def emdb_6458():
    """
    Downloads the EMDB-6458 volume map and returns the file path.

    Cryo-EM Structure of the Activated NAIP2/NLRC4 Inflammasome Reveals Nucleated Polymerization.
    This molecule exhibits C11 symmetry.
    """
    return fetch_data("emdb_6458.map")
