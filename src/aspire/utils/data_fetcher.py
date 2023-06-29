import pooch

from aspire import config

# dataset registry for ASPIRE example data.
registry = {
    "emdb_2660.map": "49aecfd4efce09afc937d1786bbed6f18c2a353c73a4e16a643a304342d0660e",
    "emdb_8012.map": "85a1c9ab958b1dd051d011515212d58decaf2537351b9c016acd3e5852e30d63",
    "emdb_2984.map": "0194f44cb28f2a8daa5d477d25852de9cc81ed093487de181ee4b30b0d77ef90",
    "emdb_8511.map": "1f03ec4a0cadb407b6b972c803ffe1e97ff5087d4c2ce9fec2c404747a7fb3fe",
    "emdb_3645.map": "a574dba9657d44665b87f368f7379e97b1a33fe6ac2540478a3707f5ea840f12",
    "emdb_4905.map": "fe9ce303b43b11ccf253c8786b339cea3273ef70ff49dd6099155d576181f3c2",
    "emdb_10835.map": "dec12cdda4c36253a15f6f8105416020855bd51eb860ac5eb04b37b000ac9156",
    "emdb_5778.map": "877cbe37b86561c3dfb255aa2308fefcdd8f51f91928b17c2ef5c8dd3afaaef7",
    "emdb_6287.map": "81463aa6d024c80efcd19aa9b5ac58f3b3464af56e1ef0f104bd25071acc9204",
    "emdb_2824.map": "7682e1ef6e5bc9f2de9edcf824a03e454ef9cb1ca33bc12920633559f7f826e4",
    "emdb_14621.map": "b45774245c2bd5e1a44e801b8fb1705a44d5850631838d060294be42e34a6900",
    "emdb_2484.map": "6a324e23352bea101c191d5e854026162a5a9b0b8fc73ac5a085cc22038e1999",
    "emdb_6458.map": "645208af6d36bbd3d172c549e58d387b81142fd320e064bc66105be0eae540d1",
}

registry_urls = {
    "emdb_2660.map": "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-2660/map/emd_2660.map.gz",
    "emdb_8012.map": "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-8012/map/emd_8012.map.gz",
    "emdb_2984.map": "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-2984/map/emd_2984.map.gz",
    "emdb_8511.map": "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-8511/map/emd_8511.map.gz",
    "emdb_3645.map": "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-3645/map/emd_3645.map.gz",
    "emdb_4905.map": "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-4905/map/emd_4905.map.gz",
    "emdb_10835.map": "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-10835/map/emd_10835.map.gz",
    "emdb_5778.map": "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-5778/map/emd_5778.map.gz",
    "emdb_6287.map": "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-6287/map/emd_6287.map.gz",
    "emdb_2824.map": "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-2824/map/emd_2824.map.gz",
    "emdb_14621.map": "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-14621/map/emd_14621.map.gz",
    "emdb_2484.map": "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-2484/map/emd_2484.map.gz",
    "emdb_6458.map": "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-6458/map/emd_6458.map.gz",
}

file_to_method_map = {
    "emdb_2660.map": "emdb_2660",
    "emdb_8012.map": "emdb_8012",
    "emdb_2984.map": "emdb_2984",
    "emdb_8511.map": "emdb_8511",
    "emdb_3645.map": "emdb_3645",
    "emdb_4905.map": "emdb_4905",
    "emdb_10835.map": "emdb_10835",
    "emdb_5778.map": "emdb_5778",
    "emdb_6287.map": "emdb_6287",
    "emdb_2824.map": "emdb_2824",
    "emdb_14621.map": "emdb_14621",
    "emdb_2484.map": "emdb_2484",
    "emdb_6458.map": "emdb_6458",
}


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


def emdb_2660():
    """
    Downloads the EMDB-2660 volume map and returns the file path.

    Cryo-EM structure of the Plasmodium falciparum 80S ribosome
    bound to the anti-protozoan drug emetine.
    """
    return fetch_data("emdb_2660.map")


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
