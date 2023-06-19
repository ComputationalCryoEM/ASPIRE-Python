from aspire import config

# dataset registry for ASPIRE example data.
registry = {
    "emdb_2660.map": "49aecfd4efce09afc937d1786bbed6f18c2a353c73a4e16a643a304342d0660e",
    "emdb_8012.map": "85a1c9ab958b1dd051d011515212d58decaf2537351b9c016acd3e5852e30d63",
}

registry_urls = {
    "emdb_2660.map": "https://zenodo.org/record/7730530/files/emd_2660.map.gz",
    "emdb_8012.map": "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-8012/map/emd_8012.map.gz",
}

method_files_map = {
    "emdb_2660": ["emdb_2660.map"],
    "emdb_8012": ["emdb_8012.map"],
}


try:
    import pooch
except ImportError:
    pooch = None
    data_fetcher = None
else:
    data_fetcher = pooch.create(
        # Use the default cache folder for the operating system
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
    if data_fetcher is None:
        raise ImportError(
            "Missing optional dependency 'pooch' required "
            "for ASPIRE example data. Please use pip or "
            "conda to install 'pooch'."
        )
    # The "fetch" method returns the full path to the downloaded data file.
    return data_fetcher.fetch(dataset_name)


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
