# dataset registry for common ASPIRE datasets
registry = {
    "emdb_2660.map": "49aecfd4efce09afc937d1786bbed6f18c2a353c73a4e16a643a304342d0660e",
}

registry_urls = {
    "emdb_2660.map": "https://zenodo.org/record/7730530/files/emd_2660.map.gz",
}

method_files_map = {
    "emdb_2660": ["emdb_2660.map"],
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
        path=pooch.os_cache("ASPIRE-data"),
        # env="ASPIRE_DATADIR", # This will be for configured cache.
        # The remote data is on Zenodo
        # base_url is a required param, even though we override this
        # using individual urls in the registry.
        base_url="https://zenodo.org/communities/computationalcryoem/",
        registry=registry,
        urls=registry_urls,
    )


def fetch_data(dataset_name):
    if data_fetcher is None:
        raise ImportError(
            "Missing optional dependency 'pooch' required "
            "for aspire.datasets module. Please use pip or "
            "conda to install 'pooch'."
        )

    return data_fetcher.fetch(dataset_name)


def emdb_2660():
    """
    Downloads the EMDB-2660 volume map and returns the file path.
    """
    return fetch_data("emdb_2660.map")
