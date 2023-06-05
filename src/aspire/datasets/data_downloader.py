from ._registry import registry, registry_urls

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
        raise ImportError("Missing optional dependency 'pooch' required "
                          "for aspire.datasets module. Please use pip or "
                          "conda to install 'pooch'.")

    return data_fetcher.fetch(dataset_name)


def emdb_2660():
    """
    Downloads the EMDB-2660 volume map and returns the file path.
    """
    return fetch_data("emdb_2660.map")
