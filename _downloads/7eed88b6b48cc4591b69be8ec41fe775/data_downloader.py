"""
Data Downloader
===============

This tutorial reviews ASPIRE's data downloading utility.
"""

# %%
# Data Downloader Introduction
# ----------------------------
# ASPIRE provides a data downloading utility for downloading and caching some
# common example datasets. When applicable, the data will be loaded as an appropriate
# ASPIRE data type, such a as a ``Volume``, ``Image``, or ``ImageSource`` instance.
#
# Below we take a look at the current list of datasets available for download.
# sphinx_gallery_start_ignore
# flake8: noqa
# sphinx_gallery_end_ignore
from aspire import downloader

downloader.available_downloads()

# %%
# Data Caching
# ------------
# By default, the data downloader selects an appropriate cache directory based
# on the user operating system. Usually, the locations will be the following:

# %%
# .. list-table:: Default Cache Locations
#    :header-rows: 1
#
#    * - OS
#      - Cache Location
#    * - Mac
#      - ~/Library/Caches/ASPIRE-data
#    * - Linux
#      - ~/.cache/ASPIRE-data
#    * - Windows
#      - C:\\Users\\<user>\\AppData\\Local\\<AppAuthor>\\ASPIRE-data\\Cache

# %%
# The cache location is configurable and can be overridden by creating a custom
# ``config.yaml``. For example, to change the cache folder to ``/tmp/ASPIRE-data`` create
# ``$HOME/.config/ASPIRE/config.yaml`` with the following contents:
#
#    .. code-block:: yaml
#
#      cache:
#        cache_dir: /tmp/ASPIRE-data
#
# See the `ASPIRE Conguration
# <https://computationalcryoem.github.io/ASPIRE-Python/auto_tutorials/configuration.html>`_
# tutorial for more details on customizing your config.

# %%
# .. note::
#     All downloads can be cleared from the cache with the ``remove_downloads()`` method::
#
#         downloader.remove_downloads()

# %%
# Download an Example Dataset
# ---------------------------
# Below we will download ``emdb_2660``, a high resolution volume map of the 80s Ribosome
# which is sourced from EMDB at https://www.ebi.ac.uk/emdb/EMD-2660. This volume map will
# load as an instance of ASPIRE's ``Volume`` class.
vol = downloader.emdb_2660()
vol

# %%
# View the Data
# ------------------
# We can take a peek at this data by viewing some projections of the volume.
import numpy as np

from aspire.utils import Rotation

rots = Rotation.about_axis("y", [0, np.pi / 2])
projection = vol.project(rots)
projection.show()
