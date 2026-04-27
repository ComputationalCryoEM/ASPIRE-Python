"""
Legacy Preprocessing Pipeline
=============================

This notebook demonstrates reproducing preprocessing results from the
legacy ASPIRE MATLAB workflow system for the EMPIAR 10028 ribosome
dataset.

https://www.ebi.ac.uk/empiar/EMPIAR-10028
"""

# %%
# Source data and Preprocessing
# -----------------------------
#
# Load the data and apply preprocessing steps corresponding to the MATLAB prompts:
#
# >>> Phaseflip projections? (Y/N)? [Y] Y
# >>> Enter pixel size in Angstrom (-1 to read from STAR file): [-1.000000]
# >>> Number of projections to read? [105247]
# >>> Crop? (Y/N)? [Y] Y
# >>> Crop to size? [360]359
# >>> Downsample? (Y/N)? [Y] Y
# >>> Downsample to size? [360]179
# >>> Normalize background of images to variance 1? (Y/N)? [Y]
# >>> Prewhiten? (Y/N)? [Y]
# >>> Split data into groups? (Y/N)? [Y] N

from aspire.source import RelionSource

# Inputs
# Note the published ``shiny_2sets.star`` requires removal of a stray '9' character on line 5476.
starfile_in = "10028/data/shiny_2sets_fixed9.star"
# Caching, while not required, will increase speed in exchange for potentially increased memory usage.
src = RelionSource(starfile_in).cache()

# Define preprocessing steps.
src = src.phase_flip().cache()
src = src.crop_pad(359).cache()
src = src.legacy_downsample(179).cache()
src = src.legacy_normalize_background().cache()
src = src.legacy_whiten().cache()
src = src.invert_contrast().cache()

# Save the preprocessed images.
# `save_mode=single` will save a STAR file and single mrcs holding the image stack.
src.save("10028_legacy_preprocessed_179px.star", save_mode="single", overwrite=True)
