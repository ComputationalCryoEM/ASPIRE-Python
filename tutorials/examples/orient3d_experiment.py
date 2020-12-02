"""
This script illustrates the estimation of orientation angles for experimental dataset
"""

from aspire.abinitio import CLSyncVoting
from aspire.source.relion import RelionSource

# Set input path and files and initialize other parameters
#DATA_FOLDER = '/path/to/untarred/empiar/dataset/'
DATA_FOLDER = "/tigress/junchaox/CryoEMdata/empiar10028"
#STARFILE_IN = '/path/to/untarred/empiar/dataset/input.star'
STARFILE_IN = "/tigress/junchaox/CryoEMdata/empiar10028/particles_denoise.star"
#STARFILE_OUT = '/path/to/output/ouput.star'
STARFILE_OUT = "/tigress/junchaox/CryoEMdata/empiar10028/Orient3D/particles_orient3D.star"
PIXEL_SIZE = 1.34
MAX_ROWS = 1024

# Create a source object for 2D images
print(f'Read in images from {STARFILE_IN}.')
source = RelionSource(
    STARFILE_IN,
    DATA_FOLDER,
    pixel_size=PIXEL_SIZE,
    max_rows=MAX_ROWS
)


# Estimate rotation matrices
print("Estimate rotation matrices.")
orient_est = CLSyncVoting(source)
orient_est.estimate_rotations()

# Create new source object and save Estimate rotation matrices
print("Save Estimate rotation matrices.")
orient_est_src = orient_est.save_rotations()

# Output orientational angles
print("Save orientational angles to STAR file.")
orient_est_src.save(STARFILE_OUT)




