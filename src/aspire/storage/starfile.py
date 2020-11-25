import logging
import os.path
from collections import OrderedDict

import mrcfile
import numpy as np
import pandas as pd

from aspire.storage import MrcStats

logger = logging.getLogger(__name__)


class StarFileBlock:
    def __init__(self, loops, name="", properties=None):
        # Note: StarFile data blocks may have have key=>value pairs that start with a '_'.
        # We serve these up to the user using getattr.
        # To avoid potential conflicts with our custom
        # attributes here, we simply make them 'public' without a leading underscore.
        self.loops = loops
        self.name = name
        self.properties = properties

    def __repr__(self):
        return f"StarFileBlock (name={self.name}) with {len(self.loops)} loops"

    def __getattr__(self, name):
        return self.properties[name]

    def __len__(self):
        return len(self.loops)

    def __getitem__(self, item):
        return self.loops[item]

    def __eq__(self, other):
        return (
            self.name == other.name
            and self.properties == other.properties
            and all([all(l1 == l2) for l1, l2 in zip(self.loops, other.loops)])
        )


class StarFile:
    def __init__(self, starfile_path=None, blocks=None):

        self.blocks = OrderedDict()

        if starfile_path is not None:
            self.init_from_starfile(starfile_path)
        elif blocks is not None:
            self.init_from_blocks(blocks)
        else:
            raise RuntimeError("Invalid constructor.")

    def init_from_starfile(self, starfile_path):
        """
        Initalize a StarFile from a star file at a given path
        :param starfile_path: Path to saved starfile.
        :return: An initialized StarFile object
        """
        logger.info(f"Parsing starfile at path {starfile_path}")
        with open(starfile_path, "r") as f:

            blocks = []  # list of StarFileBlock objects
            block_name = ""  # name of current block
            properties = {}  # key value mappings to add to current block

            loops = []  # a list of DataFrames
            in_loop = False  # whether we're inside a loop
            field_names = []  # current field names inside a loop
            rows = []  # rows to add to current loop

            for i, line in enumerate(f):
                line = line.strip()

                if line.startswith("#"):
                    continue

                # When in a 'loop', any blank line implies we break out of the loop
                if not line:
                    if in_loop:
                        if rows:  # We have accumulated data for a loop
                            loops.append(
                                pd.DataFrame(rows, columns=field_names, dtype=str)
                            )
                            field_names = []
                            rows = []
                            in_loop = False

                elif line.startswith("data_"):
                    if loops or properties:
                        blocks.append(
                            StarFileBlock(loops, name=block_name, properties=properties)
                        )
                        loops = []
                        properties = {}
                    block_name = line[
                        5:
                    ]  # note: block name might be, and most likely would be blank

                elif line.startswith("loop_"):
                    in_loop = True

                elif line.startswith("_"):  # We have a field
                    if in_loop:
                        field_names.append(line.split()[0])
                    else:
                        k, v = line.split()[:2]
                        properties[k] = v

                else:
                    # we're looking at a data row
                    tokens = line.split()
                    if len(tokens) < len(field_names):
                        logger.warning(
                            f"Line {i} - Expected {len(field_names)} values, got {len(tokens)}."
                        )
                        tokens.extend([""] * (len(field_names) - len(tokens)))
                    else:
                        tokens = tokens[: len(field_names)]  # ignore any extra tokens

                    rows.append(tokens)

            # Any pending rows to be added?
            if rows:
                loops.append(pd.DataFrame(rows, columns=field_names, dtype=str))

            # Any pending loops/properties to be added?
            if loops or properties:
                blocks.append(
                    StarFileBlock(loops, name=block_name, properties=properties)
                )

            logger.info("StarFile parse complete")

        logger.info("Initializing StarFile object from data")
        self.init_from_blocks(blocks)
        logger.info(f"Created <{self}>")

    def init_from_blocks(self, blocks):
        """
        Initialize a StarFile from a list of blocks
        :param blocks: An iterable of StarFileBlock objects
        :return: An initialized StarFile object
        """
        for block in blocks:
            self.blocks[block.name] = block

    def __repr__(self):
        return f"StarFile with {len(self.blocks)} blocks"

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.blocks[item]
        else:
            return self.blocks[list(self.blocks.keys())[item]]

    def __len__(self):
        return len(self.blocks)

    def __eq__(self, other):
        return all(b1 == b2 for b1, b2 in zip(self.blocks, other.blocks))

    def save(self, f):
        for block in self:
            f.write(f"data_{block.name}\n\n")
            if block.properties is not None:
                for k, v in block.properties.items():
                    f.write(f"{k} {v}\n")
            f.write("\n")
            for loop in block:
                f.write("loop_\n")
                for col in loop.columns:
                    f.write(f"{col}\n")
                for _, row in loop.iterrows():
                    f.write(" ".join(map(str, row)) + "\n")
                f.write("\n")


def save_star(
    image_source, starfile_filepath, batch_size=1024, save_mode=None, overwrite=False
):
    """
    Save an ImageSource to a STAR file + individual .mrcs files
    Note that .mrcs files are saved at the same location as the STAR file.

    :param image_source: The ImageSource object to save
    :param starfile_filepath: Path to STAR file where we want to save image_source
    :param batch_size: Batch size of images to query from the `ImageSource` object. Every `batch_size` rows,
        entries are written to STAR file, and the `.mrcs` files saved.
    :param save_mode: Whether to save all images in a single or multiple files in batch size.
    :param overwrite: Whether to overwrite any .mrcs files found at the target location.
    :return: None
    """

    # TODO: Accessing protected member - provide a way to get a handle on the _metadata attribute.
    df = image_source._metadata.copy()
    # Drop any column that doesn't start with a *single* underscore
    df = df.drop(
        [
            str(col)
            for col in df.columns
            if not col.startswith("_") or col.startswith("__")
        ],
        axis=1,
    )

    # Create a new column that we will be populating in the loop below
    df["_rlnImageName"] = ""

    with open(starfile_filepath, "w") as f:
        if save_mode == "single":
            # Save all images into one single mrc file

            # First, construct name for mrc file
            fdir = os.path.dirname(starfile_filepath)
            fname = os.path.basename(starfile_filepath)
            fstem = os.path.splitext(fname)[0]
            mrcs_filename = f"{fstem}_{0}_{image_source.n-1}.mrcs"
            mrcs_filepath = os.path.join(fdir, mrcs_filename)

            # Then set name in dataframe for the StarFile
            df["_rlnImageName"][0 : image_source.n] = pd.Series(
                [f"{j + 1:06}@{mrcs_filename}" for j in range(image_source.n)]
            )

            # Open new MRC file
            with mrcfile.new_mmap(
                mrcs_filepath,
                shape=(image_source.n, image_source.L, image_source.L),
                mrc_mode=2,
                overwrite=overwrite,
            ) as mrc:

                stats = MrcStats()
                # Loop over source setting data into mrc file
                for i_start in np.arange(0, image_source.n, batch_size):
                    i_end = min(image_source.n, i_start + batch_size)
                    num = i_end - i_start
                    logger.info(
                        f"Saving ImageSource[{i_start}-{i_end-1}] to {mrcs_filepath}"
                    )
                    datum = image_source.images(start=i_start, num=num).data.astype(
                        "float32"
                    )

                    # Assign to mrcfile
                    mrc.data[i_start:i_end] = datum

                    # Accumulate stats
                    stats.push(datum)

                # To be safe, explicitly set the header
                #   before the mrc file context closes.
                mrc.update_header_from_data()

                # Also write out updated statistics for this mrc.
                #   This should be an optimization over mrc.update_header_stats
                #   for large arrays.
                stats.update_header(mrc)

        else:
            # save all images into multiple mrc files in batch size
            for i_start in np.arange(0, image_source.n, batch_size):
                i_end = min(image_source.n, i_start + batch_size)
                num = i_end - i_start
                mrcs_filename = (
                    os.path.splitext(os.path.basename(starfile_filepath))[0]
                    + f"_{i_start}_{i_end-1}.mrcs"
                )
                mrcs_filepath = os.path.join(
                    os.path.dirname(starfile_filepath), mrcs_filename
                )

                logger.info(
                    f"Saving ImageSource[{i_start}-{i_end-1}] to {mrcs_filepath}"
                )
                im = image_source.images(start=i_start, num=num)
                im.save(mrcs_filepath, overwrite=overwrite)

                df["_rlnImageName"][i_start:i_end] = pd.Series(
                    ["{0:06}@{1}".format(j + 1, mrcs_filepath) for j in range(num)]
                )

        # initial the star file object and save it
        starfile = StarFile(blocks=[StarFileBlock(loops=[df])])
        starfile.save(f)
