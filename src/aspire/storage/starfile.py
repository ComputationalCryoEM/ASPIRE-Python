import logging
import os
from collections import OrderedDict

import pandas as pd
from gemmi import cif

logger = logging.getLogger(__name__)


class StarFileError(Exception):
    pass


class StarFile:
    def __init__(self, filepath="", blocks=None):
        """
        Initialize either from a path to a STAR file or from an OrderedDict of dataframes
        """
        # if blocks are given, set self.blocks, otherwise initialize an empty OrderedDict()
        self.blocks = blocks or OrderedDict()
        # if constructing from blocks, must switch pandas dtype to str
        # otherwise comparison with StarFiles read from files will fail
        # due to different data types
        self.blocks = OrderedDict(
            {
                key: (block.astype(str) if isinstance(block, pd.DataFrame) else block)
                for (key, block) in self.blocks.items()
            }
        )
        self.filepath = str(filepath)
        # raise an error if blocks and a filepath are both passed
        if bool(self.filepath) and bool(len(self.blocks)):
            raise StarFileError(
                "Pass either a path to a STAR file or an OrderedDict of Pandas DataFrames, not both"
            )
        if self.filepath:
            if not os.path.exists(self.filepath):
                logger.error(f"Could not open {self.filepath}")
                raise FileNotFoundError
            self._initialize_blocks()
        logger.info(f"Created {self}")

    def _initialize_blocks(self):
        """
        Converts a gemmi Document object representing the .star file
        at self.filepath into an OrderedDict of pandas dataframes, each of which represents one block in the .star file
        """
        logger.info(f"Parsing star file at: {self.filepath}")
        gemmi_doc = cif.read_file(self.filepath)
        # iterate over gemmi Block objects in the gemmi Document
        for gemmi_block in gemmi_doc:
            # iterating over gemmi Block objects yields Item objects
            # Items can have a Loop object and/or a Pair object
            # Loops correspond to the regular loop_ structure in a STAR file
            # Pairs have type List[str[2]] and correspond to a non-loop key value
            # pair in a STAR file, e.g.
            # _field1 \t 'value' #1

            # Our model of the .star file only allows a block to be one or the other
            block_has_pair = False
            block_has_loop = False
            # populated if this block has a pair
            pairs = {}
            # populated if this block as a loop
            loop_tags = []
            loop_data = []
            # correct for GEMMI default behavior
            # if a block is called 'data_' in the .star file, GEMMI names it '#'
            # but we want to name it '' for consistency
            if gemmi_block.name == "#":
                gemmi_block.name = ""
            for gemmi_item in gemmi_block:
                if gemmi_item.pair is not None:
                    block_has_pair = True
                    # if we find both a pair and a loop raise an error
                    if block_has_loop:
                        raise StarFileError(
                            "Blocks with multiple loops and/or pairs are not supported"
                        )
                    # assign key-value pair to dictionary
                    pair_key, pair_val = gemmi_item.pair
                    if pair_key not in pairs:
                        # read in as str because we do not want type conversion
                        pairs[pair_key] = str(pair_val)
                    else:
                        raise StarFileError(
                            f"Duplicate key in pair: {gemmi_item.pair[0]}"
                        )
                if gemmi_item.loop is not None:
                    block_has_loop = True
                    # if we find both a pair and a loop raise an error
                    if block_has_pair:
                        raise StarFileError(
                            "Blocks with multiple loops and/or pairs are not supported"
                        )
                    loop_tags = gemmi_item.loop.tags
                    # convert loop data to a list of lists
                    # using the .val(row, col) method of gemmi's Loop class
                    loop_data = [None] * gemmi_item.loop.length()
                    for row in range(gemmi_item.loop.length()):
                        loop_data[row] = [
                            gemmi_item.loop.val(row, col)
                            for col in range(gemmi_item.loop.width())
                        ]
            if block_has_pair:
                if gemmi_block.name not in self.blocks:
                    # represent a set of pairs by a dictionary
                    self.blocks[gemmi_block.name] = pairs
                else:
                    # enforce unique block names (keys of StarFile.block OrderedDict)
                    raise StarFileError(
                        f"Attempted overwrite of existing data block: {gemmi_block.name}"
                    )
            elif block_has_loop:
                if gemmi_block.name not in self.blocks:
                    # initialize DF from list of lists
                    # read in with dtype=str because we do not want type conversion
                    self.blocks[gemmi_block.name] = pd.DataFrame(
                        loop_data, columns=loop_tags, dtype=str
                    )
                else:
                    # enforce unique block names (keys of StarFile.block OrderedDict)
                    raise StarFileError(
                        f"Attempted overwrite of existing data block: {gemmi_block.name}"
                    )

    def write(self, filepath):
        """
        Converts `blocks` to a gemmi Document and writes to a starfile at the given filepath.
        """
        # create an empty Document
        _doc = cif.Document()
        filepath = str(filepath)
        for name, block in self.blocks.items():
            # construct new empty block
            _block = _doc.add_new_block(name)
            # if this block (loop or pair) is empty, continue
            if len(block) == 0:
                continue
            # are we constructing a loop (DataFrame) or a pair (Dictionary)?
            if isinstance(block, dict):
                for key, value in block.items():
                    # simply assign one pair item for each dict entry
                    # write out as str because we do not want type conversion
                    _block.set_pair(key, str(value))
            elif isinstance(block, pd.DataFrame):
                # initialize loop with column names
                _loop = _block.init_loop("", list(block.columns))
                for row in block.values.tolist():
                    # write out as str because we do not want type conversion
                    row = [str(row[x]) for x in range(len(row))]
                    _loop.add_row(row)
            else:
                raise StarFileError(f"Unsupported type for block {name}: {type(block)}")

        _doc.write_file(filepath)

    def get_block_by_index(self, index):
        """
        Retrieve a DataFrame representing a star file block by its position in the starfile
        """
        return self.blocks[list(self.blocks.keys())[index]]

    def __getitem__(self, key):
        """
        Retrieve the star file block with name `key`
        """
        return self.blocks[key]

    def __setitem__(self, key, value):
        """
        Pass in a Pandas Dataframe or dictionary to add a block named `key` with values `value`
        """
        self.blocks[key] = value

    def __repr__(self):
        return "StarFile with blocks: " + ", ".join(self.blocks.keys())

    def __iter__(self):
        return self.blocks.items().__iter__()

    def __len__(self):
        return len(self.blocks)

    def __eq__(self, other):
        if not len(self) == len(other):
            return False
        self_list = list(self.blocks.items())
        other_list = list(other.blocks.items())
        for i in range(len(self_list)):
            # test whether block names are the same
            if not self_list[i][0] == other_list[i][0]:
                return False
            # test whether blocks are both DFs or dicts
            if not isinstance(self_list[i][1], type(other_list[i][1])):
                return False
            # finally, compare the objects themselves
            if isinstance(self_list[i][1], pd.DataFrame):
                # test using pandas DataFrame.equals()
                if not self_list[i][1].equals(other_list[i][1]):
                    return False
            else:
                # test dict equality
                if not self_list[i][1] == other_list[i][1]:
                    return False
        return True
