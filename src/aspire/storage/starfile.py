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
        self.blocks = blocks or OrderedDict()
        self.filepath = str(filepath)
        if not (bool(self.filepath) ^ len(self.blocks)):
            raise StarFileError(
                "Pass a path to a STAR file or an OrderedDict of Pandas DataFrames"
            )
        if self.filepath:
            if not os.path.exists(self.filepath):
                logger.error(f"Could not open {self.filepath}")
                raise FileNotFoundError
            self._initialize_blocks()

    def _initialize_blocks(self):
        """
        This method converts a gemmi Document object representing the .star file
        at self.filepath into an OrderedDict of pandas dataframes, each of which represents one block in the .star file
        """
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
            pairs = {}
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
                    # assign key-value pair
                    pair_key, pair_val = gemmi_item.pair
                    if pair_key not in pairs:
                        pairs[pair_key] = pair_val
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
                    # initialize DF from dictionary of kv pairs
                    self.blocks[gemmi_block.name] = pd.DataFrame(
                        [pairs], columns=pairs.keys(), dtype=str
                    )
                else:
                    # enforce unique keys
                    raise StarFileError(
                        f"Attempted overwrite of existing data block: {gemmi_block.name}"
                    )
            elif block_has_loop:
                if gemmi_block.name not in self.blocks:
                    # initialize DF from list of lists
                    self.blocks[gemmi_block.name] = pd.DataFrame(
                        loop_data, columns=loop_tags, dtype=str
                    )
                else:
                    # enforce unique keys
                    raise StarFileError(
                        f"Attempted overwrite of existing data block: {gemmi_block.name}"
                    )

    def write(self, filepath):
        # construct empty document
        _doc = cif.Document()
        filepath = str(filepath)
        for name, df in self.blocks.items():
            # construct new empty block
            _block = _doc.add_new_block(name)
            # are we constructing a pair (df has 1 row) or a loop (df has >1 rows)
            if len(df) == 0:
                continue
            if len(df) == 1:
                for col in list(df):
                    # key is the column label. value is value in the df at row 0, column col
                    _block.set_pair(col, str(df.at[0, col]))
            elif len(df) > 1:
                # initialize loop with column names
                _loop = _block.init_loop("", list(df.columns))
                for row in df.values.tolist():
                    row = [str(row[x]) for x in range(len(row))]
                    _loop.add_row(row)

        _doc.write_file(filepath)

    def get_block_by_index(self, index):
        return self.blocks[list(self.blocks.keys())[index]]

    def __getitem__(self, key):
        return self.blocks[key]

    def __setitem__(self, key, value):
        self.blocks[key] = value

    def __iter__(self):
        return self.blocks.items().__iter__()

    def __len__(self):
        return len(self.blocks)

    def __eq__(self, other):
        if not len(self) == len(other):
            return False
        self_list = list(self.blocks.keys())
        other_list = list(other.blocks.keys())
        for i in range(len(self_list)):
            if not self_list[i][0] == other_list[i][0]:
                return False
            if not self_list[i][1] == other_list[i][1]:
                return False
        return True
