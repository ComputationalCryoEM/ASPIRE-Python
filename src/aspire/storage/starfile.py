from collections import OrderedDict

import pandas as pd
from gemmi import cif


class StarFileError(Exception):
    pass


class StarFile:
    def __init__(self, filepath=None, blocks=None):
        if (filepath is None) and (blocks is None):
            raise StarFileError(
                "Must specify a STAR file to read or pass an OrderedDict of dataframes"
            )
        if not (filepath is None) and not (blocks is None):
            raise StarFileError(
                "Pass either a STAR file OR an OrderedDict of dataframes"
            )

        if not (filepath is None):
            self.blocks = OrderedDict()
            filepath = str(filepath)
            self._initialize_blocks(filepath)

        elif not (blocks is None):
            self.blocks = blocks

    def _initialize_blocks(self, filepath):
        gemmi_doc = cif.read_file(filepath)
        for gemmi_block in gemmi_doc:
            block_has_pair = False
            block_has_loop = False
            pairs = {}
            loop_tags = []
            loop_data = []
            if gemmi_block.name == "#":
                gemmi_block.name = ""
            for gemmi_item in gemmi_block:
                if gemmi_item.pair is not None:
                    block_has_pair = True
                    if block_has_loop:
                        raise StarFileError(
                            "Blocks with multiple loops and/or pairs are not supported"
                        )
                    # assign key-value pair
                    if gemmi_item.pair[0] not in pairs:
                        pairs[gemmi_item.pair[0]] = gemmi_item.pair[1]
                    else:
                        raise StarFileError(
                            f"Duplicate key in pair: {gemmi_item.pair[0]}"
                        )
                if gemmi_item.loop is not None:
                    block_has_loop = True
                    if block_has_pair:
                        raise StarFileError(
                            "Blocks with multiple loops and/or pairs are not supported"
                        )
                    loop_tags = gemmi_item.loop.tags
                    loop_data = [0 for x in range(gemmi_item.loop.length())]
                    for row in range(gemmi_item.loop.length()):
                        loop_data[row] = [
                            gemmi_item.loop.val(row, col)
                            for col in range(gemmi_item.loop.width())
                        ]
            if block_has_pair:
                if gemmi_block.name not in self.blocks:
                    self.blocks[gemmi_block.name] = pd.DataFrame(
                        [pairs], columns=pairs.keys(), dtype=str
                    )
                else:
                    raise StarFileError(
                        f"Attempted overwrite of existing data block: {gemmi_block.name}"
                    )
            elif block_has_loop:
                if gemmi_block.name not in self.blocks:
                    self.blocks[gemmi_block.name] = pd.DataFrame(
                        loop_data, columns=loop_tags, dtype=str
                    )
                else:
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
