from gemmi import cif
import  pandas  as pd
import  os
from collections import OrderedDict

class StarFileError(Exception):
    def __init__(self, message):
        self.message = message

class StarFile:
    def __init__(self, filepath):
        self.filepath = filepath
        self.blocks = OrderedDict()
        self._initialize_blocks()
       
    def _initialize_blocks(self):
        gemmi_doc = cif.read_file(self.filepath)
        for gemmi_block in gemmi_doc:
            block_has_pair = False
            block_has_loop = False
            pairs = {}
            loop_tags = []
            loop_data = []              
            for gemmi_item in gemmi_block:
                if gemmi_item.pair is not None:
                    block_has_pair = True
                    if block_has_loop:
                        raise StarFileError('Blocks with multiple loops and/or pairs are not supported')
                    # assign key-value pair
                    if gemmi_item.pair[0] not in pairs:         
                        pairs[gemmi_item.pair[0]] = gemmi_item.pair[1] 
                    else:
                        raise StarFileError(f'Duplicate key in pair: {gemmi_item.pair[0]}')
                    if gemmi_block.name not in self.blocks:
                        self.blocks[gemmi_block.name] = pd.DataFrame(list(pairs.items()), columns = pairs.keys())                                  
                    else:
                        raise StarFileError(f'Attempted overwrite of existing data block: {gemmi_block.name}')
                if gemmi_item.loop is not None:
                    block_has_loop = True  
                    if block_has_pair:
                        raise StarFileError('Blocks with multiple loops and/or pairs are not supported')
                    loop_tags = gemmi_item.loop.tags 
                    loop_data = [0 for x in range(gemmi_item.loop.length())]  
                    for row in range(gemmi_item.loop.length()):
                            loop_data[row] = [gemmi_item.loop.val(row, col) for col in range(gemmi_item.loop.width())]
                    if gemmi_block.name not in self.blocks:
                        self.blocks[gemmi_block.name] = pd.DataFrame(loop_data, columns = loop_tags)
                    else:
                        raise StarFileError(f'Attempted overwrite of existing data block: {gemmi_block.name}') 
    @staticmethod
    def write(self, data, filepath):
        # construct empty document
        _doc = cif.Document()
        for name, df in data.items():
            # construct new empty block
            _block = _doc.add_new_block(name)
            # are we constructing a pair (df has 1 row) or a loop (df has >1 rows)
            if len(df) == 0:
                continue
            if len(df) == 1:
                for col in df.columns:
                    # key is the column label. value is value in the df at row 0, column col
                    _block.set_pair(col, df.at(0, col))
                    continue
            elif len(df) > 1:
                # initialize loop with column names
                _loop = _block.init_loop('', list(df.columns))
                for row in df.values.tolist():
                    _loop.add_row(row)
        
        _doc.write_file(filepath)
        
        
            
    def __getitem__(self, key):
        return self.blocks[key]
        
    def __setitem__(self, key, value):
        self.blocks[key] = value

    def __iter__(self):
        return iter(self.blocks) 
