from gemmi import cifs
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
       
    def _initialize_blocks(self):
        gemmi_doc = cif.read_file(filepath)
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
                if gemmi_item.loop is not None:
                    block_has_loop = True  
                    if block_has_pair:
                        raise StarFileError('Blocks with multiple loops and/or pairs are not supported')
                    loop_tags = gemmi_item.loop.tags 
                    loop_data = [0 for x in range(gemmi_item.loop.length)]  
                    for row in range(gemmi_item.loop.length):
                            loop_data[row] = [gemmi_item.loop.val(row, col) for col in range(gemmi_item.loopwidth)]

            
                    
