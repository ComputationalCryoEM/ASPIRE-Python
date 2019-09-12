import os.path
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class StarfileBlock:
    def __init__(self, loops, name='', properties=None):
        # Note: Starfile data blocks may have have key=>value pairs that start with a '_'.
        # We serve these up to the user using getattr.
        # To avoid potential conflicts with our custom
        # attributes here, we simply make them 'public' without a leading underscore.
        self.loops = loops
        self.name = name
        self.properties = properties

    def __repr__(self):
        return f'StarfileBlock (name={self.name}) with {len(self.loops)} loops'

    def __getattr__(self, name):
        return self.properties[name]

    def __len__(self):
        return len(self.loops)

    def __getitem__(self, item):
        return self.loops[item]

    def __eq__(self, other):
        return self.name == other.name and self.properties == other.properties and \
               all([all(l1 == l2) for l1, l2 in zip(self.loops, other.loops)])


class Starfile:
    def __init__(self, starfile_path=None, blocks=None, loop=None, dataframe=None, block_name=''):

        self.blocks = self.block_names = None

        if starfile_path is not None:
            self.init_from_starfile(starfile_path)
        elif blocks is not None:
            self.init_from_blocks(blocks)
        elif loop is not None:
            self.init_from_loop(loop, block_name)
        elif dataframe is not None:
            self.init_from_dataframe(dataframe, block_name)
        else:
            raise RuntimeError('Invalid constructor.')

    def init_from_starfile(self, starfile_path):
        """
        Initalize a Starfile from a star file at a given path
        :param starfile_path: Path to saved starfile.
        :return: An initialized Starfile object
        """
        logger.info(f'Parsing starfile at path {starfile_path}')
        with open(starfile_path, 'r') as f:

            blocks = []       # list of StarfileBlock objects
            block_name = ''   # name of current block
            properties = {}   # key value mappings to add to current block

            loops = []        # a list of DataFrames
            in_loop = False   # whether we're inside a loop
            field_names = []  # current field names inside a loop
            rows = []         # rows to add to current loop

            for i, line in enumerate(f):
                line = line.strip()

                # When in a 'loop', any blank line implies we break out of the loop
                if not line:
                    if in_loop:
                        if rows:  # We have accumulated data for a loop
                            loops.append(pd.DataFrame(rows, columns=field_names, dtype=str))
                            field_names = []
                            rows = []
                            in_loop = False

                elif line.startswith('data_'):
                    if loops or properties:
                        blocks.append(StarfileBlock(loops, name=block_name, properties=properties))
                        loops = []
                        properties = {}
                    block_name = line[5:]  # note: block name might be, and most likely would be blank

                elif line.startswith('loop_'):
                    in_loop = True

                elif line.startswith('_'):  # We have a field
                    if in_loop:
                        field_names.append(line.split()[0])
                    else:
                        k, v = line.split()[:2]
                        properties[k] = v

                else:
                    # we're looking at a data row
                    tokens = line.split()
                    assert len(tokens) == len(field_names), \
                        f'Error at line {i}. Expected {len(field_names)} values, got {len(tokens)}.'

                    rows.append(tokens)

            # Any pending rows to be added?
            if rows:
                loops.append(pd.DataFrame(rows, columns=field_names, dtype=str))

            # Any pending loops/properties to be added?
            if loops or properties:
                blocks.append(StarfileBlock(loops, name=block_name, properties=properties))

            logger.info(f'Starfile parse complete')

        logger.info(f'Initializing Starfile object from data')
        self.init_from_blocks(blocks)
        logger.info(f'Created <{self}>')

    def init_from_blocks(self, blocks):
        """
        Initialize a Starfile from a list of blocks
        :param blocks: A list of StarfileBlock objects
        :return: An initialized Starfile object
        """
        self.blocks = blocks
        self.block_names = [block.name for block in self.blocks]

    def init_from_loop(self, loop, block_name=''):
        self.init_from_blocks([StarfileBlock(loops=[loop], name=block_name)])

    def init_from_dataframe(self, dataframe, block_name=''):
        loop = dataframe
        self.init_from_loop(loop, block_name=block_name)

    def __repr__(self):
        return f'Starfile with {len(self.blocks)} blocks'

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.blocks[self.block_names.index(item)]
        else:
            return self.blocks[item]

    def __len__(self):
        return len(self.blocks)

    def __eq__(self, other):
        return all(b1 == b2 for b1, b2 in zip(self.blocks, other.blocks))

    def save(self, filename, overwrite=True):
        if not overwrite and os.path.exists(filename):
            raise RuntimeError(f'File {filename} already exists. Use overwrite=True to overwrite.')

        with open(filename, 'w') as f:
            for i, block in enumerate(self):
                f.write(f'data_{self.block_names[i]}\n\n')
                if block.properties is not None:
                    for k, v in block.properties.items():
                        f.write(f'{k} {v}\n')
                f.write('\n')
                for loop in block:
                    f.write('loop_\n')
                    for col in loop.columns:
                        f.write(f'{col}\n')
                    for _, row in loop.iterrows():
                        f.write(' '.join(map(str, row)) + '\n')
                    f.write('\n')
