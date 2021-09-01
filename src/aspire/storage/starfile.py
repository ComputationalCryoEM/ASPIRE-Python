import logging
import starfile

logger = logging.getLogger(__name__)

class StarFile:
    # StarFile class currently exists to contain the methods below, which wrap the read() and write() 
    # methods of the starfile package

    @staticmethod
    def read(filepath, n_blocks=None):
        # for consistency's sake, we enable the always_dict flag 
        # this overrides the default behavior of the method,
        # which returns a pandas dataframe when the STAR file has one block,
        # but an OrderedDict of dataframes, when the file has multiple blocks.
        # With the flag enabled, an OrderedDict is always returned 

        # the OrderedDict is indexed by the name of data block (string)
        # or, in the case of a single block, by the empty string ''
        star = starfile.read(filepath, read_n_blocks=n_blocks, always_dict=True)
        return star

    @staticmethod
    def write(df, filename, float_format='%.6f', sep='\t', na_rep='<NA>', overwrite=True):
        # wraps the write() method exactly, except that we enable the overwrite flag by default
        # to mimic the behavior of the previous StarFile class
        starfile.write(df, filename, float_format=float_format, sep=sep, na_rep=na_rep, overwrite=overwrite)
