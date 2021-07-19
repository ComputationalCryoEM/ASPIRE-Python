import logging

import numpy as np

logger = logging.getLogger(__name__)


class CoefContainer:
    """
    This class interfaces between mathematical representations of sophisticated
    basis and the linear memory representations used in lower level code.

    For performance and portability reasons ASPIRE-Python stores alternative
    basis representations, such as the Fast Fourier Bessel Basis `FFBBasis2D`
    and `FFBBasis3d`, in stacks of flattened numpy arrays mapped with
    arrays of indexes described by the basis.

    The coef data indexing is theoretically defined using some number of
    axes, typically representing frequencies.  In a classic Fourier basis,
    these frequencies are dense and uniform (stride 1), so easily
    represented by a multi-dimensional array.  However, for most other basis
    in ASPIRE the frequency space is not required to be (and generally is not)
    dense or uniform.  Instead frequencies are defined in dense arrays of
    `indices`.  The ragged arrays of coefficients are flattened,
    and the `indices` provide a map to lookup locations of desired frequencies'
    values.

    While it would be possible to store coef data as high level
    objects indexed by arbitrary frequencies using dictionaries etc,
    this would yield poor perfomance over large stacks of images commonly
    found in Cryo-EM datasets.  Instead we provide an effecient container
    for naturally accessing the array storage.

    This class is intentionally generic to container mapping dimension
    and agnostic to data types.

    `data` and `mappings` should both be two dimensional arrays.
    The stack index is trivial, and  is the first (slow) moving axis of `data`.
    The container's dimension is taken from the first (slow) axis of `mappings`.
    The second axis of mappings describes the indice mapping for each basis
    axis.  For example, consider the following sequence having two basis
    dimensions `k` and `q`, writing coefs as C_k_q:

    {C_0_0, C_0_1, C_0_2, C_1_0, C_1_1, ...} corresponds to

    `k` indices {0, 0, 0, 1, 1, ...}
    `q` indices {0, 1, 2, 0, 1, ...}

    So we would provide mapping array:
    [[0, 0, 0, 1, 1, ...],
    [0, 1, 2, 0, 1, ...]]

    In practice indices are computed and accessible from Basis classes,
    and just need to be provided to CoefContainer (or a subclass).
    """

    def __init__(self, data, mappings):
        """
        Instantiate a CoefContainer described by `mappings`
        and backed by `data`.

        :param data: Numpy array of coef data.
        :param mappings: Iterable (list,tuple,array) of mappings, see class doc.
        """

        if not isinstance(data, np.ndarray):
            logger.warning(
                f"`data` is type {type(data)} not numpy array, attmepting conversion."
            )
            try:
                data = np.array(data)
            except Exception as e:
                logger.error("Conversion of `data` to numpy array failed!")
                raise e

        self._data = data
        if self._data.ndim != 2:
            raise RuntimeError(
                f"Incorrect data array dimension {self._data.ndim}, expected 2"
            )

        self._n = data.shape[0]
        self._m = self._data.shape[1]

        # Todo, discuess mappings optionally admitting a dictionary, naming the axes...
        try:
            self._mappings = np.array(mappings)
        except Exception as e:
            logger.error(
                f"Conversion of `mappings` {type(mappings)} to numpy array failed!"
            )
            raise e

        self.dim = self._mappings.shape[0]
        for d, mapping in enumerate(mappings):
            if len(mapping) != self._m:
                raise RuntimeError(
                    f"Incorrect mappings shape {mapping.shape[0]} for dimension {d}, expected {self._m}"
                )

        # # This is potentially used many times, compute it once.
        # self._rng = [np.arange(d[0], d[1]+1) for d in self.bounds)]

    def __repr__(self):
        return f"{type(self)}({self._data}, {self._mappings})"

    def __str__(self):
        return f"{type(self)} {self._n} entries of {self._m} coefficients mapped as {self.dim} dimensions."

    def __len__(self):
        return self._n

    def _expand_tuple(self, idx):
        """
        Expands any missing slice operators in idx.

        :param idx: Index operator, typically from `__getitem__` or `__setitem__`.
        :returns: self.dim dimensional tuple of slices and/or ints.
        """

        # Convert to tuple so we can always use same logic
        if not isinstance(idx, tuple):
            idx = (idx,)

        # Completely describe every slice
        # When high dims are missing, that implies :, ie, default slice of Nones
        full_idx = [slice(None, None, None)] * (self.dim + 1)  # Mappings + Stack axes

        for i, _idx in enumerate(idx):
            full_idx[i] = _idx

        return tuple(full_idx)

    def compute_idx(self, idx):
        """
        Given a one to `self.dim`-dimensional index (slices and/or ints),
        computes the indices into the stack of flattened arrays `data`.

        :param idx: Index operator, typically from `__getitem__` or `__setitem__`.
        :returns: tuple (stack_idx, data_idx) indexing `data`.
        """

        # Make a consistent tuple so we can use the same logic for all cases.
        idx = self._expand_tuple(idx)

        # The first axis is always the "stack" axis and has traditional slicing.
        stack_idx = idx[0]
        logger.debug(f"stack_idx {stack_idx}")

        # Remaining axes are container dimensions
        #   described by `mappings`
        # Find the data array indices represented by the container's slices.
        masks = np.zeros((self.dim, self._m), dtype=bool)

        for dim, _idx in enumerate(idx[1:]):
            # Evaluate the slice, note we convert to ranges instead of slices to accomodate negative subscripts.
            if isinstance(_idx, slice):
                # Consider bounds checking
                indices = np.arange(
                    _idx.start or self.bounds[dim][0],
                    _idx.stop or self.bounds[dim][1] + 1,  # inclusive
                    _idx.step or 1,
                )
            else:
                indices = _idx
            logger.debug(f"dim {dim} indices {indices}")

            # Union all locations we find indices
            logger.debug(f"_idx {_idx}  self._mappings[dim]{self._mappings[dim]}")
            masks[dim] = np.atleast_2d(np.isin(self._mappings[dim], indices)).any(
                axis=0
            )
            logger.debug(f"masks[dim] {masks[dim]}")

        # Now find the intersection of the masks for each container axis.
        #  This represents the indices into each entry of the stack.
        data_idx = masks.all(axis=0)
        logger.debug(f"data_idx {data_idx}")

        return stack_idx, data_idx

    def __getitem__(self, idx):
        logger.debug(f"get idx {type(idx)} {idx}")
        return self._data[self.compute_idx(idx)]

    def __setitem__(self, idx, values):
        logger.debug(f"set idx {type(idx)} {idx}")
        self._data[self.compute_idx(idx)] = values

    def asnumpy(self):
        return self._data

    def copy(self):
        """
        Copy, copies `data` and `mappings` arrays backing instance.

        :returns: new container instance.
        """
        return type(self)(self._data.copy(), self._mappings.copy())

    @property
    def shape(self):
        """
        Return shape of mapping, where each dimension in shape
        corresponds to count of unique indices in that dimension.

        :returns: tuple
        """
        return tuple([len(np.unique(axis)) for axis in self._mappings])

    @property
    def bounds(self):
        """
        Return the upper and lower bounds for each dimension
        of the mapping indices.

        :returns: list of tuples
        """

        return [(np.min(axis), np.max(axis)) for axis in self._mappings]
