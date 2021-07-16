import logging

import numpy as np

logger = logging.getLogger(__name__)


class CoefContainer:
    def __init__(self, data, mappings):

        self._data = data
        if self._data.ndim != 2:
            raise RuntimeError(
                f"Incorrect data array dimension {self._data.ndim}, expected 2"
            )

        self._n = data.shape[0]
        self._m = self._data.shape[1]

        self._mappings = mappings
        self._dim = len(mappings)
        for d, mapping in enumerate(mappings):
            if len(mapping) != self._m:
                raise RuntimeError(
                    f"Incorrect mappings shape {len(mapping)} for dimension {d}, expected {self._m}"
                )

    def __repr__(self):
        return f"{tuple(self).__name__}({self._data}, {self._mappings})"

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        logger.debug(f"idx {type(idx)} {idx}")

        if isinstance(idx, tuple):
            full_idx = [slice(None, None, None)] * (
                self._dim + 1
            )  # Mappings + Stack axes
            for i, _idx in enumerate(idx):
                full_idx[i] = _idx
            idx = tuple(full_idx)
        else:
            # make it a full tuple
            return self[
                idx,
            ]

        # Now that we have a consistent tuple
        #  we can use the same logic for all cases.

        # The first axis is always the "stack" axis
        stack_idx = idx[0]
        logger.debug(f"stack_idx {stack_idx}")

        # Remaining axes are container dimensions
        #   described by `mappings`
        # Find the data array indices represented by the container's slices.
        masks = np.zeros((self._dim, self._m), dtype=bool)
        rng = np.arange(self._m)
        for dim, _idx in enumerate(idx[1:]):
            # Evaluate the slice
            indices = rng[_idx]
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

        return self._data[stack_idx, data_idx]

    def __setitem__(self):
        pass

    def asnumpy(self):
        return self._data
