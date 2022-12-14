from aspire.image.xform import (
    FilterXform,
    IndexedXform,
    Multiply,
    NoiseAdder,
    Pipeline,
    Shift,
)


def rotations_deepcopied(rots1, rots2):
    # core NumPy rotation matrices should be deep copies
    return rots1._matrices is not rots2._matrices


def xforms_deepcopied(xf1, xf2):
    # Xform attributes are heterogeneous, catch special cases
    # that need to be checked

    # FilterXform
    if all(map(lambda x: isinstance(x, FilterXform), (xf1, xf2))):
        return xf1.filter is not xf2.filter
    # NoiseAdder
    if all(map(lambda x: isinstance(x, NoiseAdder), (xf1, xf2))):
        return xf1.noise_filter is not xf2.noise_filter
    # Shift
    if all(map(lambda x: isinstance(x, Shift), (xf1, xf2))):
        return xf1.shifts is not xf2.shifts
    # Multiply
    if all(map(lambda x: isinstance(x, Multiply), (xf1, xf2))):
        return xf1.multipliers is not xf2.multipliers
    # Pipeline object
    if all(map(lambda x: isinstance(x, Pipeline), (xf1, xf2))):
        return all(
            [
                xforms_deepcopied(xf1.xforms[i], xf2.xforms[i])
                for i in range(len(xf1.xforms))
            ]
        )
    # IndexedXform object
    if all(map(lambda x: isinstance(x, IndexedXform), (xf1, xf2))):
        # 1d numpy array should be deep copied
        indices_check = xf1.indices is not xf2.indices
        return indices_check and all(
            [
                xforms_deepcopied(xf1.unique_xforms[i], xf2.unique_xforms[i])
                for i in range(len(xf1.unique_xforms))
            ]
        )

    # All other Xform objects we assume are deepcopied
    return True
