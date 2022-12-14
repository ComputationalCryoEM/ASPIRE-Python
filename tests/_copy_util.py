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


def img_accessors_deepcopied(i1, i2):
    # the image-getting function associated with the ImageAccessor
    # should have a different ID between different sources.
    # NOTE: This works because fun is a *function* not a *method*
    return i1.fun is not i2.fun


def ndarrays_deepcopied(ar1, ar2):
    # Source attributes which are just numpy arrays
    # should not have the same ID
    return ar1 is not ar2


def unique_filters_deepcopied(l1, l2):
    return all([l1[i] is not l2[i] for i in range(len(l1))])


# map ImageSource/subclass attributes to functions in this file that will
# ensure they are deep copied
_source_vars = {
    "_img_accessor": img_accessors_deepcopied,
    "_projections_accessor": img_accessors_deepcopied,
    "_clean_images_accessor": img_accessors_deepcopied,
    "_rotations": rotations_deepcopied,
    "generation_pipeline": xforms_deepcopied,
    "filter_indices": ndarrays_deepcopied,
    "unique_filters": unique_filters_deepcopied,
}

# test files can import the list of source vars to test against
source_vars = list(_source_vars.keys())


def source_vars_deepcopied(v1, v2, name):
    # if both are None, then no check necessary
    if v1 is None and v2 is None:
        return True
    fun = _source_vars[name]
    return fun(v1, v2)
