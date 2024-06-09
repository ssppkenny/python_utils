import numpy as np 
import math
cimport numpy as np
from libc.math cimport ceil
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython cimport PyObject

np.import_array()

cdef extern from *:
   ctypedef int Py_intptr_t

cdef enum:
    BACKGROUND = 0
    FOREGROUND = 1

cdef extern from "numpy/arrayobject.h" nogil:
    ctypedef struct PyArrayIterObject:
        np.npy_intp *coordinates

    void PyArray_ITER_NEXT(PyArrayIterObject *it)
    int PyArray_ITER_NOTDONE(PyArrayIterObject *it)
    void PyArray_ITER_RESET(PyArrayIterObject *it)
    void *PyArray_ITER_DATA(PyArrayIterObject *it)

    void *PyDataMem_NEW(size_t)
    void PyDataMem_FREE(void *)
    void *PyDataMem_RENEW(void *, size_t)

class NeedMoreBits(Exception):
    pass

ctypedef fused data_t:
    np.int8_t
    np.int16_t
    np.int32_t
    np.int64_t
    np.uint8_t
    np.uint16_t
    np.uint32_t
    np.uint64_t
    np.float32_t
    np.float64_t

######################################################################
# Load a line from a fused data array, setting the line to FOREGROUND wherever
# the fused data is nonzero, BACKGROUND elsewhere
######################################################################
cdef void fused_nonzero_line(data_t *p, np.intp_t stride,
                             np.uintp_t *line, np.intp_t L) noexcept nogil:
    cdef np.intp_t i
    for i in range(L):
        line[i] = FOREGROUND if \
            (<data_t *> ((<char *> p) + i * stride))[0] \
            else BACKGROUND


######################################################################
# Load a line from a fused data array to a np.uintp_t array
######################################################################
cdef void fused_read_line(data_t *p, np.intp_t stride,
                          np.uintp_t *line, np.intp_t L) noexcept nogil:
    cdef np.intp_t i
    for i in range(L):
        line[i] = <np.uintp_t> (<data_t *> ((<char *> p) + i * stride))[0]


######################################################################
# Store a line from a np.uintp_t array to a fused data array if possible,
# returning True if overflowed
######################################################################
cdef bint fused_write_line(data_t *p, np.intp_t stride,
                           np.uintp_t *line, np.intp_t L) noexcept nogil:
    cdef np.intp_t i
    for i in range(L):
        # Check before overwrite, as this prevents us accidentally writing a 0
        # in the foreground, which allows us to retry even when operating
        # in-place.
        if line[i] != <np.uintp_t> <data_t> line[i]:
            return True
        (<data_t *> ((<char *> p) + i * stride))[0] = <data_t> line[i]
    return False


ctypedef void (*nonzero_line_func_t)(void *p, np.intp_t stride,
                                     np.uintp_t *line, np.intp_t L) noexcept nogil
ctypedef void (*read_line_func_t)(void *p, np.intp_t stride,
                                  np.uintp_t *line, np.intp_t L) noexcept nogil
ctypedef bint (*write_line_func_t)(void *p, np.intp_t stride,
                                   np.uintp_t *line, np.intp_t L) noexcept nogil

def get_nonzero_line(np.ndarray[data_t] a):
    return <Py_intptr_t> fused_nonzero_line[data_t]

def get_read_line(np.ndarray[data_t] a):
    return <Py_intptr_t> fused_read_line[data_t]

def get_write_line(np.ndarray[data_t] a):
    return <Py_intptr_t> fused_write_line[data_t]

cdef extern from "wrapper.h":
    ctypedef struct pdf_size:
        int width
        int height
    cdef char* get_pdf_page(int pagenumber, char* filepath)
    pdf_size get_pdf_page_size(int pagenumber, char* filepath)


def get_page_size(pagenumber, filepath):
    bpath = bytes(filepath, 'utf-8')
    size = get_pdf_page_size(pagenumber, bpath)
    return size.width, size.height


cpdef get_pdf_page_bytes(int pagenumber, char* filepath):
    size = get_pdf_page_size(pagenumber, filepath)
    cdef char* bytes = get_pdf_page(pagenumber, filepath)
    return PyBytes_FromStringAndSize(bytes, 4*size.width*size.height)


def get_page(pagenumber, filepath):
    bpath = bytes(filepath, 'utf-8')
    return get_pdf_page_bytes(pagenumber, bpath)


######################################################################
# Mark two labels to be merged
######################################################################
cdef inline np.uintp_t mark_for_merge(np.uintp_t a,
                                      np.uintp_t b,
                                      np.uintp_t *mergetable) noexcept nogil:

    cdef:
        np.uintp_t orig_a, orig_b, minlabel

    orig_a = a
    orig_b = b
    # find smallest root for each of a and b
    while a != mergetable[a]:
        a = mergetable[a]
    while b != mergetable[b]:
        b = mergetable[b]
    minlabel = a if (a < b) else b

    # merge roots
    mergetable[a] = mergetable[b] = minlabel

    # merge every step to minlabel
    a = orig_a
    b = orig_b
    while a != minlabel:
        a, mergetable[a] = mergetable[a], minlabel
    while b != minlabel:
        b, mergetable[b] = mergetable[b], minlabel

    return minlabel


######################################################################
# Take the label of a neighbor, or mark them for merging
######################################################################
cdef inline np.uintp_t take_label_or_merge(np.uintp_t cur_label,
                                           np.uintp_t neighbor_label,
                                           np.uintp_t *mergetable) noexcept nogil:
    if neighbor_label == BACKGROUND:
        return cur_label
    if cur_label == FOREGROUND:
        return neighbor_label  # neighbor is not BACKGROUND
    if neighbor_label:
        if cur_label != neighbor_label:
            cur_label = mark_for_merge(neighbor_label, cur_label, mergetable)
    return cur_label


######################################################################
# Label one line of input, using a neighbor line that has already been labeled.
######################################################################
cdef np.uintp_t label_line_with_neighbor(np.uintp_t *line,
                                         np.uintp_t *neighbor,
                                         int neighbor_use_previous,
                                         int neighbor_use_adjacent,
                                         int neighbor_use_next,
                                         np.intp_t L,
                                         bint label_unlabeled,
                                         bint use_previous,
                                         np.uintp_t next_region,
                                         np.uintp_t *mergetable) noexcept nogil:
    cdef:
        np.intp_t i

    for i in range(L):
        if line[i] != BACKGROUND:
            # See allocation of line_buffer for why this is valid when i = 0
            if neighbor_use_previous:
                line[i] = take_label_or_merge(line[i], neighbor[i - 1], mergetable)
            if neighbor_use_adjacent:
                line[i] = take_label_or_merge(line[i], neighbor[i], mergetable)
            if neighbor_use_next:
                line[i] = take_label_or_merge(line[i], neighbor[i + 1], mergetable)
            if label_unlabeled:
                if use_previous:
                    line[i] = take_label_or_merge(line[i], line[i - 1], mergetable)
                if line[i] == FOREGROUND:  # still needs a label
                    line[i] = next_region
                    mergetable[next_region] = next_region
                    next_region += 1
    return next_region


cpdef _label(np.ndarray input,
             np.ndarray structure,
             np.ndarray output):
    # check dimensions
    # To understand the need for the casts to object in order to use
    # tuple.__eq__, see https://github.com/cython/cython/issues/863
    assert (<object> input).shape == (<object> output).shape, \
        ("Shapes must match for input and output,"
         "{} != {}".format((<object> input).shape, (<object> output).shape))

    structure = np.asanyarray(structure, dtype=np.bool_).copy()
    assert input.ndim == structure.ndim, \
        ("Structuring element must have same "
         "# of dimensions as input, "
         "{:d} != {:d}".format(input.ndim, structure.ndim))

    # Check that structuring element is of size 3 in every dimension
    assert set((<object> structure).shape) <= set([3]), \
        ("Structuring element must be size 3 in every dimension, "
         "was {}".format((<object> structure).shape))

    # check structuring element for symmetry
    assert np.all(structure == structure[(np.s_[::-1],) * structure.ndim]), \
        "Structuring element is not symmetric"

    # make sure we're dealing with a non-empty, non-scalar array
    assert input.ndim > 0 and input.size > 0, "Cannot label scalars or empty arrays"

    # if we're handed booleans, we treat them as uint8s
    if input.dtype == np.bool_:
        input = input.view(dtype=np.uint8)
    if output.dtype == np.bool_:
        # XXX - trigger special check for bit depth?
        output = output.view(dtype=np.uint8)

    cdef:
        nonzero_line_func_t nonzero_line = \
            <nonzero_line_func_t> <void *> <Py_intptr_t> get_nonzero_line(input.take([0]))
        read_line_func_t read_line = \
            <read_line_func_t> <void *> <Py_intptr_t> get_read_line(output.take([0]))
        write_line_func_t write_line = \
            <write_line_func_t> <void *> <Py_intptr_t> get_write_line(output.take([0]))
        np.flatiter _iti, _ito, _itstruct
        PyArrayIterObject *iti
        PyArrayIterObject *ito
        PyArrayIterObject *itstruct
        int axis, idim, num_neighbors, ni
        np.intp_t L, delta, i
        np.intp_t si, so, ss
        np.intp_t total_offset
        np.intp_t output_ndim, structure_ndim
        bint needs_self_labeling, valid, use_previous, overflowed
        np.ndarray _line_buffer, _neighbor_buffer
        np.uintp_t *line_buffer
        np.uintp_t *neighbor_buffer
        np.uintp_t *tmp
        np.uintp_t next_region, src_label, dest_label
        np.uintp_t mergetable_size
        np.uintp_t *mergetable

    axis = -1  # choose best axis based on output
    _ito = np.PyArray_IterAllButAxis(output, &axis)
    _iti = np.PyArray_IterAllButAxis(input, &axis)
    _itstruct = np.PyArray_IterAllButAxis(structure, &axis)

    ito = <PyArrayIterObject *> _ito
    iti = <PyArrayIterObject *> _iti
    itstruct = <PyArrayIterObject *> _itstruct

    # we only process this many neighbors from the itstruct iterator before
    # reaching the center, where we stop
    num_neighbors = structure.size // (3 * 2)

    # Create two buffer arrays for reading/writing labels.
    # Add an entry at the end and beginning to simplify some bounds checks.
    L = input.shape[axis]
    _line_buffer = np.empty(L + 2, dtype=np.uintp)
    _neighbor_buffer = np.empty(L + 2, dtype=np.uintp)
    line_buffer = <np.uintp_t *> _line_buffer.data
    neighbor_buffer = <np.uintp_t *> _neighbor_buffer.data

    # Add fenceposts with background values
    line_buffer[0] = neighbor_buffer[0] = BACKGROUND
    line_buffer[L + 1] = neighbor_buffer[L + 1] = BACKGROUND
    line_buffer = line_buffer + 1
    neighbor_buffer = neighbor_buffer + 1

    mergetable_size = 2 * output.shape[axis]
    mergetable = <np.uintp_t *> PyDataMem_NEW(mergetable_size * sizeof(np.uintp_t))
    if mergetable == NULL:
        raise MemoryError()

    try:
        # strides
        si = input.strides[axis]
        so = output.strides[axis]
        ss = structure.strides[axis]

        # 0 = background
        # 1 = foreground, needs label
        # 2... = working labels, will be compacted on output
        next_region = 2

        structure_ndim = structure.ndim
        temp = [1] * structure_ndim
        temp[axis] = 0
        use_previous = (structure[tuple(temp)] != 0)
        output_ndim = output.ndim
        output_shape = output.shape
        output_strides = output.strides

        with nogil:
            while PyArray_ITER_NOTDONE(iti):
                # Optimization - for 2D, line_buffer becomes next iteration's
                # neighbor buffer
                if output_ndim == 2:
                    tmp = line_buffer
                    line_buffer = neighbor_buffer
                    neighbor_buffer = tmp

                # copy nonzero values in input to line_buffer as FOREGROUND
                nonzero_line(PyArray_ITER_DATA(iti), si, line_buffer, L)

                # Used for labeling single lines
                needs_self_labeling = True

                # Take neighbor labels
                PyArray_ITER_RESET(itstruct)
                for ni in range(num_neighbors):
                    neighbor_use_prev = (<np.npy_bool *> PyArray_ITER_DATA(itstruct))[0]
                    neighbor_use_adjacent = (<np.npy_bool *> (<char *> PyArray_ITER_DATA(itstruct) + ss))[0]
                    neighbor_use_next = (<np.npy_bool *> (<char *> PyArray_ITER_DATA(itstruct) + 2 * ss))[0]
                    if not (neighbor_use_prev or
                            neighbor_use_adjacent or
                            neighbor_use_next):
                        PyArray_ITER_NEXT(itstruct)
                        continue

                    # Check that the neighbor line is in bounds
                    valid = True
                    total_offset = 0
                    for idim in range(structure_ndim):
                        if idim == axis:
                            continue
                        delta = (itstruct.coordinates[idim] - 1)  # 1,1,1... is center
                        if not (0 <= (ito.coordinates[idim] + delta) < output_shape[idim]):
                            valid = False
                            break
                        total_offset += delta * output_strides[idim]

                    if valid:
                        # Optimization (see above) - for 2D, line_buffer
                        # becomes next iteration's neighbor buffer, so no
                        # need to read it here.
                        if output_ndim != 2:
                            read_line(<char *> PyArray_ITER_DATA(ito) + total_offset, so,
                                      neighbor_buffer, L)

                        # be conservative about how much space we may need
                        while mergetable_size < (next_region + L):
                            mergetable_size *= 2
                            mergetable = <np.uintp_t *> \
                                PyDataMem_RENEW(<void *> mergetable,
                                                 mergetable_size * sizeof(np.uintp_t))

                        next_region = label_line_with_neighbor(line_buffer,
                                                              neighbor_buffer,
                                                              neighbor_use_prev,
                                                              neighbor_use_adjacent,
                                                              neighbor_use_next,
                                                              L,
                                                              ni == (num_neighbors - 1),
                                                              use_previous,
                                                              next_region,
                                                              mergetable)
                        if ni == (num_neighbors - 1):
                            needs_self_labeling = False
                    PyArray_ITER_NEXT(itstruct)

                if needs_self_labeling:
                    # We didn't call label_line_with_neighbor above with
                    # label_unlabeled=True, so call it now in such a way as to
                    # cause unlabeled regions to get a label.
                    while mergetable_size < (next_region + L):
                            mergetable_size *= 2
                            mergetable = <np.uintp_t *> \
                                PyDataMem_RENEW(<void *> mergetable,
                                                 mergetable_size * sizeof(np.uintp_t))

                    next_region = label_line_with_neighbor(line_buffer,
                                                          neighbor_buffer,
                                                          False, False, False,  # no neighbors
                                                          L,
                                                          True,
                                                          use_previous,
                                                          next_region,
                                                          mergetable)

                overflowed = write_line(PyArray_ITER_DATA(ito), so,
                                        line_buffer, L)
                if overflowed:
                    with gil:
                        raise NeedMoreBits()

                PyArray_ITER_NEXT(iti)
                PyArray_ITER_NEXT(ito)

            # compact the mergetable, mapping each value to its destination
            mergetable[BACKGROUND] = BACKGROUND
            mergetable[FOREGROUND] = -1  # should never be encountered
            mergetable[2] = 1  # labels started here
            dest_label = 2
            # next_region is still the original value -> we found no regions
            # set dest_label to 1 so we return 0
            if next_region < 3:
                dest_label = 1
            for src_label in range(3, next_region):
                # labels that map to themselves are new regions
                if mergetable[src_label] == src_label:
                    mergetable[src_label] = dest_label
                    dest_label += 1
                else:
                    # we've compacted every label below this, and the
                    # mergetable has an invariant (from mark_for_merge()) that
                    # it always points downward.  Therefore, we can fetch the
                    # final label by two steps of indirection.
                    mergetable[src_label] = mergetable[mergetable[src_label]]

            PyArray_ITER_RESET(ito)
            while PyArray_ITER_NOTDONE(ito):
                read_line(PyArray_ITER_DATA(ito), so, line_buffer, L)
                for i in range(L):
                    line_buffer[i] = mergetable[line_buffer[i]]
                write_line(PyArray_ITER_DATA(ito), so, line_buffer, L)
                PyArray_ITER_NEXT(ito)
    except:  # noqa: E722
        # clean up and re-raise
        PyDataMem_FREE(<void *> mergetable)
        raise

    PyDataMem_FREE(<void *> mergetable)
    return dest_label - 1


def generate_binary_structure(rank, connectivity):
    """
    Generate a binary structure for binary morphological operations.

    Parameters
    ----------
    rank : int
         Number of dimensions of the array to which the structuring element
         will be applied, as returned by `np.ndim`.
    connectivity : int
         `connectivity` determines which elements of the output array belong
         to the structure, i.e., are considered as neighbors of the central
         element. Elements up to a squared distance of `connectivity` from
         the center are considered neighbors. `connectivity` may range from 1
         (no diagonal elements are neighbors) to `rank` (all elements are
         neighbors).

    Returns
    -------
    output : ndarray of bools
         Structuring element which may be used for binary morphological
         operations, with `rank` dimensions and all dimensions equal to 3.

    See Also
    --------
    iterate_structure, binary_dilation, binary_erosion

    Notes
    -----
    `generate_binary_structure` can only create structuring elements with
    dimensions equal to 3, i.e., minimal dimensions. For larger structuring
    elements, that are useful e.g., for eroding large objects, one may either
    use `iterate_structure`, or create directly custom arrays with
    numpy functions such as `numpy.ones`.

    Examples
    --------
    >>> from scipy import ndimage
    >>> import numpy as np
    >>> struct = ndimage.generate_binary_structure(2, 1)
    >>> struct
    array([[False,  True, False],
           [ True,  True,  True],
           [False,  True, False]], dtype=bool)
    >>> a = np.zeros((5,5))
    >>> a[2, 2] = 1
    >>> a
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> b = ndimage.binary_dilation(a, structure=struct).astype(a.dtype)
    >>> b
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> ndimage.binary_dilation(b, structure=struct).astype(a.dtype)
    array([[ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  0.,  1.,  0.,  0.]])
    >>> struct = ndimage.generate_binary_structure(2, 2)
    >>> struct
    array([[ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True]], dtype=bool)
    >>> struct = ndimage.generate_binary_structure(3, 1)
    >>> struct # no diagonal elements
    array([[[False, False, False],
            [False,  True, False],
            [False, False, False]],
           [[False,  True, False],
            [ True,  True,  True],
            [False,  True, False]],
           [[False, False, False],
            [False,  True, False],
            [False, False, False]]], dtype=bool)

    """
    if connectivity < 1:
        connectivity = 1
    if rank < 1:
        return np.array(True, dtype=bool)
    output = np.fabs(np.indices([3] * rank) - 1)
    output = np.add.reduce(output, 0)
    return output <= connectivity


def label(input, structure=None, output=None):
    """
    Label features in an array.

    Parameters
    ----------
    input : array_like
        An array-like object to be labeled. Any non-zero values in `input` are
        counted as features and zero values are considered the background.
    structure : array_like, optional
        A structuring element that defines feature connections.
        `structure` must be centrosymmetric
        (see Notes).
        If no structuring element is provided,
        one is automatically generated with a squared connectivity equal to
        one.  That is, for a 2-D `input` array, the default structuring element
        is::

            [[0,1,0],
             [1,1,1],
             [0,1,0]]

    output : (None, data-type, array_like), optional
        If `output` is a data type, it specifies the type of the resulting
        labeled feature array.
        If `output` is an array-like object, then `output` will be updated
        with the labeled features from this function.  This function can
        operate in-place, by passing output=input.
        Note that the output must be able to store the largest label, or this
        function will raise an Exception.

    Returns
    -------
    label : ndarray or int
        An integer ndarray where each unique feature in `input` has a unique
        label in the returned array.
    num_features : int
        How many objects were found.

        If `output` is None, this function returns a tuple of
        (`labeled_array`, `num_features`).

        If `output` is a ndarray, then it will be updated with values in
        `labeled_array` and only `num_features` will be returned by this
        function.

    See Also
    --------
    find_objects : generate a list of slices for the labeled features (or
                   objects); useful for finding features' position or
                   dimensions

    Notes
    -----
    A centrosymmetric matrix is a matrix that is symmetric about the center.
    See [1]_ for more information.

    The `structure` matrix must be centrosymmetric to ensure
    two-way connections.
    For instance, if the `structure` matrix is not centrosymmetric
    and is defined as::

        [[0,1,0],
         [1,1,0],
         [0,0,0]]

    and the `input` is::

        [[1,2],
         [0,3]]

    then the structure matrix would indicate the
    entry 2 in the input is connected to 1,
    but 1 is not connected to 2.

    References
    ----------
    .. [1] James R. Weaver, "Centrosymmetric (cross-symmetric)
       matrices, their basic properties, eigenvalues, and
       eigenvectors." The American Mathematical Monthly 92.10
       (1985): 711-717.

    Examples
    --------
    Create an image with some features, then label it using the default
    (cross-shaped) structuring element:

    >>> from scipy.ndimage import label, generate_binary_structure
    >>> import numpy as np
    >>> a = np.array([[0,0,1,1,0,0],
    ...               [0,0,0,1,0,0],
    ...               [1,1,0,0,1,0],
    ...               [0,0,0,1,0,0]])
    >>> labeled_array, num_features = label(a)

    Each of the 4 features are labeled with a different integer:

    >>> num_features
    4
    >>> labeled_array
    array([[0, 0, 1, 1, 0, 0],
           [0, 0, 0, 1, 0, 0],
           [2, 2, 0, 0, 3, 0],
           [0, 0, 0, 4, 0, 0]])

    Generate a structuring element that will consider features connected even
    if they touch diagonally:

    >>> s = generate_binary_structure(2,2)

    or,

    >>> s = [[1,1,1],
    ...      [1,1,1],
    ...      [1,1,1]]

    Label the image using the new structuring element:

    >>> labeled_array, num_features = label(a, structure=s)

    Show the 2 labeled features (note that features 1, 3, and 4 from above are
    now considered a single feature):

    >>> num_features
    2
    >>> labeled_array
    array([[0, 0, 1, 1, 0, 0],
           [0, 0, 0, 1, 0, 0],
           [2, 2, 0, 0, 1, 0],
           [0, 0, 0, 1, 0, 0]])

    """
    input = np.asarray(input)
    if np.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    if structure is None:
        structure = generate_binary_structure(input.ndim, 1)
    structure = np.asarray(structure, dtype=bool)
    if structure.ndim != input.ndim:
        raise RuntimeError('structure and input must have equal rank')
    for ii in structure.shape:
        if ii != 3:
            raise ValueError('structure dimensions must be equal to 3')

    # Use 32 bits if it's large enough for this image.
    # _ni_label.label() needs two entries for background and
    # foreground tracking
    need_64bits = input.size >= (2**31 - 2)

    if isinstance(output, np.ndarray):
        if output.shape != input.shape:
            raise ValueError("output shape not correct")
        caller_provided_output = True
    else:
        caller_provided_output = False
        if output is None:
            output = np.empty(input.shape, np.intp if need_64bits else np.int32)
        else:
            output = np.empty(input.shape, output)

    # handle scalars, 0-D arrays
    if input.ndim == 0 or input.size == 0:
        if input.ndim == 0:
            # scalar
            maxlabel = 1 if (input != 0) else 0
            output[...] = maxlabel
        else:
            # 0-D
            maxlabel = 0
        if caller_provided_output:
            return maxlabel
        else:
            return output, maxlabel

    try:
        max_label = _label(input, structure, output)
    except NeedMoreBits as e:
        # Make another attempt with enough bits, then try to cast to the
        # new type.
        tmp_output = np.empty(input.shape, np.intp if need_64bits else np.int32)
        max_label = _label(input, structure, tmp_output)
        output[...] = tmp_output[...]
        if not np.all(output == tmp_output):
            # refuse to return bad results
            raise RuntimeError(
                "insufficient bit-depth in requested output type"
            ) from e

    if caller_provided_output:
        # result was written in-place
        return max_label
    else:
        return output, max_label


def _peak_prominences(const np.float64_t[::1] x not None,
                      np.intp_t[::1] peaks not None,
                      np.intp_t wlen):
    """
    Calculate the prominence of each peak in a signal.

    Parameters
    ----------
    x : ndarray
        A signal with peaks.
    peaks : ndarray
        Indices of peaks in `x`.
    wlen : np.intp
        A window length in samples (see `peak_prominences`) which is rounded up
        to the nearest odd integer. If smaller than 2 the entire signal `x` is
        used.

    Returns
    -------
    prominences : ndarray
        The calculated prominences for each peak in `peaks`.
    left_bases, right_bases : ndarray
        The peaks' bases as indices in `x` to the left and right of each peak.

    Raises
    ------
    ValueError
        If a value in `peaks` is an invalid index for `x`.

    Warns
    -----
    PeakPropertyWarning
        If a prominence of 0 was calculated for any peak.

    Notes
    -----
    This is the inner function to `peak_prominences`.

    .. versionadded:: 1.1.0
    """
    cdef:
        np.float64_t[::1] prominences
        np.intp_t[::1] left_bases, right_bases
        np.float64_t left_min, right_min
        np.intp_t peak_nr, peak, i_min, i_max, i
        np.uint8_t show_warning

    show_warning = False
    prominences = np.empty(peaks.shape[0], dtype=np.float64)
    left_bases = np.empty(peaks.shape[0], dtype=np.intp)
    right_bases = np.empty(peaks.shape[0], dtype=np.intp)

    with nogil:
        for peak_nr in range(peaks.shape[0]):
            peak = peaks[peak_nr]
            i_min = 0
            i_max = x.shape[0] - 1
            if not i_min <= peak <= i_max:
                with gil:
                    raise ValueError("peak {} is not a valid index for `x`"
                                     .format(peak))

            if 2 <= wlen:
                # Adjust window around the evaluated peak (within bounds);
                # if wlen is even the resulting window length is implicitly
                # rounded to next odd integer
                i_min = max(peak - wlen // 2, i_min)
                i_max = min(peak + wlen // 2, i_max)

            # Find the left base in interval [i_min, peak]
            i = left_bases[peak_nr] = peak
            left_min = x[peak]
            while i_min <= i and x[i] <= x[peak]:
                if x[i] < left_min:
                    left_min = x[i]
                    left_bases[peak_nr] = i
                i -= 1

            # Find the right base in interval [peak, i_max]
            i = right_bases[peak_nr] = peak
            right_min = x[peak]
            while i <= i_max and x[i] <= x[peak]:
                if x[i] < right_min:
                    right_min = x[i]
                    right_bases[peak_nr] = i
                i += 1

            prominences[peak_nr] = x[peak] - max(left_min, right_min)
            if prominences[peak_nr] == 0:
                show_warning = True

    # Return memoryviews as ndarrays
    return prominences.base, left_bases.base, right_bases.base

def _arg_peaks_as_expected(value):
    """Ensure argument `peaks` is a 1-D C-contiguous array of dtype('intp').

    Used in `peak_prominences` and `peak_widths` to make `peaks` compatible
    with the signature of the wrapped Cython functions.

    Returns
    -------
    value : ndarray
        A 1-D C-contiguous array with dtype('intp').
    """
    value = np.asarray(value)
    if value.size == 0:
        # Empty arrays default to np.float64 but are valid input
        value = np.array([], dtype=np.intp)
    try:
        # Safely convert to C-contiguous array of type np.intp
        value = value.astype(np.intp, order='C', casting='safe',
                             subok=False, copy=False)
    except TypeError as e:
        raise TypeError("cannot safely cast `peaks` to dtype('intp')") from e
    if value.ndim != 1:
        raise ValueError('`peaks` must be a 1-D array')
    return value

def _peak_widths(const np.float64_t[::1] x not None,
                 np.intp_t[::1] peaks not None,
                 np.float64_t rel_height,
                 np.float64_t[::1] prominences not None,
                 np.intp_t[::1] left_bases not None,
                 np.intp_t[::1] right_bases not None):
    """
    Calculate the width of each each peak in a signal.

    Parameters
    ----------
    x : ndarray
        A signal with peaks.
    peaks : ndarray
        Indices of peaks in `x`.
    rel_height : np.float64
        Chooses the relative height at which the peak width is measured as a
        percentage of its prominence (see `peak_widths`).
    prominences : ndarray
        Prominences of each peak in `peaks` as returned by `peak_prominences`.
    left_bases, right_bases : ndarray
        Left and right bases of each peak in `peaks` as returned by
        `peak_prominences`.

    Returns
    -------
    widths : ndarray
        The widths for each peak in samples.
    width_heights : ndarray
        The height of the contour lines at which the `widths` where evaluated.
    left_ips, right_ips : ndarray
        Interpolated positions of left and right intersection points of a
        horizontal line at the respective evaluation height.

    Raises
    ------
    ValueError
        If the supplied prominence data doesn't satisfy the condition
        ``0 <= left_base <= peak <= right_base < x.shape[0]`` for each peak or
        if `peaks`, `left_bases` and `right_bases` don't share the same shape.
        Or if `rel_height` is not at least 0.

    Warnings
    --------
    PeakPropertyWarning
        If a width of 0 was calculated for any peak.

    Notes
    -----
    This is the inner function to `peak_widths`.

    .. versionadded:: 1.1.0
    """
    cdef:
        np.float64_t[::1] widths, width_heights, left_ips, right_ips
        np.float64_t height, left_ip, right_ip
        np.intp_t p, peak, i, i_max, i_min
        np.uint8_t show_warning

    if rel_height < 0:
        raise ValueError('`rel_height` must be greater or equal to 0.0')
    if not (peaks.shape[0] == prominences.shape[0] == left_bases.shape[0]
            == right_bases.shape[0]):
        raise ValueError("arrays in `prominence_data` must have the same shape "
                         "as `peaks`")

    show_warning = False
    widths = np.empty(peaks.shape[0], dtype=np.float64)
    width_heights = np.empty(peaks.shape[0], dtype=np.float64)
    left_ips = np.empty(peaks.shape[0], dtype=np.float64)
    right_ips = np.empty(peaks.shape[0], dtype=np.float64)

    with nogil:
        for p in range(peaks.shape[0]):
            i_min = left_bases[p]
            i_max = right_bases[p]
            peak = peaks[p]
            # Validate bounds and order
            if not 0 <= i_min <= peak <= i_max < x.shape[0]:
                with gil:
                    raise ValueError("prominence data is invalid for peak {}"
                                     .format(peak))
            height = width_heights[p] = x[peak] - prominences[p] * rel_height

            # Find intersection point on left side
            i = peak
            while i_min < i and height < x[i]:
                i -= 1
            left_ip = <np.float64_t>i
            if x[i] < height:
                # Interpolate if true intersection height is between samples
                left_ip += (height - x[i]) / (x[i + 1] - x[i])

            # Find intersection point on right side
            i = peak
            while i < i_max and height < x[i]:
                i += 1
            right_ip = <np.float64_t>i
            if  x[i] < height:
                # Interpolate if true intersection height is between samples
                right_ip -= (height - x[i]) / (x[i - 1] - x[i])

            widths[p] = right_ip - left_ip
            if widths[p] == 0:
                show_warning = True
            left_ips[p] = left_ip
            right_ips[p] = right_ip

    return widths.base, width_heights.base, left_ips.base, right_ips.base

def peak_prominences(x, peaks, wlen=None):
    """
    Calculate the prominence of each peak in a signal.

    The prominence of a peak measures how much a peak stands out from the
    surrounding baseline of the signal and is defined as the vertical distance
    between the peak and its lowest contour line.

    Parameters
    ----------
    x : sequence
        A signal with peaks.
    peaks : sequence
        Indices of peaks in `x`.
    wlen : int, optional
        A window length in samples that optionally limits the evaluated area for
        each peak to a subset of `x`. The peak is always placed in the middle of
        the window therefore the given length is rounded up to the next odd
        integer. This parameter can speed up the calculation (see Notes).

    Returns
    -------
    prominences : ndarray
        The calculated prominences for each peak in `peaks`.
    left_bases, right_bases : ndarray
        The peaks' bases as indices in `x` to the left and right of each peak.
        The higher base of each pair is a peak's lowest contour line.

    Raises
    ------
    ValueError
        If a value in `peaks` is an invalid index for `x`.

    Warns
    -----
    PeakPropertyWarning
        For indices in `peaks` that don't point to valid local maxima in `x`,
        the returned prominence will be 0 and this warning is raised. This
        also happens if `wlen` is smaller than the plateau size of a peak.

    Warnings
    --------
    This function may return unexpected results for data containing NaNs. To
    avoid this, NaNs should either be removed or replaced.

    See Also
    --------
    find_peaks
        Find peaks inside a signal based on peak properties.
    peak_widths
        Calculate the width of peaks.

    Notes
    -----
    Strategy to compute a peak's prominence:

    1. Extend a horizontal line from the current peak to the left and right
       until the line either reaches the window border (see `wlen`) or
       intersects the signal again at the slope of a higher peak. An
       intersection with a peak of the same height is ignored.
    2. On each side find the minimal signal value within the interval defined
       above. These points are the peak's bases.
    3. The higher one of the two bases marks the peak's lowest contour line. The
       prominence can then be calculated as the vertical difference between the
       peaks height itself and its lowest contour line.

    Searching for the peak's bases can be slow for large `x` with periodic
    behavior because large chunks or even the full signal need to be evaluated
    for the first algorithmic step. This evaluation area can be limited with the
    parameter `wlen` which restricts the algorithm to a window around the
    current peak and can shorten the calculation time if the window length is
    short in relation to `x`.
    However, this may stop the algorithm from finding the true global contour
    line if the peak's true bases are outside this window. Instead, a higher
    contour line is found within the restricted window leading to a smaller
    calculated prominence. In practice, this is only relevant for the highest set
    of peaks in `x`. This behavior may even be used intentionally to calculate
    "local" prominences.

    .. versionadded:: 1.1.0

    References
    ----------
    .. [1] Wikipedia Article for Topographic Prominence:
       https://en.wikipedia.org/wiki/Topographic_prominence

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal import find_peaks, peak_prominences
    >>> import matplotlib.pyplot as plt

    Create a test signal with two overlaid harmonics

    >>> x = np.linspace(0, 6 * np.pi, 1000)
    >>> x = np.sin(x) + 0.6 * np.sin(2.6 * x)

    Find all peaks and calculate prominences

    >>> peaks, _ = find_peaks(x)
    >>> prominences = peak_prominences(x, peaks)[0]
    >>> prominences
    array([1.24159486, 0.47840168, 0.28470524, 3.10716793, 0.284603  ,
           0.47822491, 2.48340261, 0.47822491])

    Calculate the height of each peak's contour line and plot the results

    >>> contour_heights = x[peaks] - prominences
    >>> plt.plot(x)
    >>> plt.plot(peaks, x[peaks], "x")
    >>> plt.vlines(x=peaks, ymin=contour_heights, ymax=x[peaks])
    >>> plt.show()

    Let's evaluate a second example that demonstrates several edge cases for
    one peak at index 5.

    >>> x = np.array([0, 1, 0, 3, 1, 3, 0, 4, 0])
    >>> peaks = np.array([5])
    >>> plt.plot(x)
    >>> plt.plot(peaks, x[peaks], "x")
    >>> plt.show()
    >>> peak_prominences(x, peaks)  # -> (prominences, left_bases, right_bases)
    (array([3.]), array([2]), array([6]))

    Note how the peak at index 3 of the same height is not considered as a
    border while searching for the left base. Instead, two minima at 0 and 2
    are found in which case the one closer to the evaluated peak is always
    chosen. On the right side, however, the base must be placed at 6 because the
    higher peak represents the right border to the evaluated area.

    >>> peak_prominences(x, peaks, wlen=3.1)
    (array([2.]), array([4]), array([6]))

    Here, we restricted the algorithm to a window from 3 to 7 (the length is 5
    samples because `wlen` was rounded up to the next odd integer). Thus, the
    only two candidates in the evaluated area are the two neighboring samples
    and a smaller prominence is calculated.
    """
    x = _arg_x_as_expected(x)
    peaks = _arg_peaks_as_expected(peaks)
    wlen = _arg_wlen_as_expected(wlen)
    return _peak_prominences(x, peaks, wlen)

def _arg_wlen_as_expected(value):
    """Ensure argument `wlen` is of type `np.intp` and larger than 1.

    Used in `peak_prominences` and `peak_widths`.

    Returns
    -------
    value : np.intp
        The original `value` rounded up to an integer or -1 if `value` was
        None.
    """
    if value is None:
        # _peak_prominences expects an intp; -1 signals that no value was
        # supplied by the user
        value = -1
    elif 1 < value:
        # Round up to a positive integer
        if isinstance(value, float):
            value = math.ceil(value)
        value = np.intp(value)
    else:
        raise ValueError(f'`wlen` must be larger than 1, was {value}')
    return value

def _select_by_property(peak_properties, pmin, pmax):
    """
    Evaluate where the generic property of peaks confirms to an interval.

    Parameters
    ----------
    peak_properties : ndarray
        An array with properties for each peak.
    pmin : None or number or ndarray
        Lower interval boundary for `peak_properties`. ``None`` is interpreted as
        an open border.
    pmax : None or number or ndarray
        Upper interval boundary for `peak_properties`. ``None`` is interpreted as
        an open border.

    Returns
    -------
    keep : bool
        A boolean mask evaluating to true where `peak_properties` confirms to the
        interval.

    See Also
    --------
    find_peaks

    Notes
    -----

    .. versionadded:: 1.1.0
    """
    keep = np.ones(peak_properties.size, dtype=bool)
    if pmin is not None:
        keep &= (pmin <= peak_properties)
    if pmax is not None:
        keep &= (peak_properties <= pmax)
    return keep

def _select_by_peak_threshold(x, peaks, tmin, tmax):
    """
    Evaluate which peaks fulfill the threshold condition.

    Parameters
    ----------
    x : ndarray
        A 1-D array which is indexable by `peaks`.
    peaks : ndarray
        Indices of peaks in `x`.
    tmin, tmax : scalar or ndarray or None
         Minimal and / or maximal required thresholds. If supplied as ndarrays
         their size must match `peaks`. ``None`` is interpreted as an open
         border.

    Returns
    -------
    keep : bool
        A boolean mask evaluating to true where `peaks` fulfill the threshold
        condition.
    left_thresholds, right_thresholds : ndarray
        Array matching `peak` containing the thresholds of each peak on
        both sides.

    Notes
    -----

    .. versionadded:: 1.1.0
    """
    # Stack thresholds on both sides to make min / max operations easier:
    # tmin is compared with the smaller, and tmax with the greater threshold to
    # each peak's side
    stacked_thresholds = np.vstack([x[peaks] - x[peaks - 1],
                                    x[peaks] - x[peaks + 1]])
    keep = np.ones(peaks.size, dtype=bool)
    if tmin is not None:
        min_thresholds = np.min(stacked_thresholds, axis=0)
        keep &= (tmin <= min_thresholds)
    if tmax is not None:
        max_thresholds = np.max(stacked_thresholds, axis=0)
        keep &= (max_thresholds <= tmax)

    return keep, stacked_thresholds[0], stacked_thresholds[1]

def _select_by_peak_distance(np.intp_t[::1] peaks not None,
                             np.float64_t[::1] priority not None,
                             np.float64_t distance):
    """
    Evaluate which peaks fulfill the distance condition.

    Parameters
    ----------
    peaks : ndarray
        Indices of peaks in `vector`.
    priority : ndarray
        An array matching `peaks` used to determine priority of each peak. A
        peak with a higher priority value is kept over one with a lower one.
    distance : np.float64
        Minimal distance that peaks must be spaced.

    Returns
    -------
    keep : ndarray[bool]
        A boolean mask evaluating to true where `peaks` fulfill the distance
        condition.

    Notes
    -----
    Declaring the input arrays as C-contiguous doesn't seem to have performance
    advantages.

    .. versionadded:: 1.1.0
    """
    cdef:
        np.uint8_t[::1] keep
        np.intp_t[::1] priority_to_position
        np.intp_t i, j, k, peaks_size, distance_

    peaks_size = peaks.shape[0]
    # Round up because actual peak distance can only be natural number
    distance_ = <np.intp_t>ceil(distance)
    keep = np.ones(peaks_size, dtype=np.uint8)  # Prepare array of flags

    # Create map from `i` (index for `peaks` sorted by `priority`) to `j` (index
    # for `peaks` sorted by position). This allows to iterate `peaks` and `keep`
    # with `j` by order of `priority` while still maintaining the ability to
    # step to neighbouring peaks with (`j` + 1) or (`j` - 1).
    priority_to_position = np.argsort(priority)

    with nogil:
        # Highest priority first -> iterate in reverse order (decreasing)
        for i in range(peaks_size - 1, -1, -1):
            # "Translate" `i` to `j` which points to current peak whose
            # neighbours are to be evaluated
            j = priority_to_position[i]
            if keep[j] == 0:
                # Skip evaluation for peak already marked as "don't keep"
                continue

            k = j - 1
            # Flag "earlier" peaks for removal until minimal distance is exceeded
            while 0 <= k and peaks[j] - peaks[k] < distance_:
                keep[k] = 0
                k -= 1

            k = j + 1
            # Flag "later" peaks for removal until minimal distance is exceeded
            while k < peaks_size and peaks[k] - peaks[j] < distance_:
                keep[k] = 0
                k += 1

    return keep.base.view(dtype=np.bool_)  # Return as boolean array

def peak_widths(x, peaks, rel_height=0.5, prominence_data=None, wlen=None):
    """
    Calculate the width of each peak in a signal.

    This function calculates the width of a peak in samples at a relative
    distance to the peak's height and prominence.

    Parameters
    ----------
    x : sequence
        A signal with peaks.
    peaks : sequence
        Indices of peaks in `x`.
    rel_height : float, optional
        Chooses the relative height at which the peak width is measured as a
        percentage of its prominence. 1.0 calculates the width of the peak at
        its lowest contour line while 0.5 evaluates at half the prominence
        height. Must be at least 0. See notes for further explanation.
    prominence_data : tuple, optional
        A tuple of three arrays matching the output of `peak_prominences` when
        called with the same arguments `x` and `peaks`. This data are calculated
        internally if not provided.
    wlen : int, optional
        A window length in samples passed to `peak_prominences` as an optional
        argument for internal calculation of `prominence_data`. This argument
        is ignored if `prominence_data` is given.

    Returns
    -------
    widths : ndarray
        The widths for each peak in samples.
    width_heights : ndarray
        The height of the contour lines at which the `widths` where evaluated.
    left_ips, right_ips : ndarray
        Interpolated positions of left and right intersection points of a
        horizontal line at the respective evaluation height.

    Raises
    ------
    ValueError
        If `prominence_data` is supplied but doesn't satisfy the condition
        ``0 <= left_base <= peak <= right_base < x.shape[0]`` for each peak,
        has the wrong dtype, is not C-contiguous or does not have the same
        shape.

    Warns
    -----
    PeakPropertyWarning
        Raised if any calculated width is 0. This may stem from the supplied
        `prominence_data` or if `rel_height` is set to 0.

    Warnings
    --------
    This function may return unexpected results for data containing NaNs. To
    avoid this, NaNs should either be removed or replaced.

    See Also
    --------
    find_peaks
        Find peaks inside a signal based on peak properties.
    peak_prominences
        Calculate the prominence of peaks.

    Notes
    -----
    The basic algorithm to calculate a peak's width is as follows:

    * Calculate the evaluation height :math:`h_{eval}` with the formula
      :math:`h_{eval} = h_{Peak} - P \\cdot R`, where :math:`h_{Peak}` is the
      height of the peak itself, :math:`P` is the peak's prominence and
      :math:`R` a positive ratio specified with the argument `rel_height`.
    * Draw a horizontal line at the evaluation height to both sides, starting at
      the peak's current vertical position until the lines either intersect a
      slope, the signal border or cross the vertical position of the peak's
      base (see `peak_prominences` for an definition). For the first case,
      intersection with the signal, the true intersection point is estimated
      with linear interpolation.
    * Calculate the width as the horizontal distance between the chosen
      endpoints on both sides. As a consequence of this the maximal possible
      width for each peak is the horizontal distance between its bases.

    As shown above to calculate a peak's width its prominence and bases must be
    known. You can supply these yourself with the argument `prominence_data`.
    Otherwise, they are internally calculated (see `peak_prominences`).

    .. versionadded:: 1.1.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal import chirp, find_peaks, peak_widths
    >>> import matplotlib.pyplot as plt

    Create a test signal with two overlaid harmonics

    >>> x = np.linspace(0, 6 * np.pi, 1000)
    >>> x = np.sin(x) + 0.6 * np.sin(2.6 * x)

    Find all peaks and calculate their widths at the relative height of 0.5
    (contour line at half the prominence height) and 1 (at the lowest contour
    line at full prominence height).

    >>> peaks, _ = find_peaks(x)
    >>> results_half = peak_widths(x, peaks, rel_height=0.5)
    >>> results_half[0]  # widths
    array([ 64.25172825,  41.29465463,  35.46943289, 104.71586081,
            35.46729324,  41.30429622, 181.93835853,  45.37078546])
    >>> results_full = peak_widths(x, peaks, rel_height=1)
    >>> results_full[0]  # widths
    array([181.9396084 ,  72.99284945,  61.28657872, 373.84622694,
        61.78404617,  72.48822812, 253.09161876,  79.36860878])

    Plot signal, peaks and contour lines at which the widths where calculated

    >>> plt.plot(x)
    >>> plt.plot(peaks, x[peaks], "x")
    >>> plt.hlines(*results_half[1:], color="C2")
    >>> plt.hlines(*results_full[1:], color="C3")
    >>> plt.show()
    """
    x = _arg_x_as_expected(x)
    peaks = _arg_peaks_as_expected(peaks)
    if prominence_data is None:
        # Calculate prominence if not supplied and use wlen if supplied.
        wlen = _arg_wlen_as_expected(wlen)
        prominence_data = _peak_prominences(x, peaks, wlen)
    return _peak_widths(x, peaks, rel_height, *prominence_data)

def _arg_x_as_expected(value):
    """Ensure argument `x` is a 1-D C-contiguous array of dtype('float64').

    Used in `find_peaks`, `peak_prominences` and `peak_widths` to make `x`
    compatible with the signature of the wrapped Cython functions.

    Returns
    -------
    value : ndarray
        A 1-D C-contiguous array with dtype('float64').
    """
    value = np.asarray(value, order='C', dtype=np.float64)
    if value.ndim != 1:
        raise ValueError('`x` must be a 1-D array')
    return value

def _unpack_condition_args(interval, x, peaks):
    """
    Parse condition arguments for `find_peaks`.

    Parameters
    ----------
    interval : number or ndarray or sequence
        Either a number or ndarray or a 2-element sequence of the former. The
        first value is always interpreted as `imin` and the second, if supplied,
        as `imax`.
    x : ndarray
        The signal with `peaks`.
    peaks : ndarray
        An array with indices used to reduce `imin` and / or `imax` if those are
        arrays.

    Returns
    -------
    imin, imax : number or ndarray or None
        Minimal and maximal value in `argument`.

    Raises
    ------
    ValueError :
        If interval border is given as array and its size does not match the size
        of `x`.

    Notes
    -----

    .. versionadded:: 1.1.0
    """
    try:
        imin, imax = interval
    except (TypeError, ValueError):
        imin, imax = (interval, None)

    # Reduce arrays if arrays
    if isinstance(imin, np.ndarray):
        if imin.size != x.size:
            raise ValueError('array size of lower interval border must match x')
        imin = imin[peaks]
    if isinstance(imax, np.ndarray):
        if imax.size != x.size:
            raise ValueError('array size of upper interval border must match x')
        imax = imax[peaks]

    return imin, imax

def _local_maxima_1d(const np.float64_t[::1] x not None):
    """
    Find local maxima in a 1D array.

    This function finds all local maxima in a 1D array and returns the indices
    for their edges and midpoints (rounded down for even plateau sizes).

    Parameters
    ----------
    x : ndarray
        The array to search for local maxima.

    Returns
    -------
    midpoints : ndarray
        Indices of midpoints of local maxima in `x`.
    left_edges : ndarray
        Indices of edges to the left of local maxima in `x`.
    right_edges : ndarray
        Indices of edges to the right of local maxima in `x`.

    Notes
    -----
    - Compared to `argrelmax` this function is significantly faster and can
      detect maxima that are more than one sample wide. However this comes at
      the cost of being only applicable to 1D arrays.
    - A maxima is defined as one or more samples of equal value that are
      surrounded on both sides by at least one smaller sample.

    .. versionadded:: 1.1.0
    """
    cdef:
        np.intp_t[::1] midpoints, left_edges, right_edges
        np.intp_t m, i, i_ahead, i_max

    # Preallocate, there can't be more maxima than half the size of `x`
    midpoints = np.empty(x.shape[0] // 2, dtype=np.intp)
    left_edges = np.empty(x.shape[0] // 2, dtype=np.intp)
    right_edges = np.empty(x.shape[0] // 2, dtype=np.intp)
    m = 0  # Pointer to the end of valid area in allocated arrays

    with nogil:
        i = 1  # Pointer to current sample, first one can't be maxima
        i_max = x.shape[0] - 1  # Last sample can't be maxima
        while i < i_max:
            # Test if previous sample is smaller
            if x[i - 1] < x[i]:
                i_ahead = i + 1  # Index to look ahead of current sample

                # Find next sample that is unequal to x[i]
                while i_ahead < i_max and x[i_ahead] == x[i]:
                    i_ahead += 1

                # Maxima is found if next unequal sample is smaller than x[i]
                if x[i_ahead] < x[i]:
                    left_edges[m] = i
                    right_edges[m] = i_ahead - 1
                    midpoints[m] = (left_edges[m] + right_edges[m]) // 2
                    m += 1
                    # Skip samples that can't be maximum
                    i = i_ahead
            i += 1

    # Keep only valid part of array memory.
    midpoints.base.resize(m, refcheck=False)
    left_edges.base.resize(m, refcheck=False)
    right_edges.base.resize(m, refcheck=False)

    return midpoints.base, left_edges.base, right_edges.base

def find_peaks(x, height=None, threshold=None, distance=None,
               prominence=None, width=None, wlen=None, rel_height=0.5,
               plateau_size=None):
    """
    Find peaks inside a signal based on peak properties.

    This function takes a 1-D array and finds all local maxima by
    simple comparison of neighboring values. Optionally, a subset of these
    peaks can be selected by specifying conditions for a peak's properties.

    Parameters
    ----------
    x : sequence
        A signal with peaks.
    height : number or ndarray or sequence, optional
        Required height of peaks. Either a number, ``None``, an array matching
        `x` or a 2-element sequence of the former. The first element is
        always interpreted as the  minimal and the second, if supplied, as the
        maximal required height.
    threshold : number or ndarray or sequence, optional
        Required threshold of peaks, the vertical distance to its neighboring
        samples. Either a number, ``None``, an array matching `x` or a
        2-element sequence of the former. The first element is always
        interpreted as the  minimal and the second, if supplied, as the maximal
        required threshold.
    distance : number, optional
        Required minimal horizontal distance (>= 1) in samples between
        neighbouring peaks. Smaller peaks are removed first until the condition
        is fulfilled for all remaining peaks.
    prominence : number or ndarray or sequence, optional
        Required prominence of peaks. Either a number, ``None``, an array
        matching `x` or a 2-element sequence of the former. The first
        element is always interpreted as the  minimal and the second, if
        supplied, as the maximal required prominence.
    width : number or ndarray or sequence, optional
        Required width of peaks in samples. Either a number, ``None``, an array
        matching `x` or a 2-element sequence of the former. The first
        element is always interpreted as the  minimal and the second, if
        supplied, as the maximal required width.
    wlen : int, optional
        Used for calculation of the peaks prominences, thus it is only used if
        one of the arguments `prominence` or `width` is given. See argument
        `wlen` in `peak_prominences` for a full description of its effects.
    rel_height : float, optional
        Used for calculation of the peaks width, thus it is only used if `width`
        is given. See argument  `rel_height` in `peak_widths` for a full
        description of its effects.
    plateau_size : number or ndarray or sequence, optional
        Required size of the flat top of peaks in samples. Either a number,
        ``None``, an array matching `x` or a 2-element sequence of the former.
        The first element is always interpreted as the minimal and the second,
        if supplied as the maximal required plateau size.

        .. versionadded:: 1.2.0

    Returns
    -------
    peaks : ndarray
        Indices of peaks in `x` that satisfy all given conditions.
    properties : dict
        A dictionary containing properties of the returned peaks which were
        calculated as intermediate results during evaluation of the specified
        conditions:

        * 'peak_heights'
              If `height` is given, the height of each peak in `x`.
        * 'left_thresholds', 'right_thresholds'
              If `threshold` is given, these keys contain a peaks vertical
              distance to its neighbouring samples.
        * 'prominences', 'right_bases', 'left_bases'
              If `prominence` is given, these keys are accessible. See
              `peak_prominences` for a description of their content.
        * 'width_heights', 'left_ips', 'right_ips'
              If `width` is given, these keys are accessible. See `peak_widths`
              for a description of their content.
        * 'plateau_sizes', left_edges', 'right_edges'
              If `plateau_size` is given, these keys are accessible and contain
              the indices of a peak's edges (edges are still part of the
              plateau) and the calculated plateau sizes.

              .. versionadded:: 1.2.0

        To calculate and return properties without excluding peaks, provide the
        open interval ``(None, None)`` as a value to the appropriate argument
        (excluding `distance`).

    Warns
    -----
    PeakPropertyWarning
        Raised if a peak's properties have unexpected values (see
        `peak_prominences` and `peak_widths`).

    Warnings
    --------
    This function may return unexpected results for data containing NaNs. To
    avoid this, NaNs should either be removed or replaced.

    See Also
    --------
    find_peaks_cwt
        Find peaks using the wavelet transformation.
    peak_prominences
        Directly calculate the prominence of peaks.
    peak_widths
        Directly calculate the width of peaks.

    Notes
    -----
    In the context of this function, a peak or local maximum is defined as any
    sample whose two direct neighbours have a smaller amplitude. For flat peaks
    (more than one sample of equal amplitude wide) the index of the middle
    sample is returned (rounded down in case the number of samples is even).
    For noisy signals the peak locations can be off because the noise might
    change the position of local maxima. In those cases consider smoothing the
    signal before searching for peaks or use other peak finding and fitting
    methods (like `find_peaks_cwt`).

    Some additional comments on specifying conditions:

    * Almost all conditions (excluding `distance`) can be given as half-open or
      closed intervals, e.g., ``1`` or ``(1, None)`` defines the half-open
      interval :math:`[1, \\infty]` while ``(None, 1)`` defines the interval
      :math:`[-\\infty, 1]`. The open interval ``(None, None)`` can be specified
      as well, which returns the matching properties without exclusion of peaks.
    * The border is always included in the interval used to select valid peaks.
    * For several conditions the interval borders can be specified with
      arrays matching `x` in shape which enables dynamic constrains based on
      the sample position.
    * The conditions are evaluated in the following order: `plateau_size`,
      `height`, `threshold`, `distance`, `prominence`, `width`. In most cases
      this order is the fastest one because faster operations are applied first
      to reduce the number of peaks that need to be evaluated later.
    * While indices in `peaks` are guaranteed to be at least `distance` samples
      apart, edges of flat peaks may be closer than the allowed `distance`.
    * Use `wlen` to reduce the time it takes to evaluate the conditions for
      `prominence` or `width` if `x` is large or has many local maxima
      (see `peak_prominences`).

    .. versionadded:: 1.1.0

    Examples
    --------
    To demonstrate this function's usage we use a signal `x` supplied with
    SciPy (see `scipy.datasets.electrocardiogram`). Let's find all peaks (local
    maxima) in `x` whose amplitude lies above 0.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.datasets import electrocardiogram
    >>> from scipy.signal import find_peaks
    >>> x = electrocardiogram()[2000:4000]
    >>> peaks, _ = find_peaks(x, height=0)
    >>> plt.plot(x)
    >>> plt.plot(peaks, x[peaks], "x")
    >>> plt.plot(np.zeros_like(x), "--", color="gray")
    >>> plt.show()

    We can select peaks below 0 with ``height=(None, 0)`` or use arrays matching
    `x` in size to reflect a changing condition for different parts of the
    signal.

    >>> border = np.sin(np.linspace(0, 3 * np.pi, x.size))
    >>> peaks, _ = find_peaks(x, height=(-border, border))
    >>> plt.plot(x)
    >>> plt.plot(-border, "--", color="gray")
    >>> plt.plot(border, ":", color="gray")
    >>> plt.plot(peaks, x[peaks], "x")
    >>> plt.show()

    Another useful condition for periodic signals can be given with the
    `distance` argument. In this case, we can easily select the positions of
    QRS complexes within the electrocardiogram (ECG) by demanding a distance of
    at least 150 samples.

    >>> peaks, _ = find_peaks(x, distance=150)
    >>> np.diff(peaks)
    array([186, 180, 177, 171, 177, 169, 167, 164, 158, 162, 172])
    >>> plt.plot(x)
    >>> plt.plot(peaks, x[peaks], "x")
    >>> plt.show()

    Especially for noisy signals peaks can be easily grouped by their
    prominence (see `peak_prominences`). E.g., we can select all peaks except
    for the mentioned QRS complexes by limiting the allowed prominence to 0.6.

    >>> peaks, properties = find_peaks(x, prominence=(None, 0.6))
    >>> properties["prominences"].max()
    0.5049999999999999
    >>> plt.plot(x)
    >>> plt.plot(peaks, x[peaks], "x")
    >>> plt.show()

    And, finally, let's examine a different section of the ECG which contains
    beat forms of different shape. To select only the atypical heart beats, we
    combine two conditions: a minimal prominence of 1 and width of at least 20
    samples.

    >>> x = electrocardiogram()[17000:18000]
    >>> peaks, properties = find_peaks(x, prominence=1, width=20)
    >>> properties["prominences"], properties["widths"]
    (array([1.495, 2.3  ]), array([36.93773946, 39.32723577]))
    >>> plt.plot(x)
    >>> plt.plot(peaks, x[peaks], "x")
    >>> plt.vlines(x=peaks, ymin=x[peaks] - properties["prominences"],
    ...            ymax = x[peaks], color = "C1")
    >>> plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"],
    ...            xmax=properties["right_ips"], color = "C1")
    >>> plt.show()
    """
    # _argmaxima1d expects array of dtype 'float64'
    x = _arg_x_as_expected(x)
    if distance is not None and distance < 1:
        raise ValueError('`distance` must be greater or equal to 1')

    peaks, left_edges, right_edges = _local_maxima_1d(x)
    properties = {}

    if plateau_size is not None:
        # Evaluate plateau size
        plateau_sizes = right_edges - left_edges + 1
        pmin, pmax = _unpack_condition_args(plateau_size, x, peaks)
        keep = _select_by_property(plateau_sizes, pmin, pmax)
        peaks = peaks[keep]
        properties["plateau_sizes"] = plateau_sizes
        properties["left_edges"] = left_edges
        properties["right_edges"] = right_edges
        properties = {key: array[keep] for key, array in properties.items()}

    if height is not None:
        # Evaluate height condition
        peak_heights = x[peaks]
        hmin, hmax = _unpack_condition_args(height, x, peaks)
        keep = _select_by_property(peak_heights, hmin, hmax)
        peaks = peaks[keep]
        properties["peak_heights"] = peak_heights
        properties = {key: array[keep] for key, array in properties.items()}

    if threshold is not None:
        # Evaluate threshold condition
        tmin, tmax = _unpack_condition_args(threshold, x, peaks)
        keep, left_thresholds, right_thresholds = _select_by_peak_threshold(
            x, peaks, tmin, tmax)
        peaks = peaks[keep]
        properties["left_thresholds"] = left_thresholds
        properties["right_thresholds"] = right_thresholds
        properties = {key: array[keep] for key, array in properties.items()}

    if distance is not None:
        # Evaluate distance condition
        keep = _select_by_peak_distance(peaks, x[peaks], distance)
        peaks = peaks[keep]
        properties = {key: array[keep] for key, array in properties.items()}

    if prominence is not None or width is not None:
        # Calculate prominence (required for both conditions)
        wlen = _arg_wlen_as_expected(wlen)
        properties.update(zip(
            ['prominences', 'left_bases', 'right_bases'],
            _peak_prominences(x, peaks, wlen=wlen)
        ))

    if prominence is not None:
        # Evaluate prominence condition
        pmin, pmax = _unpack_condition_args(prominence, x, peaks)
        keep = _select_by_property(properties['prominences'], pmin, pmax)
        peaks = peaks[keep]
        properties = {key: array[keep] for key, array in properties.items()}

    if width is not None:
        # Calculate widths
        properties.update(zip(
            ['widths', 'width_heights', 'left_ips', 'right_ips'],
            _peak_widths(x, peaks, rel_height, properties['prominences'],
                         properties['left_bases'], properties['right_bases'])
        ))
        # Evaluate width condition
        wmin, wmax = _unpack_condition_args(width, x, peaks)
        keep = _select_by_property(properties['widths'], wmin, wmax)
        peaks = peaks[keep]
        properties = {key: array[keep] for key, array in properties.items()}

    return peaks, properties

