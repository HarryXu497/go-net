package ndarray

import (
	"fmt"
	"slices"
)

// Reshape returns a view of the same elements shaped into newShape.
// Walking the result in shape order yields the same sequence of values as walking
// the receiver in shape order; only the shape changes.
//
// When a is contiguous, Reshape returns a zero-copy view that uses the same
// underlying buffer: writes through one are visible through the other.
// When a is not contiguous, Reshape first constructs a new contiguous buffer via
// Copy, so the result no longer aliases a.
//
// Examples:
//
//	a := FromSlice([]float64{0, 1, 2, 3, 4, 5}, 2, 3)
//	a.Reshape(3, 2)               // [[0,1],[2,3],[4,5]], zero-copy
//	a.Transpose(1, 0).Reshape(6)  // copies first; result is [0,3,1,4,2,5]
func (a *NDArray) Reshape(newShape ...int) *NDArray {
	size := shapeSize(newShape...)

	if a.Size() != size {
		panic(fmt.Sprintf("cannot reshape size-%d tensor into shape with %d elements", a.Size(), size))
	}

	src := a

	if !a.IsContiguous() {
		src = a.Copy()
	}

	return &NDArray{
		data:    src.data,
		shape:   slices.Clone(newShape),
		strides: rowMajorStrides(newShape),
		offset:  src.offset,
	}
}

// Transpose returns a view with the axes reordered according to perm. The
// returned view aliases a's underlying buffer; no data is copied.
//
// perm specifies, for each axis k of the result, which axis of the receiver
// it corresponds to: result.shape[k] = a.shape[perm[k]] and
// result.strides[k] = a.strides[perm[k]]. perm must be a permutation of
// [0, a.Ndim()) — i.e., its length must equal a.Ndim() and each axis index
// must appear exactly once. Otherwise Transpose panics.
//
// As a convenience, calling Transpose with no arguments reverses all axes
// (perm = [n-1, n-2, ..., 0]). For a rank-2 tensor this is the ordinary
// matrix transpose.
//
// The result is almost always non-contiguous: walking it in shape order
// steps through the buffer in an order other than linear.
//
// Examples:
//
//	a := FromSlice([]float64{0, 1, 2, 3, 4, 5}, 2, 3)
//	a.Transpose(1, 0)   // [[0,3],[1,4],[2,5]]
//	a.Transpose()       // same as a.Transpose(1, 0)
func (a *NDArray) Transpose(perm ...int) *NDArray {
	if len(perm) == 0 {
		perm = make([]int, a.Ndim())
		for i := range perm {
			perm[i] = a.Ndim() - 1 - i
		}
	}

	if len(perm) != a.Ndim() {
		panic(fmt.Sprintf("ndarray: expected %d axes, got %d", a.Ndim(), len(perm)))
	}

	seen := make([]bool, a.Ndim())
	for _, axis := range perm {
		if axis < 0 || axis >= a.Ndim() {
			panic(fmt.Sprintf("ndarray: axis %d out of range [0, %d)", axis, a.Ndim()))
		}

		if seen[axis] {
			panic(fmt.Sprintf("ndarray: axis %d appears more than once", axis))
		}

		seen[axis] = true
	}

	newShape := make([]int, a.Ndim())
	newStrides := make([]int, a.Ndim())

	for newAxis, oldAxis := range perm {
		newShape[newAxis] = a.shape[oldAxis]
		newStrides[newAxis] = a.strides[oldAxis]
	}

	return &NDArray{
		data:    a.data,
		shape:   newShape,
		strides: newStrides,
		offset:  a.offset,
	}
}

// broadcastShape returns the output shape produced by combining shape1 and
// shape2 elementwise under numpy's broadcasting rules.
//
// The two shapes are right-aligned and the shorter one is conceptually padded
// with size-1 axes on the left until the ranks match. Then for each axis k:
//
//   - if shape1[k] == shape2[k], the output size is that shared value;
//   - if one side is 1, the output size is the other (the size-1 side would
//     be stretched when materialized via BroadcastTo);
//   - otherwise the shapes are incompatible and broadcastShape panics.
//
// The result has rank max(len(shape1), len(shape2)). The order of the inputs
// does not matter: broadcastShape is symmetric.
//
// Examples:
//
//	broadcastShape([2 3],   [2 3])    → [2 3]    // identity
//	broadcastShape([10],    [32 10])  → [32 10]  // bias case: shape1 padded to (1,10), stretched on axis 0
//	broadcastShape([32 1],  [32 10])  → [32 10]  // rowMax case: stretched on axis 1
//	broadcastShape([5 1 3], [4 3])    → [5 4 3]  // shape2 padded to (1,4,3); both sides contribute stretches
//	broadcastShape([],      [2 3])    → [2 3]    // scalar broadcasts against any shape
//	broadcastShape([3],     [4])      → panic    // 3 vs 4, neither is 1
func broadcastShape(shape1, shape2 []int) []int {
	dim := max(len(shape1), len(shape2))
	out := make([]int, dim)

	for i := range dim {
		a, b := 1, 1

		if i < len(shape1) {
			a = shape1[len(shape1)-i-1]
		}

		if i < len(shape2) {
			b = shape2[len(shape2)-i-1]
		}

		axis := dim - i - 1

		switch {
		case a == b:
			out[axis] = a
		case a == 1:
			out[axis] = b
		case b == 1:
			out[axis] = a
		default:
			panic(fmt.Sprintf("ndarray: cannot broadcast shapes %v and %v (axis %d: %d vs %d)", shape1, shape2, axis, a, b))
		}
	}

	return out
}

// BroadcastTo returns a view of the receiver expanded to the given target
// shape. The returned view aliases a's underlying buffer; no data is copied.
//
// Broadcasting follows numpy's rules. The receiver's shape is right-aligned
// against target: if target has more axes, size-1 axes are conceptually
// prepended to the source until the ranks match. Then for each axis k:
//
//   - if the source size at k equals target[k], the source's stride is kept;
//   - if the source size at k is 1, the axis is "stretched" to target[k] by
//     setting its stride to 0: every position along that axis reads the
//     same buffer cell;
//   - otherwise the shapes are incompatible and BroadcastTo panics.
//
// Examples:
//
//	a := FromSlice([]float64{1, 2, 3}, 1, 3)
//	a.BroadcastTo(2, 3)    // [[1,2,3],[1,2,3]] — row stretched across axis 0
//
//	b := FromSlice([]float64{1, 2, 3}, 3)
//	b.BroadcastTo(2, 3)    // same result; rank-1 source prepended with a size-1 axis
//
//	c := NewNDArray()      // rank-0 scalar
//	c.Set(nil, 7)
//	c.BroadcastTo(2, 3)    // [[7,7,7],[7,7,7]]
func (a *NDArray) BroadcastTo(target ...int) *NDArray {
	pad := len(target) - a.Ndim()

	if pad < 0 {
		panic(fmt.Sprintf("ndarray: broadcast target shape must have dimension >= %d, got %d", a.Ndim(), len(target)))
	}

	newShape := make([]int, len(target))
	newStrides := make([]int, len(target))

	for i, dim := range target {
		if dim < 0 {
			panic(fmt.Sprintf("ndarray: shape dimension on axis %d must be non-negative, got %d", i, dim))
		}

		var srcDim, srcStride int
		if i < pad {
			srcDim, srcStride = 1, 0
		} else {
			srcDim, srcStride = a.shape[i-pad], a.strides[i-pad]
		}

		switch srcDim {
		case dim:
			newShape[i], newStrides[i] = dim, srcStride
		case 1:
			newShape[i], newStrides[i] = dim, 0
		default:
			panic(fmt.Sprintf("ndarray: cannot broadcast %v to %v (axis %d: %d vs %d)", a.shape, target, i, srcDim, dim))
		}
	}

	return &NDArray{
		data:    a.data,
		shape:   newShape,
		strides: newStrides,
		offset:  a.offset,
	}
}

// Unsqueeze returns a view of the receiver with a size-1 axis inserted at
// position axis. The returned view aliases a's underlying buffer; no data is
// copied. Rank increases by one; the total number of elements is unchanged.
//
// axis must be in [0, a.Ndim()]: 0 inserts at the front, a.Ndim() appends at
// the end. The new axis has size 1 and stride 0.
//
// Unsqueeze is the inverse of Squeeze: a.Unsqueeze(k).Squeeze(k) reproduces a.
//
// Examples:
//
//	a := FromSlice([]float64{1, 2, 3}, 3)
//	a.Unsqueeze(0)   // shape [1, 3] — row vector
//	a.Unsqueeze(1)   // shape [3, 1] — column vector
func (a *NDArray) Unsqueeze(axis int) *NDArray {
	if axis < 0 || axis > a.Ndim() {
		panic(fmt.Sprintf("ndarray: unsqueeze axis %d out of range [0, %d]", axis, a.Ndim()))
	}

	newShape := make([]int, len(a.shape)+1)
	newStrides := make([]int, len(a.strides)+1)

	copy(newShape, a.shape[:axis])
	copy(newStrides, a.strides[:axis])

	newShape[axis] = 1
	newStrides[axis] = 0

	copy(newShape[axis+1:], a.shape[axis:])
	copy(newStrides[axis+1:], a.strides[axis:])

	return &NDArray{
		data:    a.data,
		shape:   newShape,
		strides: newStrides,
		offset:  a.offset,
	}
}

// Squeeze returns a view of the receiver with size-1 axes removed. The
// returned view aliases a's underlying buffer; no data is copied. Rank
// decreases by the number of axes dropped; the total number of elements is
// unchanged.
//
// With no arguments, Squeeze removes every size-1 axis. With explicit axes,
// Squeeze removes only the listed axes and panics if any of them is not
// size 1, is out of range, or appears more than once.
//
// Squeeze is the inverse of Unsqueeze: a.Squeeze(k).Unsqueeze(k) reproduces
// a, provided a.shape[k] == 1.
//
// Examples:
//
//	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 1, 2, 1, 3)
//	a.Squeeze()      // shape [2, 3] — drops both size-1 axes
//	a.Squeeze(0)     // shape [2, 1, 3] — drops only axis 0
//	a.Squeeze(0, 2)  // shape [2, 3] — drops the two listed axes
func (a *NDArray) Squeeze(axes ...int) *NDArray {
	drop := a.squeezeMask(axes)

	newShape := make([]int, 0, a.Ndim())
	newStrides := make([]int, 0, a.Ndim())

	for i, dim := range a.shape {
		if drop[i] {
			continue
		}

		newShape = append(newShape, dim)
		newStrides = append(newStrides, a.strides[i])
	}

	return &NDArray{
		data:    a.data,
		shape:   newShape,
		strides: newStrides,
		offset:  a.offset,
	}
}

// squeezeMask returns a per-axis mask of which axes Squeeze should drop.
// With no axes, every size-1 axis is dropped. With explicit axes, only those
// are dropped; it panics if any is out of range, repeated, or not size 1.
func (a *NDArray) squeezeMask(axes []int) []bool {
	drop := make([]bool, a.Ndim())

	if len(axes) == 0 {
		for i, dim := range a.shape {
			drop[i] = dim == 1
		}

		return drop
	}

	for _, ax := range axes {
		if ax < 0 || ax >= a.Ndim() {
			panic(fmt.Sprintf("ndarray: axis %d out of range [0, %d)", ax, a.Ndim()))
		}

		if drop[ax] {
			panic(fmt.Sprintf("ndarray: axis %d appears more than once", ax))
		}

		if a.shape[ax] != 1 {
			panic(fmt.Sprintf("ndarray: cannot squeeze axis %d with size %d (must be 1)", ax, a.shape[ax]))
		}

		drop[ax] = true
	}

	return drop
}

// SliceArg specifies how a single axis is sliced. Construct values with
// Idx (single index, drops the axis), Rng (half-open range), Step (range
// with a step), or All (the entire axis); do not construct SliceArg
// literals directly.
type SliceArg struct {
	single bool
	all    bool
	start  int
	stop   int
	step   int
}

// Idx returns a SliceArg that selects a single index along an axis and
// drops that axis from the result. i must be in [0, axis size); negative
// indices are not supported in v1.
//
//	a.Slice(Idx(2), All(), All())   // numpy: a[2, :, :]
func Idx(i int) SliceArg {
	return SliceArg{single: true, start: i}
}

// Rng returns a SliceArg that selects the half-open range [start, stop)
// along an axis. The axis is kept; its new size is stop - start. start
// and stop must satisfy 0 <= start <= stop <= axis size.
//
//	a.Slice(Rng(1, 4), All(), All())   // numpy: a[1:4, :, :]
func Rng(start, stop int) SliceArg {
	return SliceArg{start: start, stop: stop, step: 1}
}

// Step returns a SliceArg that selects every step-th element from the
// half-open range [start, stop) along an axis. The axis is kept; its new
// size is ceil((stop - start) / step). step must be positive; negative
// steps for reversing an axis are not supported in v1.
//
//	a.Slice(Step(0, 6, 2), All(), All())   // numpy: a[0:6:2, :, :]
func Step(start, stop, step int) SliceArg {
	return SliceArg{start: start, stop: stop, step: step}
}

// All returns a SliceArg that selects the entire axis. Equivalent to
// numpy's ":". Useful for leaving leading or trailing axes untouched
// while slicing a middle one.
//
//	a.Slice(All(), Idx(0), All())   // numpy: a[:, 0, :]
func All() SliceArg {
	return SliceArg{all: true, step: 1}
}

// Slice returns a view of the receiver narrowed along each axis according
// to args. The returned view aliases a's underlying buffer; no data is
// copied. There must be exactly one SliceArg per axis (len(args) ==
// a.Ndim()); construct args with Idx, Rng, Step, or All.
//
// Per-axis effect on the result's (shape, strides, offset):
//
//   - Idx(i): offset advances by i * strides[k], and axis k is dropped
//     from the result. Rank decreases by one.
//   - Rng(start, stop): offset advances by start * strides[k], and
//     shape[k] becomes stop - start. strides[k] is unchanged.
//   - Step(start, stop, s): offset advances by start * strides[k],
//     shape[k] becomes ceil((stop - start) / s), and strides[k] is
//     multiplied by s.
//   - All(): no change to offset or strides; shape[k] is preserved.
//
// Slice panics if len(args) != a.Ndim(), if any single index is out of
// range, if any range has start > stop or endpoints outside [0, axis
// size], or if any step is non-positive.
//
// The result is non-contiguous whenever a range starts above 0, a step
// is greater than 1, or an inner (non-trailing) axis was dropped via
// Idx.
//
// Examples:
//
//	// a is a 2x3x4 tensor with values 0..23 in row-major order.
//	a := FromSlice([]float64{
//	    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
//	    12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
//	}, 2, 3, 4)
//
//	a.Slice(All(), Idx(1), Rng(1, 3))
//	// numpy a[:, 1, 1:3] — shape [2, 2], values [[5,6],[17,18]]
//
//	a.Slice(Idx(0), All(), Step(0, 4, 2))
//	// numpy a[0, :, 0::2] — shape [3, 2], values [[0,2],[4,6],[8,10]]
func (a *NDArray) Slice(args ...SliceArg) *NDArray {
	if len(args) != a.Ndim() {
		panic(fmt.Sprintf("ndarray: expected %d slice args, got %d", a.Ndim(), len(args)))
	}

	offset := a.offset
	newShape := make([]int, 0, a.Ndim())
	newStrides := make([]int, 0, a.Ndim())

	for axis, arg := range args {
		size := a.shape[axis]
		stride := a.strides[axis]

		switch {
		case arg.all:
			newShape = append(newShape, size)
			newStrides = append(newStrides, stride)

		case arg.single:
			if arg.start < 0 || arg.start >= size {
				panic(fmt.Sprintf("ndarray: index %d on axis %d is out of range [0, %d)", arg.start, axis, size))
			}

			offset += arg.start * stride

		default:
			if arg.step <= 0 {
				panic(fmt.Sprintf("ndarray: step on axis %d must be positive, got %d", axis, arg.step))
			}

			if arg.start < 0 || arg.start > size {
				panic(fmt.Sprintf("ndarray: start %d on axis %d is out of range [0, %d]", arg.start, axis, size))
			}

			if arg.stop < arg.start || arg.stop > size {
				panic(fmt.Sprintf("ndarray: stop %d on axis %d is out of range [%d, %d]", arg.stop, axis, arg.start, size))
			}

			offset += arg.start * stride
			length := (arg.stop - arg.start + arg.step - 1) / arg.step
			newShape = append(newShape, length)
			newStrides = append(newStrides, stride*arg.step)
		}
	}

	return &NDArray{
		data:    a.data,
		shape:   newShape,
		strides: newStrides,
		offset:  offset,
	}
}
