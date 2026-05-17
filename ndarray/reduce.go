package ndarray

import (
	"fmt"
	"math"
)

// axisSet treats axes as the membership list of a set over [0, ndim)
// and returns that set as a bool slice (true at index i iff axis i
// appears in axes). Panics on out-of-range or duplicate axes -- the
// list cannot form a valid set in those cases, so construction fails.
//
// Callers that only need validation can ignore the returned slice;
// callers that need to distinguish "in the set" from "out of the set"
// (e.g. reductions distinguishing reduced from preserved axes) read
// the mask directly.
func axisSet(axes []int, ndim int) []bool {
	seen := make([]bool, ndim)
	for _, axis := range axes {
		if axis < 0 || axis >= ndim {
			panic(fmt.Sprintf("ndarray: axis %d out of range [0, %d)", axis, ndim))
		}

		if seen[axis] {
			panic(fmt.Sprintf("ndarray: axis %d appears more than once", axis))
		}

		seen[axis] = true
	}

	return seen
}

// Reduce collapses a along the given axes by repeatedly applying fn
// starting from init, and returns a fresh contiguous tensor with the
// result.
//
// axes selects which axes of a to reduce. The result's shape is formed
// from a.shape by either dropping the listed axes (keepDims=false) or
// replacing each with size 1 (keepDims=true). Passing an empty axes
// slice reduces over all axes.
//
// fn is invoked exactly a.Size() times in shape order with the running
// accumulator and the next input value.
//
// Reduce panics if any axis is out of range [0, a.Ndim()) or appears
// more than once.
func (a *NDArray) Reduce(axes []int, keepDims bool, init float64, fn func(float64, float64) float64) *NDArray {
	if axes == nil {
		axes = make([]int, a.Ndim())
		for i := range axes {
			axes[i] = i
		}
	}

	seen := axisSet(axes, a.Ndim())
	outShape := make([]int, 0, a.Ndim())

	for axis := range a.shape {
		if seen[axis] {
			if keepDims {
				outShape = append(outShape, 1)
			}
		} else {
			outShape = append(outShape, a.shape[axis])
		}
	}

	out := NewNDArray(outShape...)
	for i := range out.data {
		out.data[i] = init
	}

	// Pre-compute, for each input axis, the output stride it contributes
	// to. Reduced axes contribute 0: dropped axes have no output position,
	// and keepDims size-1 axes always have index 0. This lets the inner
	// loop walk every input element with a single pass over the axes.
	axisToOutStride := make([]int, a.Ndim())
	outAxis := 0

	for axis := range a.shape {
		if seen[axis] {
			if keepDims {
				outAxis++
			}
		} else {
			axisToOutStride[axis] = out.strides[outAxis]
			outAxis++
		}
	}

	for multiIndex, offset := range a.indicesAndOffsets() {
		outOffset := out.offset

		for axis, index := range multiIndex {
			outOffset += index * axisToOutStride[axis]
		}

		out.data[outOffset] = fn(out.data[outOffset], a.data[offset])
	}

	return out
}

// Sum returns a fresh contiguous tensor whose elements are the sums of
// the elements of a along the given axes, under numpy-style reduction
// semantics.
//
// axes selects which axes to collapse. keepDims controls whether the
// collapsed axes are dropped (false) or kept as size 1 (true). Passing
// an empty axes slice sums over all axes of a.
//
// Sum panics if any axis is out of range or duplicated.
func (a *NDArray) Sum(axes []int, keepDims bool) *NDArray {
	return a.Reduce(axes, keepDims, 0, func(acc, x float64) float64 { return acc + x })
}

// Max returns a fresh contiguous tensor whose elements are the maxima of
// the elements of a along the given axes, under numpy-style reduction
// semantics.
//
// axes selects which axes to collapse. keepDims controls whether the
// collapsed axes are dropped (false) or kept as size 1 (true). Passing
// an empty axes slice takes the max over all axes of a.
//
// Max panics if any axis is out of range, duplicated, or refers to a
// reduced axis of size 0 (no elements to compare). The empty-axis check
// matches numpy's behavior.
func (a *NDArray) Max(axes []int, keepDims bool) *NDArray {
	for _, ax := range axes {
		if ax >= 0 && ax < a.Ndim() && a.shape[ax] == 0 {
			panic(fmt.Sprintf("ndarray: max over empty axis %d is undefined", ax))
		}
	}

	return a.Reduce(axes, keepDims, math.Inf(-1), math.Max)
}

// Mean returns a fresh contiguous tensor whose elements are the
// arithmetic means of the elements of a along the given axes, under
// numpy-style reduction semantics.
//
// axes selects which axes to collapse. keepDims controls whether the
// collapsed axes are dropped (false) or kept as size 1 (true). Passing
// an empty axes computes the mean over all axes of a.
//
// Mean is implemented as Sum divided by the product of the reduced
// axes' sizes. If a reduced axis has size 0 the divisor is 0 and the
// result is NaN, per IEEE 754. Callers that need an explicit error
// should guard for this before calling.
//
// Mean panics if any axis is out of range or duplicated.
func (a *NDArray) Mean(axes []int, keepDims bool) *NDArray {
	if axes == nil {
		axes = make([]int, a.Ndim())
		for i := range axes {
			axes[i] = i
		}
	}

	summed := a.Sum(axes, keepDims)

	count := 1
	for _, ax := range axes {
		count *= a.shape[ax]
	}

	inv := 1.0 / float64(count)
	for i := range summed.data {
		summed.data[i] *= inv
	}

	return summed
}
