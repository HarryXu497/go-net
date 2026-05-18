package ndarray

import (
	"fmt"
	"math"
	"runtime"
	"sync"
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

// reducedAxesAreTrailing reports whether the reduced axes (the true
// entries of seen) form a contiguous suffix of seen. Vacuously true
// when no axes are reduced (all entries false) and when every axis is
// reduced (all entries true). The trailing-suffix layout is what lets
// the parallel fast path treat each output cell as a contiguous slab
// of the input buffer.
func reducedAxesAreTrailing(seen []bool) bool {
	i := 0
	for i < len(seen) && !seen[i] {
		i++
	}

	for ; i < len(seen); i++ {
		if !seen[i] {
			return false
		}
	}

	return true
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
// fn is invoked exactly a.Size() times. The serial path calls it in
// shape order. The parallel fast paths (contiguous input, offset 0)
// preserve order within each chunk and merge partials by chunk index,
// so fn only needs to be associative for parallel results to match
// serial; commutativity is not required. The trailing-axes path needs
// neither (each output cell is computed independently in shape order).
// Sum, Max, Min, and Product all qualify. A non-associative fn passed
// to a contiguous input that triggers the parallel branch will produce
// results that differ from the serial walk.
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
	out := NewNDArray(reduceOutShape(a.shape, seen, keepDims)...)

	if a.IsContiguous() && a.offset == 0 {
		switch {
		case out.Size() == 1:
			a.reduceFull(out, init, fn)
			return out
		case reducedAxesAreTrailing(seen):
			a.reduceTrailing(out, init, fn)
			return out
		}
	}

	a.reduceSlow(out, seen, keepDims, init, fn)

	return out
}

// reduceOutShape derives the output shape for a reduction. Reduced
// axes either drop out entirely or collapse to size 1 (keepDims).
func reduceOutShape(inShape []int, seen []bool, keepDims bool) []int {
	out := make([]int, 0, len(inShape))

	for axis, dim := range inShape {
		switch {
		case !seen[axis]:
			out = append(out, dim)
		case keepDims:
			out = append(out, 1)
		}
	}

	return out
}

// reduceFull is the full-reduction (scalar output) fast path. The
// input is partitioned across workers; each accumulates its chunk in
// shape order; the per-chunk partials are then folded left-to-right
// by chunk index so fn only needs to be associative, not commutative.
// Chunking is inlined because the parallel branch needs per-worker
// indices to write into the partials slice.
func (a *NDArray) reduceFull(out *NDArray, init float64, fn func(float64, float64) float64) {
	n := a.Size()
	workers := runtime.GOMAXPROCS(0)

	if workers <= 1 || n < minChunkSize*workers {
		result := init
		for i := range n {
			result = fn(result, a.data[i])
		}

		out.data[0] = result

		return
	}

	chunk := (n + workers - 1) / workers
	partials := make([]float64, workers)

	var (
		wg      sync.WaitGroup
		spawned int
	)

	for w := range workers {
		start := w * chunk
		end := min((w+1)*chunk, n)

		if start >= end {
			break
		}

		spawned++

		wg.Go(func() {
			local := init
			for i := start; i < end; i++ {
				local = fn(local, a.data[i])
			}

			partials[w] = local
		})
	}

	wg.Wait()

	result := init
	for i := range spawned {
		result = fn(result, partials[i])
	}

	out.data[0] = result
}

// reduceTrailing is the trailing-axes-suffix fast path: each output
// cell owns a contiguous input slab of length reducedSize, so output
// cells write disjointly across workers without synchronization. The
// shape-order accumulation within each cell makes this path safe for
// any fn (associativity and commutativity are both unnecessary).
func (a *NDArray) reduceTrailing(out *NDArray, init float64, fn func(float64, float64) float64) {
	outerSize := out.Size()
	reducedSize := a.Size() / outerSize

	parallelForWork(outerSize, reducedSize, func(start, end int) {
		for o := start; o < end; o++ {
			acc := init

			base := o * reducedSize
			for k := range reducedSize {
				acc = fn(acc, a.data[base+k])
			}

			out.data[o] = acc
		}
	})
}

// reduceSlow is the general iterator-driven fallback for non-contiguous
// inputs and reductions whose reduced axes don't form a trailing
// suffix. Many input cells map to each output cell, so the inner step
// is a read-modify-write and stays serial.
func (a *NDArray) reduceSlow(out *NDArray, seen []bool, keepDims bool, init float64, fn func(float64, float64) float64) {
	for i := range out.data {
		out.data[i] = init
	}

	// Pre-compute, for each input axis, the output stride it contributes
	// to. Reduced axes contribute 0: dropped axes have no output position,
	// and keepDims size-1 axes always have index 0.
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
