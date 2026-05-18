package ndarray

import (
	"fmt"
	"slices"
)

// MatMul returns the matrix product of a and b as a fresh contiguous
// tensor, following numpy's semantics: the last two axes are the
// matrix dimensions and any leading axes are broadcast as batch dims.
//
// Both operands must have rank >= 2. The contracted axis is a.shape[-1]
// (= K) against b.shape[-2] (= K); the surviving matrix axes are
// a.shape[-2] (= M) and b.shape[-1] (= N).
//
// Leading dimensions broadcast under the same rules as Add/Mul: shapes
// are right-aligned, the shorter is padded with size-1 axes on the
// left, and any size-1 axis is stretched against the other side. The
// result shape is broadcastShape(a.shape[:-2], b.shape[:-2]) +
// [M, N].
//
// MatMul panics if either operand has rank < 2, if the contracted
// axes disagree, or if the leading dims fail to broadcast.
//
// The inner kernel partitions the flattened (batch, M) output rows
// across GOMAXPROCS workers via parallelForWork. Each worker writes a
// disjoint span of out.data and only reads from aView/bView, so the
// op is race-free. Small problems (batchSize*M*N*K below the
// parallelForWork threshold) fall back to a single-goroutine pass.
//
// Examples:
//
//	// 2-D × 2-D (the Linear-layer case)
//	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)        // (2, 3)
//	b := FromSlice([]float64{7, 8, 9, 10, 11, 12}, 3, 2)     // (3, 2)
//	a.MatMul(b)                                              // (2, 2)
//
//	// 3-D × 3-D — batched 2-D matmul over the leading axis
//	x := NewNDArray(5, 3, 4)
//	w := NewNDArray(5, 4, 2)
//	x.MatMul(w)                                              // (5, 3, 2)
//
//	// 3-D × 2-D — right operand broadcast across the batch
//	x := NewNDArray(5, 3, 4)
//	w := NewNDArray(4, 2)
//	x.MatMul(w)                                              // (5, 3, 2)
//
//	// Multi-axis batch broadcast
//	x := NewNDArray(7, 1, 3, 4)
//	w := NewNDArray(1, 5, 4, 2)
//	x.MatMul(w)                                              // (7, 5, 3, 2)
func (a *NDArray) MatMul(b *NDArray) *NDArray {
	if a.Ndim() < 2 {
		panic(fmt.Sprintf("ndarray: matmul requires rank >= 2 on the left, got shape %v", a.shape))
	}

	if b.Ndim() < 2 {
		panic(fmt.Sprintf("ndarray: matmul requires rank >= 2 on the right, got shape %v", b.shape))
	}

	M := a.shape[a.Ndim()-2]
	K := a.shape[a.Ndim()-1]
	Kb := b.shape[b.Ndim()-2]
	N := b.shape[b.Ndim()-1]

	if K != Kb {
		panic(fmt.Sprintf("ndarray: matmul shape mismatch: cannot contract %v with %v (last axis %d vs second-to-last axis %d)", a.shape, b.shape, K, Kb))
	}

	// Broadcast the leading (batch) dims against each other; the matrix
	// dims pass through unchanged.
	aLead := a.shape[:a.Ndim()-2]
	bLead := b.shape[:b.Ndim()-2]
	leadOut := broadcastShape(aLead, bLead)
	leadRank := len(leadOut)

	// Expand both operands so their leading axes match leadOut. Stretched
	// axes get stride 0; the matrix axes keep their original strides.
	aView := a.BroadcastTo(slices.Concat(leadOut, []int{M, K})...)
	bView := b.BroadcastTo(slices.Concat(leadOut, []int{K, N})...)
	out := NewNDArray(slices.Concat(leadOut, []int{M, N})...)

	// Strides for the inner 2-D kernel.
	aStrideM := aView.strides[leadRank]
	aStrideK := aView.strides[leadRank+1]
	bStrideK := bView.strides[leadRank]
	bStrideN := bView.strides[leadRank+1]
	outStrideM := out.strides[leadRank]
	outStrideN := out.strides[leadRank+1]

	// Walk every batch position. For rank-2 inputs leadOut is empty and
	// this loop runs exactly once with all base offsets equal to the
	// view offsets, recovering the plain 2-D case.
	batchSize := 1
	for _, d := range leadOut {
		batchSize *= d
	}

	rows := batchSize * M

	parallelForWork(rows, N*K, func(start, end int) {
		batchIdx := make([]int, leadRank)

		for r := start; r < end; r++ {
			batch := r / M
			i := r % M

			// Unravel the flat batch counter into a multi-index over leadOut.
			rem := batch
			for k := leadRank - 1; k >= 0; k-- {
				batchIdx[k] = rem % leadOut[k]
				rem /= leadOut[k]
			}

			// Base buffer offsets of the 2-D slice for this batch index.
			aBase := aView.offset
			bBase := bView.offset
			outBase := out.offset

			for k, idx := range batchIdx {
				aBase += idx * aView.strides[k]
				bBase += idx * bView.strides[k]
				outBase += idx * out.strides[k]
			}

			// Inner kernel: out[i, j] = sum_k a[i, k] * b[k, j].
			aRow := aBase + i*aStrideM
			outRow := outBase + i*outStrideM

			for j := range N {
				var sum float64
				for k := range K {
					sum += aView.data[aRow+k*aStrideK] *
						bView.data[bBase+k*bStrideK+j*bStrideN]
				}

				out.data[outRow+j*outStrideN] = sum
			}
		}
	})

	return out
}
