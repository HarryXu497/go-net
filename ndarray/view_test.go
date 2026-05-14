package ndarray

import (
	"slices"
	"testing"
)

func TestReshapeBasicShapeChange(t *testing.T) {
	a := FromSlice([]float64{0, 1, 2, 3, 4, 5}, 2, 3)
	b := a.Reshape(3, 2)

	if !slices.Equal(b.Shape(), []int{3, 2}) {
		t.Errorf("shape = %v, want [3 2]", b.Shape())
	}

	want := []float64{0, 1, 2, 3, 4, 5}

	var got []float64
	for v := range b.All() {
		got = append(got, v)
	}

	if !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestReshapeFlattenToVector(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := a.Reshape(6)

	if !slices.Equal(b.Shape(), []int{6}) {
		t.Errorf("shape = %v, want [6]", b.Shape())
	}

	var got []float64
	for v := range b.All() {
		got = append(got, v)
	}

	want := []float64{1, 2, 3, 4, 5, 6}
	if !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestReshapeVectorToHigherRank(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 12)
	b := a.Reshape(2, 3, 2)

	if !slices.Equal(b.Shape(), []int{2, 3, 2}) {
		t.Errorf("shape = %v, want [2 3 2]", b.Shape())
	}

	// element [1, 2, 1] should be the 12th value (1-indexed) = 12.
	if got := b.Get([]int{1, 2, 1}); got != 12 {
		t.Errorf("b[1,2,1] = %v, want 12", got)
	}

	// element [0, 0, 0] should be the first value = 1.
	if got := b.Get([]int{0, 0, 0}); got != 1 {
		t.Errorf("b[0,0,0] = %v, want 1", got)
	}
}

func TestReshapeScalarToSizeOneVector(t *testing.T) {
	a := NewNDArray()
	a.Set([]int{}, 7)

	b := a.Reshape(1)

	if !slices.Equal(b.Shape(), []int{1}) {
		t.Errorf("shape = %v, want [1]", b.Shape())
	}

	if got := b.Get([]int{0}); got != 7 {
		t.Errorf("b[0] = %v, want 7", got)
	}
}

func TestReshapeSizeOneVectorToScalar(t *testing.T) {
	a := FromSlice([]float64{42}, 1)
	b := a.Reshape()

	if b.Ndim() != 0 {
		t.Errorf("Ndim = %d, want 0", b.Ndim())
	}

	if got := b.Get([]int{}); got != 42 {
		t.Errorf("scalar value = %v, want 42", got)
	}
}

func TestReshapeResultIsContiguous(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := a.Reshape(3, 2)

	if !b.IsContiguous() {
		t.Error("reshape of contiguous source should be contiguous")
	}

	// Also check the non-contiguous path: reshape of a transpose.
	c := a.Transpose(1, 0).Reshape(6)
	if !c.IsContiguous() {
		t.Error("reshape of non-contiguous source should still be contiguous")
	}
}

func TestReshapeZeroCopyWhenContiguous(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := a.Reshape(3, 2)

	// Mutation through the reshape should be visible through the source.
	b.Set([]int{0, 0}, 99)

	if got := a.Get([]int{0, 0}); got != 99 {
		t.Errorf("after b.Set, a[0,0] = %v, want 99 (reshape should be zero-copy on contiguous input)", got)
	}

	// And vice versa.
	a.Set([]int{1, 2}, -1)

	if got := b.Get([]int{2, 1}); got != -1 {
		t.Errorf("after a.Set, b[2,1] = %v, want -1", got)
	}
}

func TestReshapeOfNonContiguousMaterializesCopy(t *testing.T) {
	a := FromSlice([]float64{0, 1, 2, 3, 4, 5}, 2, 3)
	// Transpose to 3x2 then flatten. Walking the transposed view in shape order
	// yields [0,3,1,4,2,5], so the reshape's elements must match.
	b := a.Transpose(1, 0).Reshape(6)

	want := []float64{0, 3, 1, 4, 2, 5}

	var got []float64
	for v := range b.All() {
		got = append(got, v)
	}

	if !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}

	// And the result must not alias the source's buffer.
	b.Set([]int{0}, 99)

	if a.Get([]int{0, 0}) == 99 {
		t.Error("reshape of non-contiguous view should not alias source buffer")
	}
}

func TestReshapePanicsOnSizeMismatch(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)

	assertPanics(t, func() { a.Reshape(2, 2) })
	assertPanics(t, func() { a.Reshape(7) })
	assertPanics(t, func() { a.Reshape() }) // shape () has size 1, not 6
}

func TestReshapeShapeAliasingDefense(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)

	newShape := []int{3, 2}
	b := a.Reshape(newShape...)
	newShape[0] = 99

	if b.Shape()[0] != 3 {
		t.Errorf("b.Shape()[0] = %d, want 3 (caller's newShape mutation leaked through)", b.Shape()[0])
	}
}

func TestReshapeZeroSizedTensor(t *testing.T) {
	// shape (0, 3) has size 0; reshape to (3, 0) is also size 0 and should be valid.
	a := NewNDArray(0, 3)
	b := a.Reshape(3, 0)

	if !slices.Equal(b.Shape(), []int{3, 0}) {
		t.Errorf("shape = %v, want [3 0]", b.Shape())
	}

	if b.Size() != 0 {
		t.Errorf("Size = %d, want 0", b.Size())
	}
}

func TestTranspose2D(t *testing.T) {
	// 2x3:  [[0,1,2],[3,4,5]]
	// T:    [[0,3],[1,4],[2,5]]
	a := FromSlice([]float64{0, 1, 2, 3, 4, 5}, 2, 3)
	b := a.Transpose(1, 0)

	if !slices.Equal(b.Shape(), []int{3, 2}) {
		t.Errorf("shape = %v, want [3 2]", b.Shape())
	}

	var got []float64
	for v := range b.All() {
		got = append(got, v)
	}

	want := []float64{0, 3, 1, 4, 2, 5}
	if !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestTransposeNoArgsReversesAllAxes(t *testing.T) {
	a := FromSlice([]float64{0, 1, 2, 3, 4, 5}, 2, 3)

	b := a.Transpose()
	c := a.Transpose(1, 0)

	if !slices.Equal(b.Shape(), c.Shape()) {
		t.Errorf("no-args shape = %v, want %v", b.Shape(), c.Shape())
	}

	if !slices.Equal(b.Strides(), c.Strides()) {
		t.Errorf("no-args strides = %v, want %v", b.Strides(), c.Strides())
	}
}

func TestTransposeNoArgsHigherRank(t *testing.T) {
	// 2x3x4, default perm should reverse to (2, 1, 0) → shape (4, 3, 2).
	a := NewNDArray(2, 3, 4)
	b := a.Transpose()

	if !slices.Equal(b.Shape(), []int{4, 3, 2}) {
		t.Errorf("shape = %v, want [4 3 2]", b.Shape())
	}

	// strides should be permuted accordingly: a had [12, 4, 1] → reversed [1, 4, 12].
	if !slices.Equal(b.Strides(), []int{1, 4, 12}) {
		t.Errorf("strides = %v, want [1 4 12]", b.Strides())
	}
}

func TestTransposeExplicitPermHigherRank(t *testing.T) {
	// 2x3x4 transposed with perm (2, 0, 1) → shape (4, 2, 3).
	a := NewNDArray(2, 3, 4)
	b := a.Transpose(2, 0, 1)

	if !slices.Equal(b.Shape(), []int{4, 2, 3}) {
		t.Errorf("shape = %v, want [4 2 3]", b.Shape())
	}

	// a.strides = [12, 4, 1]. perm picks strides[2], strides[0], strides[1] → [1, 12, 4].
	if !slices.Equal(b.Strides(), []int{1, 12, 4}) {
		t.Errorf("strides = %v, want [1 12 4]", b.Strides())
	}
}

func TestTransposeAliasesBuffer(t *testing.T) {
	a := FromSlice([]float64{0, 1, 2, 3, 4, 5}, 2, 3)
	b := a.Transpose(1, 0)

	// Mutating the source should be visible through the transpose.
	a.Set([]int{0, 1}, 99)

	if got := b.Get([]int{1, 0}); got != 99 {
		t.Errorf("after a.Set([0,1]=99), b[1,0] = %v, want 99", got)
	}

	// And vice versa.
	b.Set([]int{2, 1}, -1)

	if got := a.Get([]int{1, 2}); got != -1 {
		t.Errorf("after b.Set([2,1]=-1), a[1,2] = %v, want -1", got)
	}
}

func TestTransposeScalar(t *testing.T) {
	// Transposing a scalar (rank 0) with no args is a no-op.
	a := NewNDArray()
	a.Set([]int{}, 42)

	b := a.Transpose()

	if b.Ndim() != 0 {
		t.Errorf("Ndim = %d, want 0", b.Ndim())
	}

	if got := b.Get([]int{}); got != 42 {
		t.Errorf("scalar value = %v, want 42", got)
	}
}

func TestTransposeNotContiguous(t *testing.T) {
	// 2D transpose of a non-trivial shape is non-contiguous.
	a := FromSlice([]float64{0, 1, 2, 3, 4, 5}, 2, 3)
	b := a.Transpose(1, 0)

	if b.IsContiguous() {
		t.Error("transpose of 2x3 should not be contiguous")
	}
}

func TestTransposePanicsOnWrongLength(t *testing.T) {
	a := NewNDArray(2, 3)

	assertPanics(t, func() { a.Transpose(0) })       // too few
	assertPanics(t, func() { a.Transpose(0, 1, 2) }) // too many
}

func TestTransposePanicsOnOutOfRange(t *testing.T) {
	a := NewNDArray(2, 3)

	assertPanics(t, func() { a.Transpose(0, 2) })
	assertPanics(t, func() { a.Transpose(-1, 0) })
}

func TestTransposePanicsOnDuplicateAxis(t *testing.T) {
	a := NewNDArray(2, 3)

	assertPanics(t, func() { a.Transpose(0, 0) })
}

func TestBroadcastToStretchSize1Axis(t *testing.T) {
	// Row (1, 3) stretched across axis 0 to (2, 3).
	a := FromSlice([]float64{1, 2, 3}, 1, 3)
	b := a.BroadcastTo(2, 3)

	if !slices.Equal(b.Shape(), []int{2, 3}) {
		t.Errorf("shape = %v, want [2 3]", b.Shape())
	}

	var got []float64
	for v := range b.All() {
		got = append(got, v)
	}

	want := []float64{1, 2, 3, 1, 2, 3}
	if !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestBroadcastToRankPrepend(t *testing.T) {
	// Rank-1 source broadcast against higher-rank target: a size-1 axis is
	// conceptually prepended to the source.
	a := FromSlice([]float64{1, 2, 3}, 3)
	b := a.BroadcastTo(2, 3)

	if !slices.Equal(b.Shape(), []int{2, 3}) {
		t.Errorf("shape = %v, want [2 3]", b.Shape())
	}

	var got []float64
	for v := range b.All() {
		got = append(got, v)
	}

	want := []float64{1, 2, 3, 1, 2, 3}
	if !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}

	// The prepended axis should have stride 0.
	if !slices.Equal(b.Strides(), []int{0, 1}) {
		t.Errorf("strides = %v, want [0 1]", b.Strides())
	}
}

func TestBroadcastToFromScalar(t *testing.T) {
	a := NewNDArray()
	a.Set([]int{}, 7)

	b := a.BroadcastTo(2, 3)

	if !slices.Equal(b.Shape(), []int{2, 3}) {
		t.Errorf("shape = %v, want [2 3]", b.Shape())
	}

	// All six cells should read the scalar value.
	for v := range b.All() {
		if v != 7 {
			t.Errorf("broadcast scalar value = %v, want 7", v)
		}
	}

	// Both axes are stretched, so both strides should be 0.
	if !slices.Equal(b.Strides(), []int{0, 0}) {
		t.Errorf("strides = %v, want [0 0]", b.Strides())
	}
}

func TestBroadcastToSameShapeIsNoOp(t *testing.T) {
	// Broadcasting to the same shape preserves shape and strides.
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := a.BroadcastTo(2, 3)

	if !slices.Equal(b.Shape(), []int{2, 3}) {
		t.Errorf("shape = %v, want [2 3]", b.Shape())
	}

	if !slices.Equal(b.Strides(), []int{3, 1}) {
		t.Errorf("strides = %v, want [3 1]", b.Strides())
	}
}

func TestBroadcastToAliasesBuffer(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3}, 1, 3)
	b := a.BroadcastTo(2, 3)

	// Writing through the source should be visible at every stretched row.
	a.Set([]int{0, 1}, 99)

	if got := b.Get([]int{0, 1}); got != 99 {
		t.Errorf("b[0,1] = %v, want 99", got)
	}

	if got := b.Get([]int{1, 1}); got != 99 {
		t.Errorf("b[1,1] = %v, want 99 (stretched axis revisits the same cell)", got)
	}
}

func TestBroadcastToHigherRank(t *testing.T) {
	// (1, 3) broadcast to (4, 2, 3): one prepended axis (size 4) and one
	// stretched axis (size 1 → 2).
	a := FromSlice([]float64{10, 20, 30}, 1, 3)
	b := a.BroadcastTo(4, 2, 3)

	if !slices.Equal(b.Shape(), []int{4, 2, 3}) {
		t.Errorf("shape = %v, want [4 2 3]", b.Shape())
	}

	// strides: prepended axis → 0, stretched axis → 0, matched axis → src stride (1).
	if !slices.Equal(b.Strides(), []int{0, 0, 1}) {
		t.Errorf("strides = %v, want [0 0 1]", b.Strides())
	}

	// Spot-check a few cells.
	for i := range 4 {
		for j := range 2 {
			for k := range 3 {
				if got, want := b.Get([]int{i, j, k}), float64((k+1)*10); got != want {
					t.Errorf("b[%d,%d,%d] = %v, want %v", i, j, k, got, want)
				}
			}
		}
	}
}

func TestBroadcastToPanicsOnLowerRankTarget(t *testing.T) {
	// Source has rank 2; target rank 1 is invalid (target rank must be >= source rank).
	a := NewNDArray(2, 3)

	assertPanics(t, func() { a.BroadcastTo(6) })
}

func TestBroadcastToPanicsOnIncompatibleShape(t *testing.T) {
	// Source axis size 2 cannot stretch to target size 3.
	a := NewNDArray(2, 3)

	assertPanics(t, func() { a.BroadcastTo(3, 3) })
	assertPanics(t, func() { a.BroadcastTo(2, 4) })
}

func TestBroadcastToPanicsOnNegativeDim(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3}, 1, 3)

	assertPanics(t, func() { a.BroadcastTo(-1, 3) })
}

func TestUnsqueezeAtFront(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3}, 3)
	b := a.Unsqueeze(0)

	if !slices.Equal(b.Shape(), []int{1, 3}) {
		t.Errorf("shape = %v, want [1 3]", b.Shape())
	}

	if !slices.Equal(b.Strides(), []int{0, 1}) {
		t.Errorf("strides = %v, want [0 1]", b.Strides())
	}

	for j := range 3 {
		if got, want := b.Get([]int{0, j}), float64(j+1); got != want {
			t.Errorf("b[0,%d] = %v, want %v", j, got, want)
		}
	}
}

func TestUnsqueezeInMiddle(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := a.Unsqueeze(1)

	if !slices.Equal(b.Shape(), []int{2, 1, 3}) {
		t.Errorf("shape = %v, want [2 1 3]", b.Shape())
	}

	// Original strides (3, 1) split around a stride-0 axis: (3, 0, 1).
	if !slices.Equal(b.Strides(), []int{3, 0, 1}) {
		t.Errorf("strides = %v, want [3 0 1]", b.Strides())
	}
}

func TestUnsqueezeScalar(t *testing.T) {
	a := NewNDArray()
	a.Set([]int{}, 5)

	b := a.Unsqueeze(0)

	if !slices.Equal(b.Shape(), []int{1}) {
		t.Errorf("shape = %v, want [1]", b.Shape())
	}

	if got := b.Get([]int{0}); got != 5 {
		t.Errorf("b[0] = %v, want 5", got)
	}
}

func TestUnsqueezeAliasesBuffer(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3}, 3)
	b := a.Unsqueeze(0)

	a.Set([]int{1}, 99)

	if got := b.Get([]int{0, 1}); got != 99 {
		t.Errorf("b[0,1] = %v, want 99", got)
	}
}

func TestUnsqueezePanicsOnAxisOutOfRange(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3}, 3)

	assertPanics(t, func() { a.Unsqueeze(-1) })
	assertPanics(t, func() { a.Unsqueeze(2) }) // valid range is [0, 1]
}

func TestSqueezeNoArgsRemovesAllSize1(t *testing.T) {
	// shape (1, 2, 1, 3) → (2, 3) after dropping both size-1 axes.
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 1, 2, 1, 3)
	b := a.Squeeze()

	if !slices.Equal(b.Shape(), []int{2, 3}) {
		t.Errorf("shape = %v, want [2 3]", b.Shape())
	}

	var got []float64
	for v := range b.All() {
		got = append(got, v)
	}

	want := []float64{1, 2, 3, 4, 5, 6}
	if !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestSqueezeSingleExplicitAxis(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 1, 2, 1, 3)
	b := a.Squeeze(0)

	if !slices.Equal(b.Shape(), []int{2, 1, 3}) {
		t.Errorf("shape = %v, want [2 1 3]", b.Shape())
	}
}

func TestSqueezeMultipleExplicitAxes(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 1, 2, 1, 3)
	b := a.Squeeze(0, 2)

	if !slices.Equal(b.Shape(), []int{2, 3}) {
		t.Errorf("shape = %v, want [2 3]", b.Shape())
	}
}

func TestSqueezeAllToScalar(t *testing.T) {
	// shape (1, 1, 1) with one element → scalar after Squeeze().
	a := FromSlice([]float64{42}, 1, 1, 1)
	b := a.Squeeze()

	if b.Ndim() != 0 {
		t.Errorf("Ndim = %d, want 0", b.Ndim())
	}

	if got := b.Get([]int{}); got != 42 {
		t.Errorf("scalar value = %v, want 42", got)
	}
}

func TestSqueezeAliasesBuffer(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3}, 1, 3)
	b := a.Squeeze()

	a.Set([]int{0, 1}, 99)

	if got := b.Get([]int{1}); got != 99 {
		t.Errorf("b[1] = %v, want 99", got)
	}
}

func TestSqueezeInverseOfUnsqueeze(t *testing.T) {
	// a.Unsqueeze(k).Squeeze(k) reproduces a's shape and strides.
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)

	for _, k := range []int{0, 1, 2} {
		b := a.Unsqueeze(k).Squeeze(k)

		if !slices.Equal(b.Shape(), a.Shape()) {
			t.Errorf("k=%d: shape = %v, want %v", k, b.Shape(), a.Shape())
		}

		if !slices.Equal(b.Strides(), a.Strides()) {
			t.Errorf("k=%d: strides = %v, want %v", k, b.Strides(), a.Strides())
		}
	}
}

func TestSqueezePanicsOnAxisOutOfRange(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3}, 1, 3)

	assertPanics(t, func() { a.Squeeze(-1) })
	assertPanics(t, func() { a.Squeeze(2) })
}

func TestSqueezePanicsOnDuplicateAxis(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3}, 1, 1, 3)

	assertPanics(t, func() { a.Squeeze(0, 0) })
}

func TestSqueezePanicsOnNonSize1Axis(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)

	assertPanics(t, func() { a.Squeeze(0) })
}

func TestSliceAllPreservesAxis(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := a.Slice(All(), All())

	if !slices.Equal(b.Shape(), []int{2, 3}) {
		t.Errorf("shape = %v, want [2 3]", b.Shape())
	}

	if !slices.Equal(b.Strides(), []int{3, 1}) {
		t.Errorf("strides = %v, want [3 1]", b.Strides())
	}
}

func TestSliceIdxDropsAxis(t *testing.T) {
	// numpy a[1, :] on a 2x3.
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := a.Slice(Idx(1), All())

	if !slices.Equal(b.Shape(), []int{3}) {
		t.Errorf("shape = %v, want [3]", b.Shape())
	}

	var got []float64
	for v := range b.All() {
		got = append(got, v)
	}

	want := []float64{4, 5, 6}
	if !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestSliceRngNarrowsAxis(t *testing.T) {
	// numpy a[:, 1:3] on a 2x3.
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := a.Slice(All(), Rng(1, 3))

	if !slices.Equal(b.Shape(), []int{2, 2}) {
		t.Errorf("shape = %v, want [2 2]", b.Shape())
	}

	var got []float64
	for v := range b.All() {
		got = append(got, v)
	}

	want := []float64{2, 3, 5, 6}
	if !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestSliceRngEmpty(t *testing.T) {
	// Rng(2, 2) is a valid empty range — start == stop.
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := a.Slice(All(), Rng(2, 2))

	if !slices.Equal(b.Shape(), []int{2, 0}) {
		t.Errorf("shape = %v, want [2 0]", b.Shape())
	}

	if b.Size() != 0 {
		t.Errorf("Size = %d, want 0", b.Size())
	}
}

func TestSliceStep(t *testing.T) {
	// numpy a[0::2] on a vector of length 6: pick indices 0, 2, 4.
	a := FromSlice([]float64{10, 11, 12, 13, 14, 15}, 6)
	b := a.Slice(Step(0, 6, 2))

	if !slices.Equal(b.Shape(), []int{3}) {
		t.Errorf("shape = %v, want [3]", b.Shape())
	}

	// strides should be source stride * step = 1 * 2 = 2.
	if !slices.Equal(b.Strides(), []int{2}) {
		t.Errorf("strides = %v, want [2]", b.Strides())
	}

	var got []float64
	for v := range b.All() {
		got = append(got, v)
	}

	want := []float64{10, 12, 14}
	if !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestSliceStepWithUnevenRange(t *testing.T) {
	// (stop - start + step - 1) / step = (7 - 1 + 2) / 2 = 4 → length 4.
	a := FromSlice([]float64{0, 1, 2, 3, 4, 5, 6}, 7)
	b := a.Slice(Step(1, 7, 2))

	if !slices.Equal(b.Shape(), []int{3}) {
		t.Errorf("shape = %v, want [3]", b.Shape())
	}

	var got []float64
	for v := range b.All() {
		got = append(got, v)
	}

	// 1, 3, 5 — index 7 is out of range; range stops at 7.
	want := []float64{1, 3, 5}
	if !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestSliceDocExampleIdxBetweenRanges(t *testing.T) {
	// 2x3x4 with values 0..23 in row-major order.
	// a[:, 1, 1:3] → shape [2, 2], values [[5, 6], [17, 18]].
	a := FromSlice([]float64{
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
		12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
	}, 2, 3, 4)
	b := a.Slice(All(), Idx(1), Rng(1, 3))

	if !slices.Equal(b.Shape(), []int{2, 2}) {
		t.Errorf("shape = %v, want [2 2]", b.Shape())
	}

	var got []float64
	for v := range b.All() {
		got = append(got, v)
	}

	want := []float64{5, 6, 17, 18}
	if !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestSliceDocExampleIdxAndStep(t *testing.T) {
	// a[0, :, 0::2] → shape [3, 2], values [[0, 2], [4, 6], [8, 10]].
	a := FromSlice([]float64{
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
		12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
	}, 2, 3, 4)
	b := a.Slice(Idx(0), All(), Step(0, 4, 2))

	if !slices.Equal(b.Shape(), []int{3, 2}) {
		t.Errorf("shape = %v, want [3 2]", b.Shape())
	}

	var got []float64
	for v := range b.All() {
		got = append(got, v)
	}

	want := []float64{0, 2, 4, 6, 8, 10}
	if !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestSliceAliasesBuffer(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := a.Slice(Idx(1), All())

	a.Set([]int{1, 0}, 99)

	if got := b.Get([]int{0}); got != 99 {
		t.Errorf("b[0] = %v, want 99 after a.Set([1,0]=99)", got)
	}

	b.Set([]int{2}, -1)

	if got := a.Get([]int{1, 2}); got != -1 {
		t.Errorf("a[1,2] = %v, want -1 after b.Set([2]=-1)", got)
	}
}

func TestSlicePanicsOnWrongNumberOfArgs(t *testing.T) {
	a := NewNDArray(2, 3)

	assertPanics(t, func() { a.Slice(All()) })
	assertPanics(t, func() { a.Slice(All(), All(), All()) })
}

func TestSlicePanicsOnIdxOutOfRange(t *testing.T) {
	a := NewNDArray(2, 3)

	assertPanics(t, func() { a.Slice(Idx(2), All()) })
	assertPanics(t, func() { a.Slice(Idx(-1), All()) })
}

func TestSlicePanicsOnRangeBounds(t *testing.T) {
	a := NewNDArray(2, 3)

	// start > stop
	assertPanics(t, func() { a.Slice(All(), Rng(2, 1)) })
	// stop > size
	assertPanics(t, func() { a.Slice(All(), Rng(0, 4)) })
	// start < 0
	assertPanics(t, func() { a.Slice(All(), Rng(-1, 2)) })
}

func TestSlicePanicsOnNonPositiveStep(t *testing.T) {
	a := NewNDArray(2, 3)

	assertPanics(t, func() { a.Slice(All(), Step(0, 3, 0)) })
	assertPanics(t, func() { a.Slice(All(), Step(0, 3, -1)) })
}
