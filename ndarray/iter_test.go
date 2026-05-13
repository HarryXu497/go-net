package ndarray

import (
	"slices"
	"testing"
)

func TestIndicesAndOffsetsScalar(t *testing.T) {
	a := NewNDArray()

	var offsets []int
	for _, off := range a.indicesAndOffsets() {
		offsets = append(offsets, off)
	}

	if len(offsets) != 1 || offsets[0] != 0 {
		t.Errorf("scalar should yield once at offset 0, got %v", offsets)
	}
}

func TestIndicesAndOffsetsEmpty(t *testing.T) {
	a := NewNDArray(0, 3)

	count := 0
	for range a.indicesAndOffsets() {
		count++
	}

	if count != 0 {
		t.Errorf("empty tensor should yield nothing, got %d yields", count)
	}
}

func TestIndicesAndOffsets2x3(t *testing.T) {
	a := NewNDArray(2, 3)

	var (
		indicesSeq [][]int
		offsetsSeq []int
	)

	for idx, off := range a.indicesAndOffsets() {
		// indices is reused across iterations — clone if we want to inspect later.
		indicesSeq = append(indicesSeq, slices.Clone(idx))
		offsetsSeq = append(offsetsSeq, off)
	}

	wantIndices := [][]int{
		{0, 0}, {0, 1}, {0, 2},
		{1, 0}, {1, 1}, {1, 2},
	}
	wantOffsets := []int{0, 1, 2, 3, 4, 5}

	if len(indicesSeq) != len(wantIndices) {
		t.Fatalf("got %d yields, want %d", len(indicesSeq), len(wantIndices))
	}

	for i := range wantIndices {
		if !slices.Equal(indicesSeq[i], wantIndices[i]) {
			t.Errorf("yield %d: indices = %v, want %v", i, indicesSeq[i], wantIndices[i])
		}

		if offsetsSeq[i] != wantOffsets[i] {
			t.Errorf("yield %d: offset = %d, want %d", i, offsetsSeq[i], wantOffsets[i])
		}
	}
}

func TestIndicesAndOffsetsTransposedView(t *testing.T) {
	// 2x3 buffer viewed as a 3x2 transpose (swapped strides).
	// Shape-order walk should yield offsets 0, 3, 1, 4, 2, 5.
	a := &NDArray{
		data:    []float64{10, 20, 30, 40, 50, 60},
		shape:   []int{3, 2},
		strides: []int{1, 3},
	}

	var got []int
	for _, off := range a.indicesAndOffsets() {
		got = append(got, off)
	}

	want := []int{0, 3, 1, 4, 2, 5}
	if !slices.Equal(got, want) {
		t.Errorf("transposed view offsets = %v, want %v", got, want)
	}
}

func TestIndicesAndOffsetsBroadcastView(t *testing.T) {
	// Length-3 buffer broadcast to shape [2, 3] via stride 0 on axis 0.
	// Each row should revisit the same 3 elements.
	a := &NDArray{
		data:    []float64{10, 20, 30},
		shape:   []int{2, 3},
		strides: []int{0, 1},
	}

	var got []int
	for _, off := range a.indicesAndOffsets() {
		got = append(got, off)
	}

	want := []int{0, 1, 2, 0, 1, 2}
	if !slices.Equal(got, want) {
		t.Errorf("broadcast view offsets = %v, want %v", got, want)
	}
}

func TestIndicesAndOffsetsRespectsBaseOffset(t *testing.T) {
	// Slice [2:5] of a length-10 buffer: shape [3], stride [1], offset 2.
	a := &NDArray{
		data:    make([]float64, 10),
		shape:   []int{3},
		strides: []int{1},
		offset:  2,
	}

	var got []int
	for _, off := range a.indicesAndOffsets() {
		got = append(got, off)
	}

	want := []int{2, 3, 4}
	if !slices.Equal(got, want) {
		t.Errorf("offsets = %v, want %v", got, want)
	}
}

func TestIndicesAndOffsetsEarlyBreak(t *testing.T) {
	a := NewNDArray(2, 3)

	count := 0
	for range a.indicesAndOffsets() {
		count++
		if count == 2 {
			break
		}
	}

	if count != 2 {
		t.Errorf("early break count = %d, want 2", count)
	}
}

func TestAllYieldsValuesInShapeOrder(t *testing.T) {
	a := FromSlice([]float64{10, 20, 30, 40, 50, 60}, 2, 3)

	var got []float64
	for v := range a.All() {
		got = append(got, v)
	}

	want := []float64{10, 20, 30, 40, 50, 60}
	if !slices.Equal(got, want) {
		t.Errorf("All values = %v, want %v", got, want)
	}
}

func TestAllIndexedClonesIndices(t *testing.T) {
	// Verifies the aliasing fix: stored indices from earlier iterations must
	// not mutate as the iterator advances. If AllIndexed yielded the shared
	// internal slice, every entry of indicesSeq would equal the final [1,2].
	a := FromSlice([]float64{10, 20, 30, 40, 50, 60}, 2, 3)

	var indicesSeq [][]int
	for idx := range a.AllIndexed() {
		indicesSeq = append(indicesSeq, idx)
	}

	wantFirst := []int{0, 0}
	wantLast := []int{1, 2}

	if !slices.Equal(indicesSeq[0], wantFirst) {
		t.Errorf("indicesSeq[0] = %v, want %v (aliasing leaked through)", indicesSeq[0], wantFirst)
	}

	if !slices.Equal(indicesSeq[5], wantLast) {
		t.Errorf("indicesSeq[5] = %v, want %v", indicesSeq[5], wantLast)
	}
}

func TestAllIndexedPairsCorrectValues(t *testing.T) {
	a := FromSlice([]float64{10, 20, 30, 40, 50, 60}, 2, 3)
	for idx, v := range a.AllIndexed() {
		// The value should equal a.Get(idx).
		if got := a.Get(idx); got != v {
			t.Errorf("AllIndexed yielded (%v, %v), but Get(%v) = %v", idx, v, idx, got)
		}
	}
}
