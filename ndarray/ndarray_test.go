package ndarray

import (
	"slices"
	"testing"
)

// assertPanics is a shared helper for the rest of the test files in this package.
// It fails the test if fn does not panic.
func assertPanics(t *testing.T, fn func()) {
	t.Helper()

	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic, got none")
		}
	}()

	fn()
}

func TestRowMajorStrides(t *testing.T) {
	tests := []struct {
		shape []int
		want  []int
	}{
		{[]int{}, []int{}},
		{[]int{3}, []int{1}},
		{[]int{2, 3}, []int{3, 1}},
		{[]int{2, 3, 4}, []int{12, 4, 1}},
		{[]int{1, 1, 5}, []int{5, 5, 1}},
	}
	for _, tc := range tests {
		got := rowMajorStrides(tc.shape)
		if !slices.Equal(got, tc.want) {
			t.Errorf("rowMajorStrides(%v) = %v, want %v", tc.shape, got, tc.want)
		}
	}
}

func TestNewNDArrayShapesAndSizes(t *testing.T) {
	tests := []struct {
		name     string
		shape    []int
		wantSize int
		wantNdim int
	}{
		{"scalar", []int{}, 1, 0},
		{"vector", []int{5}, 5, 1},
		{"matrix", []int{2, 3}, 6, 2},
		{"three-d", []int{2, 3, 4}, 24, 3},
		{"zero-dim", []int{0, 3}, 0, 2},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := NewNDArray(tc.shape...)
			if a.Size() != tc.wantSize {
				t.Errorf("Size = %d, want %d", a.Size(), tc.wantSize)
			}

			if a.Ndim() != tc.wantNdim {
				t.Errorf("Ndim = %d, want %d", a.Ndim(), tc.wantNdim)
			}

			if !a.IsContiguous() {
				t.Error("fresh NDArray should be contiguous")
			}
		})
	}
}

func TestNewNDArrayZeroInitialized(t *testing.T) {
	a := NewNDArray(2, 3)
	for v := range a.All() {
		if v != 0 {
			t.Errorf("expected zero-initialized, got %v", v)
		}
	}
}

func TestNewNDArrayPanicsOnNegativeDim(t *testing.T) {
	assertPanics(t, func() { NewNDArray(-1, 3) })
}

func TestNewNDArrayShapeAliasingDefense(t *testing.T) {
	shape := []int{2, 3}
	a := NewNDArray(shape...)
	shape[0] = 99

	if a.Shape()[0] != 2 {
		t.Errorf("a.Shape()[0] = %d, want 2 (caller's shape mutation leaked through)", a.Shape()[0])
	}
}

func TestShapeReturnsClone(t *testing.T) {
	a := NewNDArray(2, 3)
	s := a.Shape()
	s[0] = 99

	if a.Shape()[0] != 2 {
		t.Error("Shape() returned an aliased slice, not a clone")
	}
}

func TestStridesReturnsClone(t *testing.T) {
	a := NewNDArray(2, 3)
	s := a.Strides()
	s[0] = 99

	if a.Strides()[0] != 3 {
		t.Error("Strides() returned an aliased slice, not a clone")
	}
}

func TestIsContiguousFresh(t *testing.T) {
	a := NewNDArray(2, 3, 4)
	if !a.IsContiguous() {
		t.Error("fresh NDArray should be contiguous")
	}
}

func TestIsContiguousFalseForTransposedView(t *testing.T) {
	// Hand-construct a non-contiguous view: a 2x3 row-major buffer viewed
	// as 3x2 with swapped strides (the transpose).
	a := &NDArray{
		data:    []float64{1, 2, 3, 4, 5, 6},
		shape:   []int{3, 2},
		strides: []int{1, 3},
	}
	if a.IsContiguous() {
		t.Error("a transposed view should not be contiguous")
	}
}

func TestCopyProducesIndependentBuffer(t *testing.T) {
	a := NewNDArray(2, 3)
	a.Set([]int{0, 0}, 1.5)
	a.Set([]int{1, 2}, 2.5)

	b := a.Copy()

	if b.Get([]int{0, 0}) != 1.5 || b.Get([]int{1, 2}) != 2.5 {
		t.Error("Copy did not preserve values")
	}

	a.Set([]int{0, 0}, 99)

	if b.Get([]int{0, 0}) != 1.5 {
		t.Error("modifying source affected copy")
	}
}

func TestCopyIsContiguous(t *testing.T) {
	a := NewNDArray(2, 3)
	if !a.Copy().IsContiguous() {
		t.Error("Copy should produce a contiguous tensor")
	}
}

func TestCopyScalar(t *testing.T) {
	a := NewNDArray()
	a.Set([]int{}, 42)

	b := a.Copy()
	if b.Get([]int{}) != 42 {
		t.Errorf("scalar copy: got %v, want 42", b.Get([]int{}))
	}
}

func TestCopyOfTransposedViewMaterializesContiguously(t *testing.T) {
	// 2x3 buffer [1,2,3,4,5,6] viewed as 3x2 transpose (strides [1,3]).
	// Logical shape-order read should be: 1,4,2,5,3,6.
	src := &NDArray{
		data:    []float64{1, 2, 3, 4, 5, 6},
		shape:   []int{3, 2},
		strides: []int{1, 3},
	}

	cp := src.Copy()
	if !cp.IsContiguous() {
		t.Error("Copy of non-contiguous view should produce a contiguous tensor")
	}

	want := []float64{1, 4, 2, 5, 3, 6}
	if !slices.Equal(cp.data, want) {
		t.Errorf("copy data = %v, want %v", cp.data, want)
	}
}

func TestFromSliceAliasesData(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	a := FromSlice(data, 2, 3)

	if a.Get([]int{0, 0}) != 1 || a.Get([]int{1, 2}) != 6 {
		t.Error("FromSlice did not preserve values")
	}

	data[0] = 99

	if a.Get([]int{0, 0}) != 99 {
		t.Error("FromSlice should alias data (no copy)")
	}
}

func TestFromSlicePanicsOnLengthMismatch(t *testing.T) {
	assertPanics(t, func() { FromSlice([]float64{1, 2, 3}, 2, 3) })
}

func TestFromSlicePanicsOnNegativeDim(t *testing.T) {
	assertPanics(t, func() { FromSlice([]float64{}, -1, 3) })
}
