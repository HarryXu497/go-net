package ndarray

import "testing"

func TestFlatIndex2x3(t *testing.T) {
	a := NewNDArray(2, 3)

	tests := []struct {
		indices []int
		want    int
	}{
		{[]int{0, 0}, 0},
		{[]int{0, 1}, 1},
		{[]int{0, 2}, 2},
		{[]int{1, 0}, 3},
		{[]int{1, 1}, 4},
		{[]int{1, 2}, 5},
	}
	for _, tc := range tests {
		got := a.flatIndex(tc.indices)
		if got != tc.want {
			t.Errorf("flatIndex(%v) = %d, want %d", tc.indices, got, tc.want)
		}
	}
}

func TestFlatIndexScalar(t *testing.T) {
	a := NewNDArray()
	if got := a.flatIndex([]int{}); got != 0 {
		t.Errorf("flatIndex([]) on scalar = %d, want 0", got)
	}
}

func TestFlatIndexHigherRank(t *testing.T) {
	a := NewNDArray(2, 3, 4) // strides [12, 4, 1]
	// element [1, 2, 3] should be at 1*12 + 2*4 + 3*1 = 23
	if got := a.flatIndex([]int{1, 2, 3}); got != 23 {
		t.Errorf("flatIndex([1,2,3]) = %d, want 23", got)
	}
}

func TestFlatIndexRespectsOffset(t *testing.T) {
	a := &NDArray{
		data:    make([]float64, 10),
		shape:   []int{3},
		strides: []int{1},
		offset:  4,
	}
	if got := a.flatIndex([]int{0}); got != 4 {
		t.Errorf("flatIndex([0]) with offset 4 = %d, want 4", got)
	}

	if got := a.flatIndex([]int{2}); got != 6 {
		t.Errorf("flatIndex([2]) with offset 4 = %d, want 6", got)
	}
}

func TestFlatIndexPanicsOnRankMismatch(t *testing.T) {
	a := NewNDArray(2, 3)

	assertPanics(t, func() { a.flatIndex([]int{0}) })
}

func TestFlatIndexPanicsOnOutOfRange(t *testing.T) {
	a := NewNDArray(2, 3)

	assertPanics(t, func() { a.flatIndex([]int{0, 3}) })
}

func TestFlatIndexPanicsOnNegativeIndex(t *testing.T) {
	a := NewNDArray(2, 3)

	assertPanics(t, func() { a.flatIndex([]int{0, -1}) })
}

func TestSetGetRoundtrip(t *testing.T) {
	a := NewNDArray(2, 3)
	a.Set([]int{1, 2}, 3.14)

	if got := a.Get([]int{1, 2}); got != 3.14 {
		t.Errorf("Get after Set = %v, want 3.14", got)
	}
}

func TestSetGetMultiple(t *testing.T) {
	a := NewNDArray(2, 3)
	values := []float64{1, 2, 3, 4, 5, 6}
	idx := 0

	for i := range 2 {
		for j := range 3 {
			a.Set([]int{i, j}, values[idx])
			idx++
		}
	}

	idx = 0

	for i := range 2 {
		for j := range 3 {
			if got := a.Get([]int{i, j}); got != values[idx] {
				t.Errorf("Get([%d,%d]) = %v, want %v", i, j, got, values[idx])
			}

			idx++
		}
	}
}

func TestGetPanicsOnInvalidIndex(t *testing.T) {
	a := NewNDArray(2, 3)

	assertPanics(t, func() { a.Get([]int{5, 5}) })
}

func TestSetPanicsOnInvalidIndex(t *testing.T) {
	a := NewNDArray(2, 3)

	assertPanics(t, func() { a.Set([]int{5, 5}, 0) })
}
