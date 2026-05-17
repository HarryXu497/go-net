package data

import "testing"

// TestSliceDatasetLenAndGet tests the basic round-trip: a dataset
// constructed from a slice reports the right length and returns the
// right element at each index.
func TestSliceDatasetLenAndGet(t *testing.T) {
	dataset := NewSliceDataset([]int{10, 20, 30})

	if got, want := dataset.Len(), 3; got != want {
		t.Errorf("Len() = %d, want %d", got, want)
	}

	tests := []struct {
		name  string
		index int
		want  int
	}{
		{"first", 0, 10},
		{"middle", 1, 20},
		{"last", 2, 30},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := dataset.Get(tc.index); got != tc.want {
				t.Errorf("Get(%d) = %d, want %d", tc.index, got, tc.want)
			}
		})
	}
}

// TestSliceDatasetEmpty tets that a zero-length dataset is a legal
// state. The DataLoader relies on Len() == 0 yielding zero batches
// without panicking.
func TestSliceDatasetEmpty(t *testing.T) {
	dataset := NewSliceDataset([]int{})

	wantLen := 0
	if dataset.Len() != wantLen {
		t.Errorf("empty dataset len = %d, want %d", dataset.Len(), wantLen)
	}
}

// TestSliceDatasetRetainsBackingSlice tests NewSliceDataset's documented
// retention contract: the input slice is borrowed, not copied.
func TestSliceDatasetRetainsBackingSlice(t *testing.T) {
	items := []int{1, 2, 3}
	dataset := NewSliceDataset(items)

	items[0] = 99

	want := 99
	if got := dataset.Get(0); got != want {
		t.Errorf("after mutating backing slice, Get(0) = %d, want %d (slice should be retained, not copied)", got, want)
	}
}
