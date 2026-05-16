package tensor

import (
	"slices"
	"testing"

	"harryxu.ca/goml/ndarray"
)

// TestUnbroadcast covers the three branches of unbroadcast in isolation:
// the fast path (shapes match), the leading-axis sweep (rank > target),
// the size-1 sweep (same rank, stretched axis), and one combined case
// that exercises both sweeps in sequence.
func TestUnbroadcast(t *testing.T) {
	cases := []struct {
		name      string
		g         *ndarray.NDArray
		target    []int
		wantShape []int
		want      []float64
	}{
		{
			name:      "identity (shapes match)",
			g:         ndarray.FromSlice([]float64{1, 2, 3, 4}, 2, 2),
			target:    []int{2, 2},
			wantShape: []int{2, 2},
			want:      []float64{1, 2, 3, 4},
		},
		{
			name:      "leading axis: (2,3) -> (3)",
			g:         ndarray.FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3),
			target:    []int{3},
			wantShape: []int{3},
			want:      []float64{5, 7, 9},
		},
		{
			name:      "size-1 axis: (2,3) -> (2,1)",
			g:         ndarray.FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3),
			target:    []int{2, 1},
			wantShape: []int{2, 1},
			want:      []float64{6, 15},
		},
		{
			name:      "both sweeps: (2,3,2) -> (3,1)",
			g:         ndarray.FromSlice([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 2, 3, 2),
			target:    []int{3, 1},
			wantShape: []int{3, 1},
			want:      []float64{18, 26, 34},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := unbroadcast(tc.g, tc.target)
			if !slices.Equal(got.Shape(), tc.wantShape) {
				t.Fatalf("shape: got %v, want %v", got.Shape(), tc.wantShape)
			}
			gotValues := slices.Collect(got.All())
			if !slices.Equal(gotValues, tc.want) {
				t.Errorf("values: got %v, want %v", gotValues, tc.want)
			}
		})
	}
}
