package data

import (
	"reflect"
	"testing"
)

// TestCollate tests that Collate stacks samples into a (batchSize, dim)
// ndarray with rows in the input order, and returns labels in a parallel
// []int.
func TestCollate(t *testing.T) {
	batch := []Sample{
		{X: []float64{1, 2, 3}, Y: 7},
		{X: []float64{4, 5, 6}, Y: 8},
		{X: []float64{7, 8, 9}, Y: 9},
	}

	x, y := Collate(batch)

	wantShape := []int{3, 3}
	if got := x.Shape(); !reflect.DeepEqual(got, wantShape) {
		t.Errorf("Collate x shape = %v, want %v", got, wantShape)
	}

	for i, sample := range batch {
		for j, v := range sample.X {
			if got := x.Get([]int{i, j}); got != v {
				t.Errorf("Collate x[%d,%d] = %v, want %v", i, j, got, v)
			}
		}
	}

	wantY := []int{7, 8, 9}
	if !reflect.DeepEqual(y, wantY) {
		t.Errorf("Collate y = %v, want %v", y, wantY)
	}
}
