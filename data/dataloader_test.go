package data

import (
	"math/rand"
	"reflect"
	"testing"
)

// TestDataLoaderNoShuffleOrder tests that with shuffle=false, batches are
// yielded in natural index order. With 10 items and batchSize=3 it also
// exercises the short-final-batch case.
func TestDataLoaderNoShuffleOrder(t *testing.T) {
	dataset := NewSliceDataset([]int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
	loader := NewDataLoader(dataset, 3, false, nil)

	var got [][]int
	for batch := range loader.Batches() {
		got = append(got, batch)
	}

	want := [][]int{
		{0, 1, 2},
		{3, 4, 5},
		{6, 7, 8},
		{9},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Batches() = %v, want %v", got, want)
	}
}

// TestDataLoaderShuffleReproducible tests that two runs over the same
// dataset using independently-seeded RNGs with the same seed produce
// identical batch sequences.
func TestDataLoaderShuffleReproducible(t *testing.T) {
	dataset := NewSliceDataset([]int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9})

	runOnce := func() [][]int {
		rng := rand.New(rand.NewSource(42))
		loader := NewDataLoader(dataset, 3, true, rng)

		var batches [][]int
		for batch := range loader.Batches() {
			batches = append(batches, batch)
		}

		return batches
	}

	first := runOnce()

	second := runOnce()
	if !reflect.DeepEqual(first, second) {
		t.Errorf("same seed produced different batches:\n first  = %v\n second = %v", first, second)
	}
}

// TestDataLoaderShuffleIsActuallyDifferent tests that shuffle=true
// actually permutes the indices.
func TestDataLoaderShuffleIsActuallyDifferent(t *testing.T) {
	dataset := NewSliceDataset([]int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
	rng := rand.New(rand.NewSource(42))
	loader := NewDataLoader(dataset, 3, true, rng)

	var got [][]int
	for batch := range loader.Batches() {
		got = append(got, batch)
	}

	natural := [][]int{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9}}
	if reflect.DeepEqual(got, natural) {
		t.Errorf("shuffle=true returned natural order: %v", got)
	}
}

// TestDataLoaderEpochsAreIndependent tests that calling Batches() twice
// on the same loader yields two different orderings.
func TestDataLoaderEpochsAreIndependent(t *testing.T) {
	dataset := NewSliceDataset([]int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
	rng := rand.New(rand.NewSource(42))
	loader := NewDataLoader(dataset, 3, true, rng)

	collect := func() [][]int {
		var batches [][]int
		for batch := range loader.Batches() {
			batches = append(batches, batch)
		}

		return batches
	}

	epoch1 := collect()

	epoch2 := collect()
	if reflect.DeepEqual(epoch1, epoch2) {
		t.Errorf("two consecutive epochs had the same order: %v", epoch1)
	}
}
