package ndarray

import (
	"runtime"
	"sync/atomic"
	"testing"
)

// TestParallelForSerialFallback verifies that for n below the
// minChunkSize*workers threshold, parallelFor invokes body exactly
// once with the full range in the caller's goroutine (no spawn).
func TestParallelForSerialFallback(t *testing.T) {
	workers := runtime.GOMAXPROCS(0)
	n := minChunkSize*workers - 1

	var (
		calls            int
		gotStart, gotEnd int
	)

	parallelFor(n, func(start, end int) {
		calls++
		gotStart = start
		gotEnd = end
	})

	if calls != 1 {
		t.Errorf("calls = %d, want 1", calls)
	}

	if gotStart != 0 || gotEnd != n {
		t.Errorf("range = [%d,%d), want [0,%d)", gotStart, gotEnd, n)
	}
}

// TestParallelForSingleWorker verifies that when GOMAXPROCS=1,
// parallelFor takes the serial path even for large n.
func TestParallelForSingleWorker(t *testing.T) {
	prev := runtime.GOMAXPROCS(1)

	t.Cleanup(func() { runtime.GOMAXPROCS(prev) })

	n := minChunkSize * 8

	var (
		calls            int
		gotStart, gotEnd int
	)

	parallelFor(n, func(start, end int) {
		calls++
		gotStart = start
		gotEnd = end
	})

	if calls != 1 {
		t.Errorf("calls = %d, want 1", calls)
	}

	if gotStart != 0 || gotEnd != n {
		t.Errorf("range = [%d,%d), want [0,%d)", gotStart, gotEnd, n)
	}
}

// TestParallelForCoverage verifies that for a large n (well above the
// parallel threshold), every index in [0, n) is visited exactly once
// across all chunks. Each chunk writes 1 into its slot of a shared
// []int32; concurrent writes to disjoint indices are race-free in Go,
// so this test is also meaningful under -race.
func TestParallelForCoverage(t *testing.T) {
	workers := runtime.GOMAXPROCS(0)
	n := minChunkSize * workers * 4

	visits := make([]int32, n)

	parallelFor(n, func(start, end int) {
		for i := start; i < end; i++ {
			atomic.AddInt32(&visits[i], 1)
		}
	})

	for i, v := range visits {
		if v != 1 {
			t.Errorf("visits[%d] = %d, want 1", i, v)
		}
	}
}

// TestParallelForBoundary exercises n exactly at the serial/parallel
// threshold (minChunkSize*workers) and one element on either side, to
// lock down the branch behavior of the cutoff. Every index must be
// visited once regardless of which path is taken.
func TestParallelForBoundary(t *testing.T) {
	workers := runtime.GOMAXPROCS(0)
	threshold := minChunkSize * workers

	for _, n := range []int{threshold - 1, threshold, threshold + 1} {
		t.Run("", func(t *testing.T) {
			visits := make([]int32, n)

			parallelFor(n, func(start, end int) {
				for i := start; i < end; i++ {
					atomic.AddInt32(&visits[i], 1)
				}
			})

			for i, v := range visits {
				if v != 1 {
					t.Errorf("n=%d visits[%d] = %d, want 1", n, i, v)
				}
			}
		})
	}
}

// TestParallelForNonMultiple verifies correctness when n does not
// divide evenly across workers, so the last chunk is shorter than the
// rest and one or more trailing workers may exit on the start>=end
// break.
func TestParallelForNonMultiple(t *testing.T) {
	workers := runtime.GOMAXPROCS(0)
	if workers < 2 {
		t.Skip("needs GOMAXPROCS >= 2 to exercise uneven chunks")
	}

	// A size that ceil-divides into a chunk leaving a short tail.
	n := minChunkSize*workers + 1

	visits := make([]int32, n)

	parallelFor(n, func(start, end int) {
		for i := start; i < end; i++ {
			atomic.AddInt32(&visits[i], 1)
		}
	})

	for i, v := range visits {
		if v != 1 {
			t.Errorf("visits[%d] = %d, want 1", i, v)
		}
	}
}

// TestParallelForConcurrentWrites is the race-detector smoke test:
// many workers each writing real arithmetic into disjoint slots of a
// shared []float64. Pass under `go test -race` means no aliasing bug
// snuck into the partitioning math.
func TestParallelForConcurrentWrites(t *testing.T) {
	n := 1 << 16

	out := make([]float64, n)

	parallelFor(n, func(start, end int) {
		for i := start; i < end; i++ {
			out[i] = float64(i) * 2.5
		}
	})

	for i, v := range out {
		want := float64(i) * 2.5
		if v != want {
			t.Errorf("out[%d] = %g, want %g", i, v, want)
			break
		}
	}
}

// TestParallelForNoopOnZero pins down the current contract: calling
// parallelFor with n <= 0 is a silent no-op (body never invoked). This
// matters because callers may pass out.Size() for empty tensors.
func TestParallelForNoopOnZero(t *testing.T) {
	for _, n := range []int{0, -1, -100} {
		t.Run("", func(t *testing.T) {
			var calls int

			parallelFor(n, func(start, end int) {
				calls++
			})

			if calls != 0 {
				t.Errorf("n=%d: body invoked %d times, want 0", n, calls)
			}
		})
	}
}

// TestParallelForWorkForcesParallel verifies that a small n with a
// large workPerIndex crosses the threshold and runs in parallel — this
// is the matmul/reduction case where each index does a chunk of inner
// work (e.g. an N*K dot product). Coverage must still be exact.
func TestParallelForWorkForcesParallel(t *testing.T) {
	workers := runtime.GOMAXPROCS(0)
	if workers < 2 {
		t.Skip("needs GOMAXPROCS >= 2 to exercise parallel path")
	}

	// n*workPerIndex = 2 * workers * minChunkSize > minChunkSize*workers,
	// so the parallel branch must trigger despite the small n.
	n := workers
	workPerIndex := minChunkSize * 2

	var calls atomic.Int64

	visits := make([]int32, n)

	parallelForWork(n, workPerIndex, func(start, end int) {
		calls.Add(1)

		for i := start; i < end; i++ {
			atomic.AddInt32(&visits[i], 1)
		}
	})

	if got := calls.Load(); got < 2 {
		t.Errorf("calls = %d, want >= 2 (high workPerIndex should force parallel)", got)
	}

	for i, v := range visits {
		if v != 1 {
			t.Errorf("visits[%d] = %d, want 1", i, v)
		}
	}
}

// TestParallelForWorkLowWorkStaysSerial verifies that an n*workPerIndex
// product below the threshold keeps us on the serial path even if n
// itself is moderately large. Locks in that workPerIndex is actually
// consulted, not ignored.
func TestParallelForWorkLowWorkStaysSerial(t *testing.T) {
	n := 1000
	workPerIndex := 1

	var (
		calls            int
		gotStart, gotEnd int
	)

	parallelForWork(n, workPerIndex, func(start, end int) {
		calls++
		gotStart = start
		gotEnd = end
	})

	if calls != 1 {
		t.Errorf("calls = %d, want 1 (low work should stay serial)", calls)
	}

	if gotStart != 0 || gotEnd != n {
		t.Errorf("range = [%d,%d), want [0,%d)", gotStart, gotEnd, n)
	}
}
