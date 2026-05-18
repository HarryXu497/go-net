package ndarray

import (
	"runtime"
	"sync"
)

// minChunkSize is the smallest amount of work per goroutine that still
// pays off after the overhead spawn cost. Below this threshold, fall back to a
// serial loop in the calling goroutine.
const minChunkSize = 512

// parallelForWork splits the index range [0, n) into roughly equal chunks
// across GOMAXPROCS workers and runs body(start, end) on each chunk
// in its own goroutine. body must be safe for concurrent execution on
// disjoint ranges. Blocks until every chunk has returned.
//
// workPerIndex is the approximate amount of work each index of body does,
// in arbitrary units consistent with minChunkSize. The serial fallback
// triggers when n*workPerIndex < minChunkSize*workers.
func parallelForWork(n, workPerIndex int, body func(start, end int)) {
	if n <= 0 {
		return
	}

	workers := runtime.GOMAXPROCS(0)
	// Not worth spawning goroutines for
	if workers <= 1 || n*workPerIndex < minChunkSize*workers {
		body(0, n)
		return
	}

	chunk := (n + workers - 1) / workers

	var wg sync.WaitGroup

	for w := range workers {
		start := w * chunk
		end := min((w+1)*chunk, n)

		if start >= end {
			break
		}

		wg.Go(func() { body(start, end) })
	}

	wg.Wait()
}

// parallelFor is parallelForWork with workPerIndex=1, suitable for
// elementwise loops where each index does a constant small amount of
// work.
func parallelFor(n int, body func(start, end int)) {
	parallelForWork(n, 1, body)
}
