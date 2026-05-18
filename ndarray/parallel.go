package ndarray

import (
	"runtime"
	"sync"
)

// minChunkSize is the smallest amount of work per goroutine that still
// pays off after dispatch overhead. Below this threshold, fall back to
// a serial loop in the calling goroutine.
const minChunkSize = 512

// parallelJob carries a single chunk of work submitted to the
// persistent pool. body runs on disjoint [start, end) ranges; wg
// signals the submitting caller when the job completes.
type parallelJob struct {
	body       func(start, end int)
	wg         *sync.WaitGroup
	start, end int
}

var (
	parallelJobs    chan parallelJob
	parallelWorkers int
)

// init starts a persistent pool of GOMAXPROCS worker goroutines that
// block on parallelJobs for the lifetime of the process. Reusing the
// same workers across every parallelFor call avoids the per-call
// goroutine spawn time.
//
// The pool size is fixed at init time; callers that change
// runtime.GOMAXPROCS later still get correct results but may either
// over-enqueue (extra jobs queue up and run serially in the pool) or
// leave workers idle.
//
// Worker bodies must not themselves call parallelForWork.
func init() {
	parallelWorkers = runtime.GOMAXPROCS(0)
	parallelJobs = make(chan parallelJob, parallelWorkers)

	for range parallelWorkers {
		go func() {
			for job := range parallelJobs {
				job.body(job.start, job.end)
				job.wg.Done()
			}
		}()
	}
}

// parallelForWork splits the index range [0, n) into roughly equal
// chunks across GOMAXPROCS workers and dispatches each chunk to the
// persistent worker pool. body must be safe for concurrent execution
// on disjoint ranges. Blocks until every chunk has returned.
//
// workPerIndex is the approximate amount of work each index of body
// does, in arbitrary units consistent with minChunkSize. The serial
// fallback (body runs once in the caller's goroutine) triggers when
// n*workPerIndex < minChunkSize*workers.
func parallelForWork(n, workPerIndex int, body func(start, end int)) {
	if n <= 0 {
		return
	}

	workers := runtime.GOMAXPROCS(0)
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

		wg.Add(1)
		parallelJobs <- parallelJob{body: body, start: start, end: end, wg: &wg}
	}

	wg.Wait()
}

// parallelFor is parallelForWork with workPerIndex=1, suitable for
// elementwise loops where each index does a constant small amount of
// work.
func parallelFor(n int, body func(start, end int)) {
	parallelForWork(n, 1, body)
}
