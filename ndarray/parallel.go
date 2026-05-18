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

// workerPool owns a fixed set of long-lived goroutines that pull jobs
// from a shared channel. Reusing the same workers across every
// parallelFor call avoids the per-call goroutine spawn cost that
// dominates pprof when many small ops dispatch in a tight loop.
//
// Worker bodies must not themselves call parallelForWork — a nested
// call could deadlock if every pool worker is occupied running the
// outer body. None of ndarray's current callers nest.
type workerPool struct {
	jobs    chan parallelJob
	once    sync.Once
	workers int
}

// ensure lazily spins up the pool on the first parallel-path call.
// Pool size is fixed at the GOMAXPROCS value observed when ensure
// first runs; later runtime.GOMAXPROCS changes are not reflected.
func (p *workerPool) ensure() {
	p.once.Do(func() {
		p.workers = runtime.GOMAXPROCS(0)
		p.jobs = make(chan parallelJob, p.workers)

		for range p.workers {
			go func() {
				for job := range p.jobs {
					job.body(job.start, job.end)
					job.wg.Done()
				}
			}()
		}
	})
}

//nolint:gochecknoglobals // process-wide singleton, accessed via ensure()
var pool workerPool

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

	pool.ensure()

	chunk := (n + workers - 1) / workers

	var wg sync.WaitGroup

	for w := range workers {
		start := w * chunk
		end := min((w+1)*chunk, n)

		if start >= end {
			break
		}

		wg.Add(1)

		pool.jobs <- parallelJob{body: body, start: start, end: end, wg: &wg}
	}

	wg.Wait()
}

// parallelFor is parallelForWork with workPerIndex=1, suitable for
// elementwise loops where each index does a constant small amount of
// work.
func parallelFor(n int, body func(start, end int)) {
	parallelForWork(n, 1, body)
}
