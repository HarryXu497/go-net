package ndarray

import (
	"math"
	"runtime"
	"slices"
	"testing"
)

func TestSumAxis0(t *testing.T) {
	// 2x3 → reduce axis 0 → shape (3,), each output is a column sum.
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	c := a.Sum([]int{0}, false)

	if !slices.Equal(c.Shape(), []int{3}) {
		t.Errorf("shape = %v, want [3]", c.Shape())
	}

	want := []float64{5, 7, 9}
	if got := allValues(c); !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestSumAxis1(t *testing.T) {
	// 2x3 → reduce axis 1 → shape (2,), each output is a row sum.
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	c := a.Sum([]int{1}, false)

	if !slices.Equal(c.Shape(), []int{2}) {
		t.Errorf("shape = %v, want [2]", c.Shape())
	}

	want := []float64{6, 15}
	if got := allValues(c); !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestSumKeepDimsTrue(t *testing.T) {
	// keepDims=true keeps the reduced axis as size 1.
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	c := a.Sum([]int{1}, true)

	if !slices.Equal(c.Shape(), []int{2, 1}) {
		t.Errorf("shape = %v, want [2 1]", c.Shape())
	}

	want := []float64{6, 15}
	if got := allValues(c); !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestSumFullReductionToScalar(t *testing.T) {
	// Spelling out every axis reduces to a rank-0 scalar.
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	c := a.Sum([]int{0, 1}, false)

	if c.Ndim() != 0 {
		t.Errorf("Ndim = %d, want 0", c.Ndim())
	}

	if got := c.Get([]int{}); got != 21 {
		t.Errorf("scalar = %v, want 21", got)
	}
}

func TestSumMultiAxisKeepDimsFalse(t *testing.T) {
	// (2, 3, 4) reduce axes 0 and 2 → shape (3,).
	data := make([]float64, 24)
	for i := range data {
		data[i] = float64(i + 1)
	}

	a := FromSlice(data, 2, 3, 4)
	c := a.Sum([]int{0, 2}, false)

	if !slices.Equal(c.Shape(), []int{3}) {
		t.Errorf("shape = %v, want [3]", c.Shape())
	}

	// Each output cell sums 2*4=8 input cells.
	// out[j] = sum over i,k of a[i,j,k].
	want := []float64{0, 0, 0}

	for i := range 2 {
		for j := range 3 {
			for k := range 4 {
				want[j] += data[i*12+j*4+k]
			}
		}
	}

	if got := allValues(c); !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestSumMultiAxisKeepDimsTrue(t *testing.T) {
	// (2, 3, 4) reduce axes 0 and 2 with keepDims → shape (1, 3, 1).
	a := NewNDArray(2, 3, 4)
	c := a.Sum([]int{0, 2}, true)

	if !slices.Equal(c.Shape(), []int{1, 3, 1}) {
		t.Errorf("shape = %v, want [1 3 1]", c.Shape())
	}
}

func TestSumNoAxesIsIdentity(t *testing.T) {
	// With this codebase's convention, axes=[] reduces nothing.
	a := FromSlice([]float64{1, 2, 3, 4}, 2, 2)
	c := a.Sum([]int{}, false)

	if !slices.Equal(c.Shape(), []int{2, 2}) {
		t.Errorf("shape = %v, want [2 2]", c.Shape())
	}

	if got := allValues(c); !slices.Equal(got, []float64{1, 2, 3, 4}) {
		t.Errorf("values = %v, want input unchanged", got)
	}
}

func TestSumScalarInput(t *testing.T) {
	// Rank-0 input, no axes to reduce → rank-0 output equal to input.
	a := NewNDArray()
	a.Set([]int{}, 42)

	c := a.Sum(nil, false)

	if c.Ndim() != 0 {
		t.Errorf("Ndim = %d, want 0", c.Ndim())
	}

	if got := c.Get([]int{}); got != 42 {
		t.Errorf("scalar = %v, want 42", got)
	}
}

func TestSumNonContiguousInput(t *testing.T) {
	// Transposed (2x3 → 3x2). Logical layout is [[1,4],[2,5],[3,6]].
	// Row sums must be [5, 7, 9].
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3).Transpose()
	c := a.Sum([]int{1}, false)

	if !slices.Equal(c.Shape(), []int{3}) {
		t.Errorf("shape = %v, want [3]", c.Shape())
	}

	want := []float64{5, 7, 9}
	if got := allValues(c); !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestSumOnEmptyTensor(t *testing.T) {
	// shape (0, 3): zero elements, sum along axis 0 → (3,) of zeros.
	a := NewNDArray(0, 3)
	c := a.Sum([]int{0}, false)

	if !slices.Equal(c.Shape(), []int{3}) {
		t.Errorf("shape = %v, want [3]", c.Shape())
	}

	want := []float64{0, 0, 0}
	if got := allValues(c); !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestSumPanicsOnDuplicateAxis(t *testing.T) {
	a := NewNDArray(2, 3)

	assertPanics(t, func() { a.Sum([]int{0, 0}, false) })
}

func TestSumPanicsOnOutOfRangeAxis(t *testing.T) {
	a := NewNDArray(2, 3)

	assertPanics(t, func() { a.Sum([]int{2}, false) })
	assertPanics(t, func() { a.Sum([]int{-1}, false) })
}

func TestSumDoesNotMutateInput(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4}, 2, 2)
	c := a.Sum([]int{0}, false)

	c.Set([]int{0}, 999)

	if got := allValues(a); !slices.Equal(got, []float64{1, 2, 3, 4}) {
		t.Errorf("a was mutated: %v", got)
	}
}

func TestMaxBasic(t *testing.T) {
	// 2x3 → reduce axis 1 → row maxima.
	a := FromSlice([]float64{1, 3, 2, 6, 4, 5}, 2, 3)
	c := a.Max([]int{1}, false)

	if !slices.Equal(c.Shape(), []int{2}) {
		t.Errorf("shape = %v, want [2]", c.Shape())
	}

	want := []float64{3, 6}
	if got := allValues(c); !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestMaxFullReduction(t *testing.T) {
	a := FromSlice([]float64{-10, 5, 3, -2, 100, 7}, 2, 3)
	c := a.Max([]int{0, 1}, false)

	if got := c.Get([]int{}); got != 100 {
		t.Errorf("max = %v, want 100", got)
	}
}

func TestMaxWithNegativeValues(t *testing.T) {
	// All-negative input: confirms the -Inf init is properly beaten.
	a := FromSlice([]float64{-5, -3, -1, -4}, 4)
	c := a.Max([]int{0}, false)

	if got := c.Get([]int{}); got != -1 {
		t.Errorf("max = %v, want -1", got)
	}
}

func TestMaxKeepDimsTrue(t *testing.T) {
	// Softmax-style: per-row max with keepDims so it broadcasts back.
	a := FromSlice([]float64{1, 3, 2, 6, 4, 5}, 2, 3)
	c := a.Max([]int{1}, true)

	if !slices.Equal(c.Shape(), []int{2, 1}) {
		t.Errorf("shape = %v, want [2 1]", c.Shape())
	}

	want := []float64{3, 6}
	if got := allValues(c); !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestMaxPanicsOnEmptyReducedAxis(t *testing.T) {
	// numpy-style: reducing a size-0 axis has no defined max value.
	a := NewNDArray(2, 0, 3)

	assertPanics(t, func() { a.Max([]int{1}, false) })
}

func TestMaxOnSizeZeroOutputIsFine(t *testing.T) {
	// The empty-axis guard fires only for *reduced* axes. Here axis 0
	// is reduced (size 2, non-empty) while a kept axis happens to be
	// size 0. The output is correctly empty; no panic.
	a := NewNDArray(2, 0, 3)
	c := a.Max([]int{0}, false)

	if !slices.Equal(c.Shape(), []int{0, 3}) {
		t.Errorf("shape = %v, want [0 3]", c.Shape())
	}
}

func TestMaxPropagatesNaN(t *testing.T) {
	// math.Max(NaN, x) = NaN; once NaN enters a cell it stays.
	nan := math.NaN()
	a := FromSlice([]float64{1, nan, 2, 3, 4, 5}, 2, 3)
	c := a.Max([]int{1}, false)

	got := allValues(c)
	if !math.IsNaN(got[0]) {
		t.Errorf("row 0 max = %v, want NaN", got[0])
	}

	if got[1] != 5 {
		t.Errorf("row 1 max = %v, want 5", got[1])
	}
}

func TestMaxNonContiguousInput(t *testing.T) {
	// Transposed input, logical layout [[1,4],[2,5],[3,6]].
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3).Transpose()
	c := a.Max([]int{1}, false)

	want := []float64{4, 5, 6}
	if got := allValues(c); !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestMaxPanicsOnDuplicateAxis(t *testing.T) {
	a := NewNDArray(2, 3)

	assertPanics(t, func() { a.Max([]int{1, 1}, false) })
}

func TestMeanBasic(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	c := a.Mean([]int{0}, false)

	if !slices.Equal(c.Shape(), []int{3}) {
		t.Errorf("shape = %v, want [3]", c.Shape())
	}

	// Column means of [[1,2,3],[4,5,6]] are [2.5, 3.5, 4.5].
	want := []float64{2.5, 3.5, 4.5}
	if got := allValues(c); !floatsClose(got, want, 1e-12) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestMeanFullReduction(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	c := a.Mean([]int{0, 1}, false)

	// (1+2+3+4+5+6) / 6 = 3.5
	if got := c.Get([]int{}); math.Abs(got-3.5) > 1e-12 {
		t.Errorf("mean = %v, want 3.5", got)
	}
}

func TestMeanKeepDims(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	c := a.Mean([]int{1}, true)

	if !slices.Equal(c.Shape(), []int{2, 1}) {
		t.Errorf("shape = %v, want [2 1]", c.Shape())
	}

	// Row means are [(1+2+3)/3, (4+5+6)/3] = [2, 5].
	want := []float64{2, 5}
	if got := allValues(c); !floatsClose(got, want, 1e-12) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestMeanNoAxesIsIdentity(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4}, 2, 2)
	c := a.Mean([]int{}, false)

	if got := allValues(c); !slices.Equal(got, []float64{1, 2, 3, 4}) {
		t.Errorf("values = %v, want input unchanged", got)
	}
}

func TestMeanOverEmptyAxisReturnsNaN(t *testing.T) {
	// Documented behavior: dividing by zero count yields NaN.
	a := NewNDArray(2, 0, 3)
	c := a.Mean([]int{1}, false)

	for _, v := range allValues(c) {
		if !math.IsNaN(v) {
			t.Errorf("got %v, want NaN", v)
		}
	}
}

func TestMeanNonContiguousInput(t *testing.T) {
	// Transposed: logical [[1,4],[2,5],[3,6]]. Column means = [2, 5].
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3).Transpose()
	c := a.Mean([]int{0}, false)

	want := []float64{2, 5}
	if got := allValues(c); !floatsClose(got, want, 1e-12) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestReduceCustomFnProduct(t *testing.T) {
	// Exercise the exposed Reduce API with a non-Sum/Max/Mean reducer.
	a := FromSlice([]float64{1, 2, 3, 4}, 2, 2)
	c := a.Reduce([]int{0, 1}, false, 1, func(acc, x float64) float64 { return acc * x })

	if got := c.Get([]int{}); got != 24 {
		t.Errorf("product = %v, want 24", got)
	}
}

func TestSoftmaxStabilityPattern(t *testing.T) {
	// The whole point of Max + Sum with keepDims: subtract row-max,
	// exp, sum, divide. Verify the shape plumbing works end-to-end.
	logits := FromSlice([]float64{1, 2, 3, 1, 2, 3}, 2, 3)

	rowMax := logits.Max([]int{1}, true)
	if !slices.Equal(rowMax.Shape(), []int{2, 1}) {
		t.Fatalf("rowMax shape = %v, want [2 1]", rowMax.Shape())
	}

	shifted := logits.Sub(rowMax)
	expS := shifted.Exp()
	denom := expS.Sum([]int{1}, true)
	probs := expS.Div(denom)

	// Each row of probs must sum to 1.
	rowSums := probs.Sum([]int{1}, false)

	want := []float64{1, 1}
	if got := allValues(rowSums); !floatsClose(got, want, 1e-12) {
		t.Errorf("row sums = %v, want %v", got, want)
	}
}

// TestSumTrailingAxisParallelPath uses a shape large enough that
// outerSize*reducedSize crosses parallelForWork's threshold, so the
// axis-reduction fast path partitions output cells across workers.
// Compared against a per-row reference sum.
func TestSumTrailingAxisParallelPath(t *testing.T) {
	const rows, cols = 256, 1024

	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = float64(i%101) * 0.1
	}

	a := FromSlice(data, rows, cols)
	got := allValues(a.Sum([]int{1}, false))

	want := make([]float64, rows)
	for i := range rows {
		var s float64
		for j := range cols {
			s += data[i*cols+j]
		}

		want[i] = s
	}

	if !floatsClose(got, want, 1e-9) {
		t.Errorf("parallel trailing-axis sum disagrees with reference")
	}
}

// TestMaxTrailingAxisParallelPath exercises the same fast path with a
// non-Sum reducer (different init and fn) to confirm it isn't
// Sum-specific.
func TestMaxTrailingAxisParallelPath(t *testing.T) {
	const rows, cols = 256, 1024

	data := make([]float64, rows*cols)
	for i := range data {
		// Strictly increasing within each row, so the max is the last
		// element. Pattern keeps the reference trivial to compute.
		data[i] = float64(i)
	}

	a := FromSlice(data, rows, cols)
	got := allValues(a.Max([]int{1}, false))

	want := make([]float64, rows)
	for i := range rows {
		want[i] = data[i*cols+cols-1]
	}

	if !floatsClose(got, want, 0) {
		t.Errorf("parallel trailing-axis max disagrees with reference")
	}
}

// TestSumFullReductionParallelPath exercises the full-reduction fast
// path: input is large, output is a single scalar, workers compute
// per-chunk partials merged under a mutex. fp addition is not strictly
// associative so allow a small tolerance.
func TestSumFullReductionParallelPath(t *testing.T) {
	const n = 1 << 20

	data := make([]float64, n)
	for i := range data {
		data[i] = float64(i%101) * 0.1
	}

	a := FromSlice(data, n)
	got := allValues(a.Sum(nil, false))

	var want float64
	for _, v := range data {
		want += v
	}

	if len(got) != 1 {
		t.Fatalf("scalar result expected, got shape %v", a.Sum(nil, false).Shape())
	}

	if math.Abs(got[0]-want) > 1e-6 {
		t.Errorf("full-reduction parallel sum = %v, want %v (diff %v)", got[0], want, got[0]-want)
	}
}

// TestSumLeadingAxisStillSerial: reducing axis 0 of a (rows, cols)
// tensor is NOT a trailing-suffix reduction, so it must fall through
// to the serial path even at sizes that would otherwise parallelize.
// This protects against a future refactor accidentally taking the
// fast path for non-trailing axes (which would compute wrong answers,
// since output cells wouldn't own contiguous slabs).
func TestSumLeadingAxisStillSerial(t *testing.T) {
	const rows, cols = 256, 1024

	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = float64(i%101) * 0.1
	}

	a := FromSlice(data, rows, cols)
	got := allValues(a.Sum([]int{0}, false))

	want := make([]float64, cols)
	for j := range cols {
		var s float64
		for i := range rows {
			s += data[i*cols+j]
		}

		want[j] = s
	}

	if !floatsClose(got, want, 1e-9) {
		t.Errorf("leading-axis sum disagrees with reference")
	}
}

// TestReduceOrderStableMerge proves the full-reduction parallel path
// preserves chunk order: a non-commutative but associative fn produces
// the same result as a serial walk.
//
// The "right-bias" fn (acc, x) -> x is associative (any nested
// composition returns the rightmost argument) but not commutative.
// Reducing [a0, a1, ..., a_{n-1}] with init=0 gives a_{n-1} under
// serial order. The parallel path must agree, which it can only do if
// per-chunk partials are merged left-to-right by chunk index (not in
// goroutine-completion order).
//
// Size is chosen well above minChunkSize*workers so the parallel
// branch fires.
func TestReduceOrderStableMerge(t *testing.T) {
	n := minChunkSize * runtime.GOMAXPROCS(0) * 4

	data := make([]float64, n)
	for i := range data {
		data[i] = float64(i + 1)
	}

	a := FromSlice(data, n)

	rightBias := func(_, x float64) float64 { return x }

	got := a.Reduce(nil, false, 0, rightBias)

	want := data[n-1]
	if v := got.Get([]int{}); v != want {
		t.Errorf("ordered merge produced %v, want %v (last element)", v, want)
	}
}
