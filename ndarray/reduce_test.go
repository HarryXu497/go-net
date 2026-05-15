package ndarray

import (
	"math"
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
