package ndarray

import (
	"math"
	"slices"
	"testing"
)

const elementwiseTol = 1e-9

// floatsClose compares two float64 slices with a per-call tolerance,
// treating Inf and NaN explicitly: NaN matches NaN, Inf must match sign
// exactly, and finite values are compared with abs tolerance.
func floatsClose(got, want []float64, tol float64) bool {
	if len(got) != len(want) {
		return false
	}

	for i := range got {
		switch {
		case math.IsNaN(got[i]) || math.IsNaN(want[i]):
			if math.IsNaN(got[i]) != math.IsNaN(want[i]) {
				return false
			}
		case math.IsInf(got[i], 0) || math.IsInf(want[i], 0):
			if got[i] != want[i] {
				return false
			}
		default:
			if math.Abs(got[i]-want[i]) > tol {
				return false
			}
		}
	}

	return true
}

// allValues collects every value of a in shape order.
func allValues(a *NDArray) []float64 {
	var got []float64

	for v := range a.All() {
		got = append(got, v)
	}

	return got
}

func TestAddSameShape(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4}, 2, 2)
	b := FromSlice([]float64{10, 20, 30, 40}, 2, 2)
	c := a.Add(b)

	if !slices.Equal(c.Shape(), []int{2, 2}) {
		t.Errorf("shape = %v, want [2 2]", c.Shape())
	}

	want := []float64{11, 22, 33, 44}
	if got := allValues(c); !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestAddBroadcastBias(t *testing.T) {
	// (2, 3) + (3,) — bias added to each row.
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := FromSlice([]float64{10, 20, 30}, 3)
	c := a.Add(b)

	if !slices.Equal(c.Shape(), []int{2, 3}) {
		t.Errorf("shape = %v, want [2 3]", c.Shape())
	}

	want := []float64{11, 22, 33, 14, 25, 36}
	if got := allValues(c); !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestAddBroadcastColumn(t *testing.T) {
	// (2, 3) + (2, 1) — column added across columns.
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := FromSlice([]float64{10, 20}, 2, 1)
	c := a.Add(b)

	want := []float64{11, 12, 13, 24, 25, 26}
	if got := allValues(c); !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestAddBroadcastScalar(t *testing.T) {
	// rank-0 + (2, 3): scalar added to every cell.
	s := NewNDArray()
	s.Set([]int{}, 100)

	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	c := a.Add(s)

	want := []float64{101, 102, 103, 104, 105, 106}
	if got := allValues(c); !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestAddSelf(t *testing.T) {
	// a.Add(a) must work and must not corrupt a.
	a := FromSlice([]float64{1, 2, 3, 4}, 2, 2)
	c := a.Add(a)

	want := []float64{2, 4, 6, 8}
	if got := allValues(c); !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}

	if got := allValues(a); !slices.Equal(got, []float64{1, 2, 3, 4}) {
		t.Errorf("a was mutated: %v", got)
	}
}

func TestAddDoesNotMutateInputs(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4}, 2, 2)
	b := FromSlice([]float64{10, 20, 30, 40}, 2, 2)
	c := a.Add(b)

	c.Set([]int{0, 0}, 999)

	if got := a.Get([]int{0, 0}); got != 1 {
		t.Errorf("a[0,0] = %v, want 1 (input mutated)", got)
	}

	if got := b.Get([]int{0, 0}); got != 10 {
		t.Errorf("b[0,0] = %v, want 10 (input mutated)", got)
	}
}

func TestAddPanicsOnIncompatibleShape(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3}, 3)
	b := FromSlice([]float64{1, 2, 3, 4}, 4)

	assertPanics(t, func() { a.Add(b) })
}

func TestAddNonContiguousInput(t *testing.T) {
	// Transpose makes the input non-contiguous. binaryOp must still
	// read values in logical shape order.
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3).Transpose() // logical [[1,4],[2,5],[3,6]]
	b := NewNDArray(3, 2)                                         // zeros
	c := a.Add(b)

	want := []float64{1, 4, 2, 5, 3, 6}
	if got := allValues(c); !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestSubBasic(t *testing.T) {
	a := FromSlice([]float64{10, 20, 30}, 3)
	b := FromSlice([]float64{1, 2, 3}, 3)
	c := a.Sub(b)

	want := []float64{9, 18, 27}
	if got := allValues(c); !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestSubNonCommutative(t *testing.T) {
	a := FromSlice([]float64{10, 20}, 2)
	b := FromSlice([]float64{1, 2}, 2)

	ab := allValues(a.Sub(b))
	ba := allValues(b.Sub(a))

	if !slices.Equal(ab, []float64{9, 18}) {
		t.Errorf("a-b = %v, want [9 18]", ab)
	}

	if !slices.Equal(ba, []float64{-9, -18}) {
		t.Errorf("b-a = %v, want [-9 -18]", ba)
	}
}

func TestMulHadamard(t *testing.T) {
	// Same-shape Mul is the Hadamard (elementwise) product.
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	c := a.Mul(b)

	if !slices.Equal(c.Shape(), []int{2, 3}) {
		t.Errorf("shape = %v, want [2 3]", c.Shape())
	}

	want := []float64{1, 4, 9, 16, 25, 36}
	if got := allValues(c); !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestMulBroadcast(t *testing.T) {
	// (2, 3) * (3,) — scales each column by a different factor.
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := FromSlice([]float64{10, 100, 1000}, 3)
	c := a.Mul(b)

	want := []float64{10, 200, 3000, 40, 500, 6000}
	if got := allValues(c); !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestDivBasic(t *testing.T) {
	a := FromSlice([]float64{10, 20, 30}, 3)
	b := FromSlice([]float64{2, 4, 5}, 3)
	c := a.Div(b)

	want := []float64{5, 5, 6}
	if got := allValues(c); !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestDivByZeroFollowsIEEE(t *testing.T) {
	// Documented behavior: nonzero/0 → ±Inf, 0/0 → NaN. No panic.
	a := FromSlice([]float64{1, -1, 0}, 3)
	b := FromSlice([]float64{0, 0, 0}, 3)
	c := a.Div(b)

	got := allValues(c)

	if !math.IsInf(got[0], +1) {
		t.Errorf("1/0 = %v, want +Inf", got[0])
	}

	if !math.IsInf(got[1], -1) {
		t.Errorf("-1/0 = %v, want -Inf", got[1])
	}

	if !math.IsNaN(got[2]) {
		t.Errorf("0/0 = %v, want NaN", got[2])
	}
}

func TestNegBasic(t *testing.T) {
	a := FromSlice([]float64{1, -2, 0, 3.5}, 4)
	c := a.Neg()

	want := []float64{-1, 2, 0, -3.5}
	if got := allValues(c); !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestNegNonContiguousInput(t *testing.T) {
	// unaryOp must walk a in shape order and write
	// out in fresh-buffer order. A transposed input is the simplest
	// non-contiguous case.
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3).Transpose() // logical [[1,4],[2,5],[3,6]]
	c := a.Neg()

	if !slices.Equal(c.Shape(), []int{3, 2}) {
		t.Errorf("shape = %v, want [3 2]", c.Shape())
	}

	want := []float64{-1, -4, -2, -5, -3, -6}
	if got := allValues(c); !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestNegDoesNotMutateInput(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4}, 2, 2)
	_ = a.Neg()

	if got := allValues(a); !slices.Equal(got, []float64{1, 2, 3, 4}) {
		t.Errorf("a was mutated: %v", got)
	}
}

func TestExpBasic(t *testing.T) {
	a := FromSlice([]float64{0, 1, 2}, 3)
	c := a.Exp()

	want := []float64{1, math.E, math.E * math.E}
	if got := allValues(c); !floatsClose(got, want, elementwiseTol) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestExpNegativeInputs(t *testing.T) {
	// exp(-x) = 1 / exp(x).
	a := FromSlice([]float64{-1, -2}, 2)
	c := a.Exp()

	want := []float64{1 / math.E, 1 / (math.E * math.E)}
	if got := allValues(c); !floatsClose(got, want, elementwiseTol) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestLogBasic(t *testing.T) {
	a := FromSlice([]float64{1, math.E, math.E * math.E}, 3)
	c := a.Log()

	want := []float64{0, 1, 2}
	if got := allValues(c); !floatsClose(got, want, elementwiseTol) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestLogZeroAndNegativeFollowIEEE(t *testing.T) {
	// log(0) = -Inf, log(-1) = NaN. No panic.
	a := FromSlice([]float64{0, -1}, 2)
	c := a.Log()

	got := allValues(c)

	if !math.IsInf(got[0], -1) {
		t.Errorf("log(0) = %v, want -Inf", got[0])
	}

	if !math.IsNaN(got[1]) {
		t.Errorf("log(-1) = %v, want NaN", got[1])
	}
}

func TestExpLogRoundTrip(t *testing.T) {
	// log(exp(x)) ≈ x for finite x.
	a := FromSlice([]float64{-2, -0.5, 0, 0.5, 2}, 5)
	c := a.Exp().Log()

	if got := allValues(c); !floatsClose(got, allValues(a), elementwiseTol) {
		t.Errorf("log(exp(a)) = %v, want %v", got, allValues(a))
	}
}

func TestReLUBasic(t *testing.T) {
	a := FromSlice([]float64{-2, -1, 0, 1, 2}, 5)
	c := a.ReLU()

	want := []float64{0, 0, 0, 1, 2}
	if got := allValues(c); !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestReLUNonContiguousInput(t *testing.T) {
	// Transposed input: logical layout is [[-1,4],[2,-5],[-3,6]].
	a := FromSlice([]float64{-1, 2, -3, 4, -5, 6}, 2, 3).Transpose()
	c := a.ReLU()

	if !slices.Equal(c.Shape(), []int{3, 2}) {
		t.Errorf("shape = %v, want [3 2]", c.Shape())
	}

	want := []float64{0, 4, 2, 0, 0, 6}
	if got := allValues(c); !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}
