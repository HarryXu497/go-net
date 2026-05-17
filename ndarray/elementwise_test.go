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

func TestZeroContiguous(t *testing.T) {
	// Fast path: a freshly built, contiguous, offset-0 tensor owns its
	// backing buffer entirely.
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	a.Zero()

	want := []float64{0, 0, 0, 0, 0, 0}
	if got := allValues(a); !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestZeroScalar(t *testing.T) {
	// Rank-0 tensor: shape (), size 1. Hits the fast path.
	s := NewNDArray()
	s.Set([]int{}, 7)
	s.Zero()

	if got := s.Get([]int{}); got != 0 {
		t.Errorf("scalar = %v, want 0", got)
	}
}

func TestZeroEmpty(t *testing.T) {
	// Zero-size tensor (an axis has length 0): must not panic and has
	// no values to check afterwards.
	a := NewNDArray(0, 3)
	a.Zero()

	if a.Size() != 0 {
		t.Errorf("Size() = %d, want 0", a.Size())
	}
}

func TestZeroTransposedView(t *testing.T) {
	// Slow path: transposed views are non-contiguous, so the iterator
	// branch runs. Aliasing means the original is zeroed too.
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := a.Transpose()
	b.Zero()

	if got := allValues(b); !slices.Equal(got, []float64{0, 0, 0, 0, 0, 0}) {
		t.Errorf("transposed view = %v, want all zeros", got)
	}

	if got := allValues(a); !slices.Equal(got, []float64{0, 0, 0, 0, 0, 0}) {
		t.Errorf("original after zeroing transposed view = %v, want all zeros", got)
	}
}

func TestZeroSlicedViewWithOffset(t *testing.T) {
	// Slow path with offset > 0: a Rng slice starting above 0 advances
	// the offset and produces a non-contiguous view that does not own
	// the entire backing buffer.
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 6)
	mid := a.Slice(Rng(2, 5)) // logical [3,4,5]; offset = 2
	mid.Zero()

	// The slice itself is zeroed.
	if got := allValues(mid); !slices.Equal(got, []float64{0, 0, 0}) {
		t.Errorf("sliced view = %v, want [0 0 0]", got)
	}

	// Elements outside the slice are untouched.
	want := []float64{1, 2, 0, 0, 0, 6}
	if got := allValues(a); !slices.Equal(got, want) {
		t.Errorf("original = %v, want %v", got, want)
	}
}

func TestZeroBroadcastView(t *testing.T) {
	// Slow path with stride 0 on a stretched axis. The iterator writes
	// 0 to the same backing offset multiple times — still correct, and
	// the underlying unbroadcast tensor is zeroed by aliasing.
	a := FromSlice([]float64{1, 2, 3}, 1, 3)
	b := a.BroadcastTo(4, 3)
	b.Zero()

	if got := allValues(a); !slices.Equal(got, []float64{0, 0, 0}) {
		t.Errorf("underlying = %v, want all zeros", got)
	}
}

func TestFillContiguous(t *testing.T) {
	// Fast path with a non-zero value. Zero tests cover v == 0; this
	// exercises the loop branch in Fill.
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	a.Fill(7)

	want := []float64{7, 7, 7, 7, 7, 7}
	if got := allValues(a); !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestFillScalar(t *testing.T) {
	s := NewNDArray()
	s.Fill(-3.5)

	if got := s.Get([]int{}); got != -3.5 {
		t.Errorf("scalar = %v, want -3.5", got)
	}
}

func TestFillEmpty(t *testing.T) {
	// Zero-size tensor: must not panic regardless of the fill value.
	a := NewNDArray(0, 3)
	a.Fill(99)

	if a.Size() != 0 {
		t.Errorf("Size() = %d, want 0", a.Size())
	}
}

func TestFillTransposedView(t *testing.T) {
	// Slow path: a transposed view is non-contiguous. The fill value
	// reaches every logical cell, and aliasing zeros the original too.
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := a.Transpose()
	b.Fill(2)

	want := []float64{2, 2, 2, 2, 2, 2}
	if got := allValues(b); !slices.Equal(got, want) {
		t.Errorf("transposed view = %v, want %v", got, want)
	}

	if got := allValues(a); !slices.Equal(got, want) {
		t.Errorf("original = %v, want %v", got, want)
	}
}

func TestFillSlicedViewWithOffset(t *testing.T) {
	// Slow path with offset > 0: only the sliced region is touched;
	// surrounding elements are unchanged.
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 6)
	mid := a.Slice(Rng(2, 5))
	mid.Fill(-1)

	if got := allValues(mid); !slices.Equal(got, []float64{-1, -1, -1}) {
		t.Errorf("sliced view = %v, want [-1 -1 -1]", got)
	}

	want := []float64{1, 2, -1, -1, -1, 6}
	if got := allValues(a); !slices.Equal(got, want) {
		t.Errorf("original = %v, want %v", got, want)
	}
}

func TestFillBroadcastView(t *testing.T) {
	// Filling a broadcast view writes through to the underlying
	// unbroadcast data via aliasing. The slow path re-writes the same
	// offsets multiple times; result is still correct.
	a := FromSlice([]float64{1, 2, 3}, 1, 3)
	b := a.BroadcastTo(4, 3)
	b.Fill(5)

	if got := allValues(a); !slices.Equal(got, []float64{5, 5, 5}) {
		t.Errorf("underlying = %v, want [5 5 5]", got)
	}
}

func TestFillSpecialValues(t *testing.T) {
	// Fill must accept any float64 — including NaN and ±Inf — without
	// special-casing or panicking.
	cases := []struct {
		name string
		v    float64
	}{
		{"positive infinity", math.Inf(+1)},
		{"negative infinity", math.Inf(-1)},
		{"NaN", math.NaN()},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			a := NewNDArray(2, 2)
			a.Fill(tc.v)

			want := []float64{tc.v, tc.v, tc.v, tc.v}
			if got := allValues(a); !floatsClose(got, want, 0) {
				t.Errorf("values = %v, want %v", got, want)
			}
		})
	}
}

func TestAddInPlaceSameShape(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4}, 2, 2)
	b := FromSlice([]float64{10, 20, 30, 40}, 2, 2)
	a.AddInPlace(b)

	want := []float64{11, 22, 33, 44}
	if got := allValues(a); !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}

	// b must be unchanged.
	if got := allValues(b); !slices.Equal(got, []float64{10, 20, 30, 40}) {
		t.Errorf("b was mutated: %v", got)
	}
}

func TestAddInPlaceBroadcastBias(t *testing.T) {
	// (2, 3) += (3,) — bias added to each row.
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := FromSlice([]float64{10, 20, 30}, 3)
	a.AddInPlace(b)

	want := []float64{11, 22, 33, 14, 25, 36}
	if got := allValues(a); !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}

	// Shape must not change.
	if !slices.Equal(a.Shape(), []int{2, 3}) {
		t.Errorf("shape = %v, want [2 3]", a.Shape())
	}
}

func TestAddInPlaceBroadcastColumn(t *testing.T) {
	// (2, 3) += (2, 1) — column added across columns.
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := FromSlice([]float64{10, 20}, 2, 1)
	a.AddInPlace(b)

	want := []float64{11, 12, 13, 24, 25, 26}
	if got := allValues(a); !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestAddInPlaceBroadcastScalar(t *testing.T) {
	// (2, 3) += scalar — receiver shape preserved.
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	s := NewNDArray()
	s.Set([]int{}, 100)
	a.AddInPlace(s)

	want := []float64{101, 102, 103, 104, 105, 106}
	if got := allValues(a); !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestAddInPlaceAccumulates(t *testing.T) {
	// two AddInPlace calls with the same b must accumulate instead of overwriting.
	a := FromSlice([]float64{0, 0, 0}, 3)
	b := FromSlice([]float64{1, 2, 3}, 3)
	a.AddInPlace(b)
	a.AddInPlace(b)

	want := []float64{2, 4, 6}
	if got := allValues(a); !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestAddInPlaceSelfAdd(t *testing.T) {
	// a.AddInPlace(a) is the only safe self-aliasing case: both iters
	// walk the buffer in the same order, so a becomes 2*a.
	a := FromSlice([]float64{1, 2, 3, 4}, 2, 2)
	a.AddInPlace(a)

	want := []float64{2, 4, 6, 8}
	if got := allValues(a); !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestAddInPlaceNonContiguousReceiver(t *testing.T) {
	// A transposed receiver writes through aliased buffer. The
	// underlying tensor reflects the mutation in transposed layout.
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	view := a.Transpose() // shape (3,2), non-contiguous
	b := FromSlice([]float64{10, 20, 30, 40, 50, 60}, 3, 2)
	view.AddInPlace(b)

	// view logical [[1+10, 4+20],[2+30, 5+40],[3+50, 6+60]]
	wantView := []float64{11, 24, 32, 45, 53, 66}
	if got := allValues(view); !slices.Equal(got, wantView) {
		t.Errorf("view = %v, want %v", got, wantView)
	}

	// Underlying tensor sees the same writes in its own layout:
	// a logical [[11,32,53],[24,45,66]]
	wantOrig := []float64{11, 32, 53, 24, 45, 66}
	if got := allValues(a); !slices.Equal(got, wantOrig) {
		t.Errorf("original = %v, want %v", got, wantOrig)
	}
}

func TestAddInPlaceNonContiguousAddend(t *testing.T) {
	// The addend can be any view; the iterator reads it in logical
	// shape order. Receiver is contiguous, addend is transposed.
	a := FromSlice([]float64{0, 0, 0, 0, 0, 0}, 3, 2)
	b := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3).Transpose() // logical [[1,4],[2,5],[3,6]]
	a.AddInPlace(b)

	want := []float64{1, 4, 2, 5, 3, 6}
	if got := allValues(a); !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestAddInPlacePanicsOnIncompatibleShape(t *testing.T) {
	// Shapes that aren't broadcast-compatible at all.
	a := FromSlice([]float64{1, 2, 3}, 3)
	b := FromSlice([]float64{1, 2, 3, 4}, 4)

	assertPanics(t, func() { a.AddInPlace(b) })
}

func TestAddInPlacePanicsWhenAddendLarger(t *testing.T) {
	// AddInPlace is one-directional: b cannot be larger than a on any
	// axis, because the result must fit into a. (1,3) += (2,3) must
	// panic, even though Add would happily produce a (2,3) result.
	a := FromSlice([]float64{1, 2, 3}, 1, 3)
	b := FromSlice([]float64{10, 20, 30, 40, 50, 60}, 2, 3)

	assertPanics(t, func() { a.AddInPlace(b) })
}

func TestAddInPlaceScalarReceiver(t *testing.T) {
	// Rank-0 receiver and rank-0 addend.
	a := NewNDArray()
	a.Set([]int{}, 5)

	b := NewNDArray()
	b.Set([]int{}, 3)
	a.AddInPlace(b)

	if got := a.Get([]int{}); got != 8 {
		t.Errorf("scalar a = %v, want 8", got)
	}
}

func TestScaleBasic(t *testing.T) {
	a := FromSlice([]float64{1, -2, 3, -4}, 2, 2)
	c := a.Scale(2)

	if !slices.Equal(c.Shape(), []int{2, 2}) {
		t.Errorf("shape = %v, want [2 2]", c.Shape())
	}

	want := []float64{2, -4, 6, -8}
	if got := allValues(c); !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestScaleByZero(t *testing.T) {
	// For finite inputs, c=0 yields all zeros.
	a := FromSlice([]float64{1, 2, 3, 4}, 2, 2)
	c := a.Scale(0)

	want := []float64{0, 0, 0, 0}
	if got := allValues(c); !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestScaleNonContiguousInput(t *testing.T) {
	// Transposed input is non-contiguous. unaryOp walks in shape order
	// and writes to a fresh row-major buffer, so the output is correct
	// regardless of input strides.
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3).Transpose() // logical [[1,4],[2,5],[3,6]]
	c := a.Scale(10)

	if !slices.Equal(c.Shape(), []int{3, 2}) {
		t.Errorf("shape = %v, want [3 2]", c.Shape())
	}

	want := []float64{10, 40, 20, 50, 30, 60}
	if got := allValues(c); !slices.Equal(got, want) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestScaleDoesNotMutateInput(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4}, 2, 2)
	_ = a.Scale(7)

	if got := allValues(a); !slices.Equal(got, []float64{1, 2, 3, 4}) {
		t.Errorf("a was mutated: %v", got)
	}
}

func TestScaleScalar(t *testing.T) {
	// Rank-0 tensor: shape (), size 1.
	s := NewNDArray()
	s.Set([]int{}, 3)
	c := s.Scale(4)

	if got := c.Get([]int{}); got != 12 {
		t.Errorf("scalar = %v, want 12", got)
	}
}

func TestScaleEmpty(t *testing.T) {
	// Zero-size tensor: must not panic and result is also empty.
	a := NewNDArray(0, 3)
	c := a.Scale(5)

	if c.Size() != 0 {
		t.Errorf("Size() = %d, want 0", c.Size())
	}

	if !slices.Equal(c.Shape(), []int{0, 3}) {
		t.Errorf("shape = %v, want [0 3]", c.Shape())
	}
}

func TestScaleSpecialValues(t *testing.T) {
	// 0 * Inf = NaN, c * NaN = NaN, finite * Inf = signed Inf.
	a := FromSlice([]float64{math.Inf(+1), math.Inf(-1), math.NaN(), 2}, 4)

	gotZero := allValues(a.Scale(0))

	wantZero := []float64{math.NaN(), math.NaN(), math.NaN(), 0}
	if !floatsClose(gotZero, wantZero, 0) {
		t.Errorf("Scale(0) on [+Inf,-Inf,NaN,2] = %v, want %v", gotZero, wantZero)
	}

	gotTwo := allValues(a.Scale(2))

	wantTwo := []float64{math.Inf(+1), math.Inf(-1), math.NaN(), 4}
	if !floatsClose(gotTwo, wantTwo, 0) {
		t.Errorf("Scale(2) on [+Inf,-Inf,NaN,2] = %v, want %v", gotTwo, wantTwo)
	}

	// c=NaN propagates regardless of input.
	b := FromSlice([]float64{1, 2, 3}, 3)
	gotNaN := allValues(b.Scale(math.NaN()))

	wantNaN := []float64{math.NaN(), math.NaN(), math.NaN()}
	if !floatsClose(gotNaN, wantNaN, 0) {
		t.Errorf("Scale(NaN) = %v, want %v", gotNaN, wantNaN)
	}
}

// TestAxpyInPlaceUpdatesElements pins the formula a += alpha*x with a
// negative alpha to mimic the SGD use case (a is params, x is grads,
// alpha is -lr).
func TestAxpyInPlaceUpdatesElements(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4}, 2, 2)
	x := FromSlice([]float64{10, 20, 30, 40}, 2, 2)

	a.AxpyInPlace(-0.1, x)

	got := allValues(a)
	want := []float64{0, 0, 0, 0}
	if !floatsClose(got, want, 1e-12) {
		t.Errorf("AxpyInPlace(-0.1, x): got %v, want %v", got, want)
	}
}

// TestAxpyInPlacePanicsOnShapeMismatch: same-shape is a hard
// precondition, and the panic is the only thing standing between a
// caller's bug and silent buffer-overrun corruption.
func TestAxpyInPlacePanicsOnShapeMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic on shape mismatch, got nil")
		}
	}()
	a := FromSlice([]float64{1, 2, 3, 4}, 2, 2)
	x := FromSlice([]float64{1, 2, 3}, 3)
	a.AxpyInPlace(1.0, x)
}
