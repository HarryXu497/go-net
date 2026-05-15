package ndarray

import (
	"slices"
	"testing"
)

const matmulTol = 1e-9

// matmul2DReference computes a @ b for two rank-2 NDArrays with a
// straightforward triple loop and returns the flat row-major result.
// Used to validate Matmul's batched output against a known-simple
// implementation.
func matmul2DReference(a, b *NDArray) []float64 {
	M := a.shape[0]
	K := a.shape[1]
	N := b.shape[1]

	out := make([]float64, M*N)
	for i := range M {
		for j := range N {
			var s float64
			for k := range K {
				s += a.Get([]int{i, k}) * b.Get([]int{k, j})
			}

			out[i*N+j] = s
		}
	}

	return out
}

func TestMatmul2D(t *testing.T) {
	// 2x3 @ 3x2 = 2x2; hand-computed.
	//   [[1,2,3],     [[ 7, 8],     [[ 58,  64],
	//    [4,5,6]]  @   [ 9,10],  =   [139, 154]]
	//                  [11,12]]
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := FromSlice([]float64{7, 8, 9, 10, 11, 12}, 3, 2)
	c := a.Matmul(b)

	if !slices.Equal(c.Shape(), []int{2, 2}) {
		t.Errorf("shape = %v, want [2 2]", c.Shape())
	}

	want := []float64{58, 64, 139, 154}
	if got := allValues(c); !floatsClose(got, want, matmulTol) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestMatmulIdentity(t *testing.T) {
	// A @ I = A for any A.
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	I := FromSlice([]float64{1, 0, 0, 0, 1, 0, 0, 0, 1}, 3, 3)

	got := allValues(a.Matmul(I))

	want := allValues(a)
	if !floatsClose(got, want, matmulTol) {
		t.Errorf("A @ I = %v, want %v", got, want)
	}
}

func TestMatmulThroughTransposeView(t *testing.T) {
	// Matmul must honor strides, not just contiguous buffers.
	// a = [[1,2,3],[4,5,6]] (2x3); a.T is 3x2 with non-contiguous strides.
	// a.T @ a = (3x2) @ (2x3) = 3x3.
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	aT := a.Transpose()

	c := aT.Matmul(a)

	if !slices.Equal(c.Shape(), []int{3, 3}) {
		t.Errorf("shape = %v, want [3 3]", c.Shape())
	}

	// Reference via Copy() of the transpose (contiguous), which must
	// agree with the strided path.
	want := matmul2DReference(aT.Copy(), a)
	if got := allValues(c); !floatsClose(got, want, matmulTol) {
		t.Errorf("values = %v, want %v", got, want)
	}
}

func TestMatmulBatched(t *testing.T) {
	// (2, 3, 4) @ (2, 4, 5) = (2, 3, 5): two independent 3x4 @ 4x5
	// matmuls stacked on the leading axis.
	aData := make([]float64, 2*3*4)
	for i := range aData {
		aData[i] = float64(i + 1)
	}

	bData := make([]float64, 2*4*5)
	for i := range bData {
		bData[i] = float64(i + 1)
	}

	a := FromSlice(aData, 2, 3, 4)
	b := FromSlice(bData, 2, 4, 5)
	c := a.Matmul(b)

	if !slices.Equal(c.Shape(), []int{2, 3, 5}) {
		t.Errorf("shape = %v, want [2 3 5]", c.Shape())
	}

	for batch := range 2 {
		aMat := a.Slice(Idx(batch), All(), All())
		bMat := b.Slice(Idx(batch), All(), All())
		cMat := c.Slice(Idx(batch), All(), All())

		want := matmul2DReference(aMat, bMat)
		if got := allValues(cMat); !floatsClose(got, want, matmulTol) {
			t.Errorf("batch %d: values = %v, want %v", batch, got, want)
		}
	}
}

func TestMatmulBroadcastRightOperand(t *testing.T) {
	// (2, 3, 4) @ (4, 5) = (2, 3, 5): the same right operand is reused
	// across both batch positions of the left.
	aData := make([]float64, 2*3*4)
	for i := range aData {
		aData[i] = float64(i + 1)
	}

	bData := make([]float64, 4*5)
	for i := range bData {
		bData[i] = float64(i + 1)
	}

	a := FromSlice(aData, 2, 3, 4)
	b := FromSlice(bData, 4, 5)
	c := a.Matmul(b)

	if !slices.Equal(c.Shape(), []int{2, 3, 5}) {
		t.Errorf("shape = %v, want [2 3 5]", c.Shape())
	}

	for batch := range 2 {
		aMat := a.Slice(Idx(batch), All(), All())
		cMat := c.Slice(Idx(batch), All(), All())

		want := matmul2DReference(aMat, b)
		if got := allValues(cMat); !floatsClose(got, want, matmulTol) {
			t.Errorf("batch %d: values = %v, want %v", batch, got, want)
		}
	}
}

func TestMatmulBroadcastLeftOperand(t *testing.T) {
	// (3, 4) @ (2, 4, 5) = (2, 3, 5): the same left operand multiplies
	// against both batch positions of the right.
	aData := make([]float64, 3*4)
	for i := range aData {
		aData[i] = float64(i + 1)
	}

	bData := make([]float64, 2*4*5)
	for i := range bData {
		bData[i] = float64(i + 1)
	}

	a := FromSlice(aData, 3, 4)
	b := FromSlice(bData, 2, 4, 5)
	c := a.Matmul(b)

	if !slices.Equal(c.Shape(), []int{2, 3, 5}) {
		t.Errorf("shape = %v, want [2 3 5]", c.Shape())
	}

	for batch := range 2 {
		bMat := b.Slice(Idx(batch), All(), All())
		cMat := c.Slice(Idx(batch), All(), All())

		want := matmul2DReference(a, bMat)
		if got := allValues(cMat); !floatsClose(got, want, matmulTol) {
			t.Errorf("batch %d: values = %v, want %v", batch, got, want)
		}
	}
}

func TestMatmulMultiAxisBatchBroadcast(t *testing.T) {
	// (2, 1, 3, 4) @ (1, 5, 4, 6) = (2, 5, 3, 6): the size-1 axes on
	// each side stretch against the other.
	aData := make([]float64, 2*1*3*4)
	for i := range aData {
		aData[i] = float64(i + 1)
	}

	bData := make([]float64, 1*5*4*6)
	for i := range bData {
		bData[i] = float64(i + 1)
	}

	a := FromSlice(aData, 2, 1, 3, 4)
	b := FromSlice(bData, 1, 5, 4, 6)
	c := a.Matmul(b)

	if !slices.Equal(c.Shape(), []int{2, 5, 3, 6}) {
		t.Errorf("shape = %v, want [2 5 3 6]", c.Shape())
	}

	// For each (i, j) in the broadcast leading shape, the 2-D slice is
	// a[i, 0, :, :] @ b[0, j, :, :].
	for i := range 2 {
		for j := range 5 {
			aMat := a.Slice(Idx(i), Idx(0), All(), All())
			bMat := b.Slice(Idx(0), Idx(j), All(), All())
			cMat := c.Slice(Idx(i), Idx(j), All(), All())

			want := matmul2DReference(aMat, bMat)
			if got := allValues(cMat); !floatsClose(got, want, matmulTol) {
				t.Errorf("batch (%d,%d): values = %v, want %v", i, j, got, want)
			}
		}
	}
}

func TestMatmulContractionMismatchPanics(t *testing.T) {
	a := NewNDArray(2, 3)
	b := NewNDArray(4, 5) // 3 != 4

	defer func() {
		if recover() == nil {
			t.Error("expected panic on contraction mismatch")
		}
	}()

	a.Matmul(b)
}

func TestMatmulBatchBroadcastMismatchPanics(t *testing.T) {
	// Leading dims 3 vs 4 — neither is 1, so broadcasting fails.
	a := NewNDArray(3, 2, 3)
	b := NewNDArray(4, 3, 5)

	defer func() {
		if recover() == nil {
			t.Error("expected panic on batch broadcast mismatch")
		}
	}()

	a.Matmul(b)
}

func TestMatmulLowRankPanics(t *testing.T) {
	cases := []struct {
		a    *NDArray
		b    *NDArray
		name string
	}{
		{FromSlice([]float64{1, 2, 3}, 3), NewNDArray(3, 2), "left rank 1"},
		{NewNDArray(2, 3), FromSlice([]float64{1, 2, 3}, 3), "right rank 1"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Errorf("%s: expected panic", tc.name)
				}
			}()

			tc.a.Matmul(tc.b)
		})
	}
}
