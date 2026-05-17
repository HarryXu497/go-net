package nn

import (
	"math/rand"
	"testing"

	"harryxu.ca/gonet/ndarray"
	"harryxu.ca/gonet/tensor"
)

const NumEpochs = 2000

func TestXOR(t *testing.T) {
	rng := rand.New(rand.NewSource(42))

	x := tensor.NewTensor(ndarray.FromSlice([]float64{0, 0, 0, 1, 1, 0, 1, 1}, 4, 2))
	y := []int{0, 1, 1, 0}

	model := NewSequential(
		NewLinear(rng, 2, 4, He(), Zeros()),
		&ReLU{},
		NewLinear(rng, 4, 2, Glorot(), Zeros()),
	)

	optim := NewSGD(model.Parameters(), ConstantLR{Rate: 0.5})

	var lossVal float64
	for range NumEpochs {
		logits := model.Forward(x)
		loss := CrossEntropyLoss(logits, y)

		optim.ZeroGrad()
		loss.Backward()
		optim.Step()

		lossVal = loss.Data().Get([]int{0})
	}

	if lossVal > 0.05 {
		t.Errorf("XOR loss after 2000 steps = %f, want < 0.05", lossVal)
	}
}
