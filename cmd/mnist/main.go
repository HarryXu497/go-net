package main

import (
	"fmt"
	"log"
	"math/rand"

	"harryxu.ca/gonet/data"
	"harryxu.ca/gonet/ndarray"
	"harryxu.ca/gonet/nn"
	"harryxu.ca/gonet/nn/optim"
	"harryxu.ca/gonet/tensor"
)

const (
	numPixels  = 784
	numClasses = 10
	batchSize  = 64
	numEpochs  = 25
)

func main() {
	rng := rand.New(rand.NewSource(42))

	trainFlat, _, err := LoadImages("mnistdata/train-images-idx3-ubyte")
	if err != nil {
		log.Fatalf("load train images: %v", err)
	}

	trainY, err := LoadLabels("mnistdata/train-labels-idx1-ubyte")
	if err != nil {
		log.Fatalf("load train labels: %v", err)
	}

	testFlat, nTest, err := LoadImages("mnistdata/t10k-images-idx3-ubyte")
	if err != nil {
		log.Fatalf("load test images: %v", err)
	}

	testY, err := LoadLabels("mnistdata/t10k-labels-idx1-ubyte")
	if err != nil {
		log.Fatalf("load test labels: %v", err)
	}

	trainSet := data.NewSliceDataset(BuildSamples(trainFlat, trainY, numPixels))
	loader := data.NewDataLoader(trainSet, batchSize, true, rng)

	testX := ndarray.FromSlice(testFlat, nTest, numPixels)

	model := nn.NewSequential(
		nn.NewLinear(rng, numPixels, 128, nn.He(), nn.Zeros()),
		&nn.ReLU{},
		nn.NewLinear(rng, 128, numClasses, nn.Glorot(), nn.Zeros()),
	)
	optimizer := optim.NewAdamDefault(model.Parameters())

	for epoch := range numEpochs {
		for batch := range loader.Batches() {
			xb, yb := data.Collate(batch)

			logits := model.Forward(tensor.NewTensor(xb))
			loss := nn.CrossEntropyLoss(logits, yb)

			optimizer.ZeroGrad()
			loss.Backward()
			optimizer.Step()
		}

		acc := evaluate(model, testX, testY)
		fmt.Printf("epoch %d, test acc=%.4f\n", epoch, acc)
	}
}

// evaluate runs one forward pass over (x, y) and returns the accuracy
// as the fraction of rows whose argmax over the logits matches the
// label. No autograd: x is wrapped with tensor.NewTensor, not NewLeaf,
// so the eval graph never enters a backward chain.
func evaluate(model nn.Module, x *ndarray.NDArray, y []int) float64 {
	logits := model.Forward(tensor.NewTensor(x))
	out := logits.Data()
	correct := 0

	for i := range y {
		best, bestVal := 0, out.Get([]int{i, 0})
		for j := range numClasses {
			if v := out.Get([]int{i, j}); v > bestVal {
				best, bestVal = j, v
			}
		}

		if best == y[i] {
			correct++
		}
	}

	return float64(correct) / float64(len(y))
}
