# GoNet

A tiny PyTorch-style deep learning library written from scratch in Go, with reverse-mode autograd over a strided n-dimensional array. Achieves a 97.90% test accuracy on MNIST handwritten digit classification.

I built this to practice Go and to deepen my understanding of neural networks. Like PyTorch, autograd here runs over **tensors**: broadcasting, strided views, and batched matmul all participate in the backward pass directly. Every exported symbol carries a full doc comment, and every package is covered by an extensive unit-test suite.

## Highlights

- **No external dependencies.** Everything written with the Go standard library.
- **Three-layer abstraction**, modeled on PyTorch:
  - `ndarray`: strided, row-major float64 buffers with broadcasting, reductions, matmul, and views (transpose, reshape, slice).
  - `tensor` : nodes wrapping ndarrays; each op records a `backward` closure, and `Backward()` walks a topologically-sorted graph in reverse to accumulate gradients into leaves.
  - `nn`: composable `Module`s: `Linear`, `ReLU`, `Tanh`, `Sigmoid`, `Sequential`, `CrossEntropyLoss`, plus the `SGD` optimizer and weight initializers (`He`, `Glorot`, `Zeros`) and LR schedules (`ConstantLR`, `ExpDecayLR`, `StepDecayLR`).
- **`data` package** with a generic `Dataset[T]` interface, an in-memory `SliceDataset[T]`, a shuffling `DataLoader[T]` that yields batches via Go 1.23 range-over-func iterators, and a `Collate` helper for stacking samples into batched ndarrays.
- **Stable SoftmaxCrossEntropy** — fused forward (subtracts per-row max before `exp` to avoid overflow) and the closed-form `(softmax − one_hot) / batch` gradient.
- **Fully documented and tested.** Every public type, function, and method has a doc comment explaining edge cases and preconditions, and every package ships with a thorough unit-test suite.

## MNIST

`cmd/mnist` trains a 784 → 128 → 10 MLP on the [MNIST](http://yann.lecun.com/exdb/mnist/) handwritten digit dataset:

- He init on the hidden layer (ReLU), Glorot init on the output layer
- ReLU activation
- Softmax + cross-entropy loss
- SGD with a constant learning rate of 0.1
- Batch size 64, 25 epochs

```bash
make run    # train and evaluate on the MNIST test set
make test   # run the full unit-test suite
make lint   # golangci-lint
```

Place the raw MNIST `idx` files under `mnistdata/` before running.

## Layout

```
ndarray/   strided float64 arrays + ops (add, sub, mul, matmul, reductions, broadcasting, views)
tensor/    autograd: forward ops record backward closures; Backward() does a reverse topo walk
nn/        Module interface, Linear, ReLU, Tanh, Sigmoid, Sequential, CrossEntropy, SGD, initializers, LR schedules
data/      Dataset[T], SliceDataset[T], DataLoader[T], Collate
cmd/mnist/ end-to-end MNIST training loop
```

## What's next

- Implement parallelization using **goroutines** to speed up computation.
- Add `Conv2d` and `MaxPool2d` to support **CNNs** for image tasks.

