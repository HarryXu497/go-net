# Adam optimizer — implementation spec

Target file: `nn/optim/adam.go` (with `adam_test.go` alongside).

## 1. Contract

`Adam` implements `optim.Optimizer` (`Step()` and `ZeroGrad()`). It must
behave identically to PyTorch's `torch.optim.Adam` with default
hyperparameters and `amsgrad=False`. Like `SGD`, `Step` bypasses
autograd: it writes directly into each parameter's backing ndarray via
in-place ops so it does not extend the computation graph.

## 2. The math

For parameters `p`, gradients `g = p.Grad()`, and step `t = 1, 2, 3, ...`:

```
m_t = β1 · m_{t-1} + (1 − β1) · g_t
v_t = β2 · v_{t-1} + (1 − β2) · g_t²
m̂_t = m_t / (1 − β1^t)            // bias correction (first moment)
v̂_t = v_t / (1 − β2^t)            // bias correction (second moment)
p   ← p − lr · m̂_t / (√v̂_t + ε)
```

- `m₀ = 0`, `v₀ = 0` (allocated zero-filled the first time `Step` runs,
  or eagerly in `NewAdam`).
- `t` is global to the optimizer, shared across all parameters. It
  starts at 0 in the struct and is incremented to 1 on the first `Step`
  call before the update math runs.
- `g²` is **elementwise** — Hadamard square, not matmul.
- Bias correction matters most early in training (small `t`); both
  `(1 − β1^t)` and `(1 − β2^t)` approach 1 as `t` grows.

## 3. Default hyperparameters

Match PyTorch / the original paper (Kingma & Ba, 2014):

| Symbol | Field name | Default |
|---|---|---|
| `lr` | `LR` | `1e-3` |
| `β1` | `Beta1` | `0.9` |
| `β2` | `Beta2` | `0.999` |
| `ε`  | `Eps`   | `1e-8` |

Constructor takes them explicitly; no globals.

## 4. New ndarray primitives required

Adam needs primitives the current `ndarray` package doesn't have. Add
these first, with tests, before touching the optimizer:

- **`(a *NDArray) Sqrt() *NDArray`** — fresh contiguous tensor of
  `sqrt(a)`. Same shape as `a`. Trivial unary op via `unaryOp(math.Sqrt)`,
  mirrors the existing `Exp` / `Log` / `Tanh`. No tensor-level wrapper is
  needed because Adam never differentiates through it.

- **`(a *NDArray) AddScalar(c float64) *NDArray`** — fresh contiguous
  tensor of `a + c`. Mirrors `Scale` (which is the scalar-Mul form).
  Needed for the `√v̂ + ε` term without allocating a scalar ndarray.

Both should follow the existing style: doc comment with the formula,
"receiver unchanged", non-contiguous-input test, no-mutation test.

### Optional: a fused in-place update

The naive Adam step allocates ~7 temporaries per parameter per batch
(`g²`, scaled `m`, scaled `v`, `m̂`, `v̂`, `√v̂+ε`, `m̂/(√v̂+ε)`). For
MNIST that's fine, but the right end state is a fused

```go
func AdamStepInPlace(p, m, v, g *NDArray, lr, beta1, beta2, eps, biasC1, biasC2 float64)
```

that walks all four buffers in a single loop, no temporaries. Don't
build this first — make the allocating version correct, profile, then
fuse.

## 5. Public API

```go
package optim

import (
    "harryxu.ca/gonet/ndarray"
    "harryxu.ca/gonet/tensor"
)

// Adam ...
type Adam struct {
    params []*tensor.Tensor
    m, v   []*ndarray.NDArray // parallel to params, same shape as each param's data

    lr    float64
    beta1 float64
    beta2 float64
    eps   float64

    step int
}

// NewAdam constructs an Adam optimizer over params with the given
// hyperparameters. The m and v buffers are eagerly allocated zero-filled
// per parameter so Step has no first-call branch.
//
// Typical defaults: lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8.
//
// Panics if any beta is not in [0, 1) or eps <= 0.
func NewAdam(params []*tensor.Tensor, lr, beta1, beta2, eps float64) *Adam
```

Doc comment on `Adam` should mention:
- That `m` and `v` are aligned with `params` by index and that `params`
  must not be mutated after construction.
- That the step counter starts at 0 and increments to 1 on the first
  `Step`, so `t` in the formulas is always ≥ 1.
- That `ZeroGrad` only resets `p.grad`, **not** `m` or `v` — those are
  per-optimizer state and persist across steps by design.

## 6. Step algorithm (allocating version)

```
step++
t = step
biasC1 = 1 − beta1^t
biasC2 = 1 − beta2^t

for i, p in params:
    if p.grad == nil:
        continue                         // unused param this batch
    g = p.grad
    m[i] = beta1·m[i] + (1−beta1)·g       // AddInPlace pattern, allocates
    v[i] = beta2·v[i] + (1−beta2)·g·g
    mHat = m[i] / biasC1                 // Scale(1/biasC1)
    vHat = v[i] / biasC2
    update = mHat / (vHat.Sqrt().AddScalar(eps))
    p.data.AxpyInPlace(-lr, update)
```

Use `math.Pow(beta1, float64(t))` for `beta1^t`. Don't roll your own.

## 7. ZeroGrad

Identical to `SGD.ZeroGrad`: walk `params`, call `p.grad.Zero()` where
`p.grad != nil`. Do **not** touch `m` or `v`.

## 8. Test plan

Mirror the existing `sgd_test.go` patterns. All tests should construct
parameters via `tensor.NewLeaf` and seed `.grad` manually (rather than
running a real backward) so the optimizer behavior is tested in
isolation.

### Numerical correctness

- **`TestAdamFirstStepMatchesClosedForm`** — one parameter, one step, a
  known gradient. Compute the expected post-update value by hand and
  assert with `math.Abs(...) < 1e-12`. This pins the formula end-to-end
  including bias correction.

- **`TestAdamSecondStepUsesMomentum`** — two consecutive steps with the
  same gradient. The second step must move *more* than the first if the
  gradient is consistent in sign (momentum builds). Compute both
  expected values closed-form and compare.

- **`TestAdamBiasCorrectionEarlyVsLate`** — run 1 step and 1000 steps
  with a constant gradient; the per-step delta should approach
  `-lr · sign(g)` as `t → ∞` (since `m̂` → `g` and `√v̂` → `|g|`). Pin
  this loosely with a tolerance — it confirms the bias-correction term
  is wired in the right direction.

### Multi-parameter / shape invariance

- **`TestAdamMultipleParameters`** — two leaves of different shapes
  (e.g. `(3,)` and `(2, 3)`), distinct grads, one step. Both update
  correctly and independently.

- **`TestAdamScalarParameter`** — rank-0 leaf. Sanity check that the
  shape-allocation code in `NewAdam` doesn't choke on empty-shape
  tensors.

### State semantics

- **`TestAdamZeroGradResetsGradsButNotMV`** — seed grads, call `Step`,
  inspect `m` and `v` (via unexported test access or an exported
  test-only helper), call `ZeroGrad`, confirm `m` and `v` are unchanged
  and `p.grad` is now zero.

- **`TestAdamStepWithNilGradSkipsParam`** — two leaves, only one has a
  grad. The other's data, `m`, and `v` are all unchanged.

### Defensive checks

- **`TestNewAdamPanicsOnBadHyperparams`** — table-driven:
  `beta1 = -0.1`, `beta1 = 1`, `beta2 = 1`, `eps = 0`, `eps = -1` all
  panic. `lr = 0` is **allowed** (no-op update is well-defined and
  occasionally useful).

### Integration

- **`TestAdamConvergesOnSimpleQuadratic`** — minimize `f(x) = sum(x²)`
  starting from a non-zero `x`. Run ~500 steps with default
  hyperparameters and assert `|x|` is below some threshold (e.g. 1e-3).
  This is the "does the whole thing work in a real loop" smoke test.

## 9. Pitfalls

- **`m` and `v` must alias `params` by index, not by pointer.** If you
  store them in a `map[*tensor.Tensor]*ndarray.NDArray`, iteration order
  becomes non-deterministic — fine for correctness, bad for debugging.
- **`g²` is `g.Mul(g)`, not `g.Scale(...)`.** Easy slip when typing
  fast.
- **Bias correction divides, not multiplies.** `m̂ = m / (1 − β1^t)`,
  not `m · (1 − β1^t)`.
- **The PyTorch implementation has subtle ordering** (it does
  `step_size = lr / biasC1` and then divides by `√(v/biasC2) + ε`,
  rearranged for stability). Either ordering is mathematically
  equivalent; the form in §6 is fine for fp64.
- **Don't reset `m` and `v` on `ZeroGrad`.** That would defeat the
  point of the optimizer.

## 10. Follow-on optimizers

Once Adam works, momentum SGD, RMSProp, AdaGrad, and AdamW reuse the
same shape: per-parameter state buffers, a `step` counter for any
bias-correction or schedule, a `Step` method that does in-place updates.
AdamW differs from Adam only in one extra `p ← p − lr·λ·p` weight-decay
step before the moment-based update.
