"""Train a tiny classifier end to end through the Triton operators, forward and backward.

The model uses all five operators: matmul (three linear layers), softmax (a gate), add (a
skip connection), residual RMSNorm and row_sum (an L1 penalty on the logits). Weights are
FP32 masters cast to BF16 inside the model, the usual mixed-precision recipe: gradients
flow back to the masters through the cast. The same model built from plain PyTorch
operations trains beside it from the same initial weights and batches, so the two loss
curves can be compared. The custom ops (`torch.ops.kernel_portfolio.*`) are what a
compiled model uses, so `--compile` compiles the whole training step.

    python examples/train_tiny.py
    python examples/train_tiny.py --compile --steps 100
"""

import argparse

import torch
import torch.nn.functional as F

import kernel_portfolio.library  # noqa: F401  (registers torch.ops.kernel_portfolio.*)
from kernel_portfolio import references

DIM, HIDDEN, CLASSES, BATCH = 64, 64, 10, 256
EPS = 1e-5


def make_batches(steps, seed=2026):
    """Points labelled by a fixed random linear teacher: a task the model can learn."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    teacher = torch.randn((DIM, CLASSES), device="cuda", generator=generator)
    for _ in range(steps):
        x = torch.randn((BATCH, DIM), device="cuda", generator=generator)
        yield x.to(torch.bfloat16), (x @ teacher).argmax(-1)


def init_params(seed=2026):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def weight(rows, cols):
        return (
            torch.randn((rows, cols), device="cuda", generator=generator) / rows**0.5
        ).requires_grad_()

    return {
        "w_in": weight(DIM, HIDDEN),
        "w_gate": weight(HIDDEN, HIDDEN),
        "gain": torch.ones(HIDDEN, device="cuda", requires_grad=True),
        "w_out": weight(HIDDEN, CLASSES),
    }


def forward_triton(x, params):
    ops = torch.ops.kernel_portfolio
    w_in, w_gate, gain, w_out = (p.to(torch.bfloat16) for p in params.values())
    stream = ops.matmul(x, w_in)
    gate = ops.softmax(ops.matmul(stream, w_gate))
    mixed = ops.add(stream.reshape(-1), gate.reshape(-1)).view_as(gate)
    normed = ops.residual_rmsnorm(mixed, stream, gain, EPS)
    logits = ops.matmul(normed, w_out)
    return logits, ops.row_sum(logits.abs()).mean()


def forward_torch(x, params):
    w_in, w_gate, gain, w_out = (p.to(torch.bfloat16) for p in params.values())
    stream = x @ w_in
    gate = torch.softmax(stream @ w_gate, -1)
    normed = references.residual_rmsnorm(stream + gate, stream, gain, EPS)
    logits = normed @ w_out
    return logits, logits.abs().sum(-1, dtype=torch.float32).mean()


def train(forward, steps, lr=3e-3, penalty=1e-3, compile_step=False, seed=2026):
    """Adam on the FP32 masters; returns the loss and accuracy after every step."""
    params = init_params(seed)
    optimizer = torch.optim.Adam(params.values(), lr=lr)

    def step(x, labels):
        logits, l1 = forward(x, params)
        loss = F.cross_entropy(logits.float(), labels) + penalty * l1
        loss.backward()
        return loss.detach(), (logits.argmax(-1) == labels).float().mean()

    if compile_step:
        step = torch.compile(step)
    history = []
    for x, labels in make_batches(steps, seed):
        optimizer.zero_grad(set_to_none=True)
        loss, accuracy = step(x, labels)
        optimizer.step()
        history.append((loss.item(), accuracy.item()))
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--compile", action="store_true", help="torch.compile the Triton step")
    args = parser.parse_args()
    triton_history = train(forward_triton, args.steps, compile_step=args.compile)
    torch_history = train(forward_torch, args.steps)
    print("step   loss (Triton)  loss (PyTorch)   accuracy (Triton)  accuracy (PyTorch)")
    for i in sorted({0, 1, 2, 5, 10, 20, 40, 60, 80, 100, args.steps - 1}):
        if i < args.steps:
            (lt, at), (lp, ap) = triton_history[i], torch_history[i]
            print(f"{i:4d}   {lt:13.4f}  {lp:14.4f}   {at:17.3f}  {ap:18.3f}")


if __name__ == "__main__":
    main()
