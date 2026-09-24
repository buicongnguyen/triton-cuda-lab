"""Independent oracles and contract checks for the five GPU workshops."""

import math

import torch


def tolerance(name, dtype):
    if name in ("strided_softmax", "split_softmax"):
        return {
            "atol": 2e-7 if dtype == torch.float32 else 2e-5,
            "rtol": 3e-5 if dtype == torch.float32 else 0.01,
        }
    if name == "softmax_backward":
        return {"atol": 2e-7 if dtype == torch.float32 else 0.0003, "rtol": 0.01}
    if name == "fused_gemm":
        return {"atol": 0.06 if dtype == torch.bfloat16 else 0.015, "rtol": 0.02}
    return {
        "atol": 0.015 if dtype == torch.bfloat16 else 0.002,
        "rtol": 0.02 if dtype == torch.bfloat16 else 0.005,
    }


def attention_oracle(q, k, v, causal):
    scores = q.double() @ k.double().T / math.sqrt(q.shape[1])
    if causal:
        scores.masked_fill_(torch.ones_like(scores, dtype=torch.bool).triu(1), -math.inf)
    return (scores.softmax(-1) @ v.double()).to(q.dtype)


def verify(call, inputs, expected, name):
    saved = [x.clone() for x in inputs]
    actual = call()
    torch.testing.assert_close(actual, expected, **tolerance(name, expected.dtype))
    if not actual.is_contiguous():
        raise AssertionError("Output must be contiguous")
    for value, before in zip(inputs, saved):
        torch.testing.assert_close(value, before, atol=0, rtol=0)
        if value.numel() and actual.numel() and actual.data_ptr() == value.data_ptr():
            raise AssertionError("Output must not alias an input")
    return actual


def expect_error(call):
    try:
        call()
    except (ValueError, TypeError):
        return
    raise AssertionError("Expected ValueError or TypeError")


def gpu_cases(name):
    torch.manual_seed(2026)

    def rand(shape, dtype=torch.float32):
        return torch.randn(shape, device="cuda", dtype=dtype)

    def softmax_case(module, rows, n, dtype, layout="contiguous", knob=None, extreme=False):
        if layout == "transpose":
            x = rand((n, rows), dtype).T
        elif layout == "slice":
            x = rand((rows + 1, 2 * n + 3), dtype)[1:, 1 : 2 * n + 1 : 2]
        elif layout == "broadcast":
            x = rand((1, n), dtype).expand(rows, n)
        else:
            x = rand((rows, n), dtype)
        if extreme and rows:
            x[0] = 1000
            if rows > 1:
                x[1] = -1000
            if rows > 2:
                x[2] = -1000
                x[2, -1] = 1000
        options = {"num_warps": knob} if name == "strided_softmax" else {"chunk_size": knob}
        options = options if knob is not None else {}
        result = verify(
            lambda: module.softmax(x, **options), [x], x.double().softmax(-1).to(dtype), name
        )
        torch.testing.assert_close(
            result.float().sum(-1),
            torch.ones(rows, device="cuda"),
            atol=0.005 if dtype == torch.bfloat16 else 0.001,
            rtol=0,
        )

    def backward_case(module, rows, n, dtype, strided=False):
        y = rand((rows, n), dtype).float().softmax(-1).to(dtype)
        g = rand((n, rows), dtype).T if strided else rand((rows, n), dtype)
        # Contract: VJP from the supplied (possibly rounded) probabilities.
        expected = y.double() * (g.double() - (g.double() * y.double()).sum(-1, keepdim=True))
        verify(lambda: module.backward(y, g), [y, g], expected.to(dtype), name)

    def autograd_case(module):
        x = rand((3, 17)).double().requires_grad_()
        g = rand(x.shape).double()
        y = x.softmax(-1)
        (expected,) = torch.autograd.grad((y * g).sum(), x)
        result = module.backward(y.detach().float(), g.float())
        torch.testing.assert_close(result.double(), expected, atol=2e-7, rtol=2e-5)
        torch.testing.assert_close(result.sum(-1), torch.zeros(3, device="cuda"), atol=3e-7, rtol=0)
        # Central difference of L=sum(softmax(x)*g), independent of the VJP formula.
        step = 1e-5
        plus, minus = x.detach().clone(), x.detach().clone()
        plus[1, 7] += step
        minus[1, 7] -= step
        finite_difference = ((plus.softmax(-1) * g).sum() - (minus.softmax(-1) * g).sum()) / (
            2 * step
        )
        torch.testing.assert_close(result[1, 7].double(), finite_difference, atol=2e-7, rtol=2e-5)

    def both_strided_backward(module):
        y = rand((33, 7)).T.softmax(-1).T.contiguous().T
        g = rand((7, 69))[:, 1:67:2]
        expected = y.double() * (g.double() - (y.double() * g.double()).sum(-1, keepdim=True))
        verify(lambda: module.backward(y, g, num_warps=8), [y, g], expected.float(), name)

    def gemm_case(module, shape, dtype, group=4, tile=32):
        m, n, k = shape
        a, b, bias = rand((m, k), dtype), rand((k, n), dtype), rand((n,), dtype)
        expected = (a.double() @ b.double() + bias.double()).relu().to(dtype)
        verify(
            lambda: module.matmul_bias_relu(a, b, bias, group_m=group, tile_m=tile),
            [a, b, bias],
            expected,
            name,
        )

    def attention_case(module, n, d, dtype, causal, block=32, uniform=False, extreme=False):
        q, k, v = (rand((n, d), dtype) for _ in range(3))
        if uniform:
            q.zero_()
        if extreme:
            q *= 4
            k *= 4
        expected = attention_oracle(q, k, v, causal)
        result = verify(
            lambda: module.attention(q, k, v, causal=causal, block_n=block),
            [q, k, v],
            expected,
            name,
        )
        if causal and n > 1:
            changed = v.clone()
            changed[n // 2 :] += 10
            after = module.attention(q, k, changed, causal=True, block_n=block)
            torch.testing.assert_close(after[: n // 2], result[: n // 2], atol=0, rtol=0)

    cases = []

    def add(label, fn):
        cases.append((label, fn))

    if name in ("strided_softmax", "split_softmax"):
        shapes = (
            [(0, 3), (1, 1), (7, 33), (3, 8192)]
            if name == "strided_softmax"
            else [(0, 3), (1, 1), (3, 1025), (2, 32769), (1, 131072)]
        )
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for rows, n in shapes:
                add(f"{rows}x{n} {dtype}", lambda m, r=rows, n=n, d=dtype: softmax_case(m, r, n, d))
        add(
            "large constants and isolated maximum",
            lambda m: softmax_case(m, 3, 4097, torch.float32, extreme=True),
        )
        if name == "strided_softmax":
            for layout in ("transpose", "slice", "broadcast"):
                add(
                    f"{layout} physical addressing",
                    lambda m, layout=layout: softmax_case(m, 7, 33, torch.float32, layout, 8),
                )
            add("reject zero width", lambda m: expect_error(lambda: m.softmax(rand((3, 0)))))
            add(
                "reject bad warps",
                lambda m: expect_error(lambda: m.softmax(rand((3, 3)), num_warps=3)),
            )
            add(
                "reject oversized width", lambda m: expect_error(lambda: m.softmax(rand((1, 8193))))
            )
        else:
            for chunk in (256, 4096):
                add(
                    f"chunk={chunk} has partial tail",
                    lambda m, c=chunk: softmax_case(
                        m, 3, 8193, torch.float32, knob=c, extreme=True
                    ),
                )
            add(
                "reject bad chunk",
                lambda m: expect_error(lambda: m.softmax(rand((1, 3)), chunk_size=511)),
            )
            add(
                "reject noncontiguous input",
                lambda m: expect_error(lambda: m.softmax(rand((3, 7)).T)),
            )
            add(
                "reject oversized width",
                lambda m: expect_error(lambda: m.softmax(rand((1, 131073)))),
            )
    elif name == "softmax_backward":
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for rows, n in ((0, 3), (1, 1), (7, 33), (3, 8192)):
                add(
                    f"{rows}x{n} {dtype}", lambda m, r=rows, n=n, d=dtype: backward_case(m, r, n, d)
                )
        add("strided upstream gradient", lambda m: backward_case(m, 7, 33, torch.float32, True))
        add("both operands strided and eight warps", both_strided_backward)
        add("autograd, finite difference and row-sum invariant", autograd_case)
        add(
            "reject shape mismatch",
            lambda m: expect_error(lambda: m.backward(rand((1, 3)), rand((2, 3)))),
        )
        add(
            "reject dtype mismatch",
            lambda m: expect_error(lambda: m.backward(rand((1, 3)), rand((1, 3), torch.float16))),
        )
    elif name == "fused_gemm":
        for dtype in (torch.float16, torch.bfloat16):
            for shape in ((0, 7, 3), (3, 0, 7), (3, 5, 0), (1, 1, 1), (33, 65, 17), (257, 65, 33)):
                add(f"M,N,K={shape} {dtype}", lambda m, s=shape, d=dtype: gemm_case(m, s, d))
        for group in (1, 4, 8):
            for tile in (16, 32, 64):
                add(
                    f"group={group}, BM={tile}, incomplete group",
                    lambda m, g=group, t=tile: gemm_case(m, (257, 65, 33), torch.float16, g, t),
                )
        add(
            "reject FP32 GEMM",
            lambda m: expect_error(
                lambda: m.matmul_bias_relu(rand((2, 3)), rand((3, 4)), rand((4,)))
            ),
        )
        add(
            "reject bias length",
            lambda m: expect_error(
                lambda: m.matmul_bias_relu(
                    rand((2, 3), torch.float16),
                    rand((3, 4), torch.float16),
                    rand((3,), torch.float16),
                )
            ),
        )
    elif name == "streaming_attention":
        for dtype in (torch.float16, torch.bfloat16):
            for causal in (False, True):
                for n, d in ((0, 16), (1, 32), (17, 16), (65, 32), (129, 64)):
                    add(
                        f"N={n},D={d},causal={causal},{dtype}",
                        lambda m, n=n, d=d, c=causal, dt=dtype: attention_case(m, n, d, dt, c),
                    )
        add(
            "N=2048 maximum supported sequence",
            lambda m: attention_case(m, 2048, 64, torch.float16, True, 64),
        )
        add(
            "uniform causal = prefix average",
            lambda m: attention_case(m, 65, 32, torch.float16, True, 64, uniform=True),
        )
        add(
            "large logits, running maximum changes",
            lambda m: attention_case(m, 65, 32, torch.float16, False, 64, extreme=True),
        )
        add(
            "reject unsupported D",
            lambda m: expect_error(
                lambda: m.attention(*[rand((3, 24), torch.float16) for _ in range(3)])
            ),
        )
        add(
            "reject unequal lengths",
            lambda m: expect_error(
                lambda: m.attention(
                    rand((3, 32), torch.float16),
                    rand((4, 32), torch.float16),
                    rand((4, 32), torch.float16),
                )
            ),
        )
    else:
        raise ValueError(name)

    def on_stream(module):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            if name in ("strided_softmax", "split_softmax"):
                softmax_case(module, 3, 1025, torch.float32)
            elif name == "softmax_backward":
                backward_case(module, 3, 33, torch.float32)
            elif name == "fused_gemm":
                gemm_case(module, (33, 65, 17), torch.float16)
            else:
                attention_case(module, 33, 32, torch.float16, True)
        stream.synchronize()

    add("non-default CUDA stream", on_stream)

    def reject_common(module, cpu=False, requires_grad=False):
        dtype = torch.float16 if name in ("fused_gemm", "streaming_attention") else torch.float32
        width = 16 if name == "streaming_attention" else 3
        args = [rand((3, width), dtype)]
        if name == "softmax_backward":
            args.append(rand((3, width), dtype))
            fn = module.backward
        elif name == "fused_gemm":
            args += [rand((width, 5), dtype), rand((5,), dtype)]
            fn = module.matmul_bias_relu
        elif name == "streaming_attention":
            args += [rand((3, width), dtype), rand((3, width), dtype)]
            fn = module.attention
        else:
            fn = module.softmax
        if cpu:
            args = [x.cpu() for x in args]
        if requires_grad:
            args[0].requires_grad_()
        expect_error(lambda: fn(*args))

    add("reject CPU tensors", lambda m: reject_common(m, cpu=True))
    add("reject requires_grad tensors", lambda m: reject_common(m, requires_grad=True))
    return cases
