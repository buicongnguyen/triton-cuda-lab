"""Imported only when a learner explicitly selects a GPU exercise."""

import importlib.util


def gpu_available():
    for package in ("torch", "triton"):
        if importlib.util.find_spec(package) is None:
            return f"{package} is not installed"
    import torch

    if not torch.cuda.is_available() or torch.version.hip is not None:
        return "an NVIDIA CUDA GPU is required"
    return None


def gpu_cases(name):
    import torch

    def add_case(module, n, dtype):
        torch.manual_seed(2026)
        x = torch.randn(n, device="cuda", dtype=dtype)
        y = torch.randn_like(x)
        saved = (x.clone(), y.clone())
        result = module.add(x, y)
        torch.testing.assert_close(result, x + y, atol=0, rtol=0)
        for actual, expected in zip((x, y), saved):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def tail_guard_case(module, n, block=256):
        # Launch the kernel into the front of a larger buffer. Tail lanes past N must not
        # store, so the sentinel values after the output stay untouched. (A missing load
        # mask reads past the inputs without changing any output; compute-sanitizer's
        # memcheck reports it.)
        import triton

        torch.manual_seed(2026)
        x = torch.randn(n, device="cuda")
        y = torch.randn_like(x)
        buffer = torch.full((n + block,), 7.0, device="cuda")
        module.add_kernel[(triton.cdiv(n, block),)](x, y, buffer, n, block)
        torch.testing.assert_close(buffer[:n], x + y, atol=0, rtol=0)
        if not (buffer[n:] == 7.0).all():
            raise AssertionError("Lanes past N wrote memory: store with mask=offsets < N")

    def row_case(module, rows, width, dtype, strided=False, extreme=False):
        torch.manual_seed(2026)
        x = torch.randn((rows, width + (5 if strided else 0)), device="cuda", dtype=dtype)[
            :, :width
        ]
        if extreme and rows:
            x[0] = -1000
            if rows > 1:
                x[1] = 1000
        saved = x.clone()
        if name == "triton_sum":
            result = module.row_sum(x)
            expected = x.double().sum(-1).float()
            torch.testing.assert_close(result, expected, atol=2e-4, rtol=2e-4)
        else:
            result = module.softmax(x)
            expected = torch.softmax(x.double(), -1).to(dtype)
            torch.testing.assert_close(result, expected, atol=1e-6, rtol=0.002)
            torch.testing.assert_close(
                result.float().sum(-1), torch.ones(rows, device="cuda"), atol=0.002, rtol=0.002
            )
        torch.testing.assert_close(x, saved, atol=0, rtol=0)

    if name == "triton_add":
        return [
            (f"N={n}, {dtype}", lambda m, n=n, dtype=dtype: add_case(m, n, dtype))
            for dtype in (torch.float32, torch.float16)
            for n in (0, 1, 257, 4097, 65537)
        ] + [
            (f"N={n}: no store past the end", lambda m, n=n: tail_guard_case(m, n))
            for n in (1, 257)
        ]
    if name not in ("triton_sum", "triton_softmax"):
        raise ValueError(name)
    cases = [
        (f"shape={shape}, {dtype}", lambda m, shape=shape, dtype=dtype: row_case(m, *shape, dtype))
        for dtype in (torch.float32, torch.float16)
        for shape in ((0, 3), (1, 1), (7, 33), (8, 1024), (3, 4097))
    ]
    cases.append(
        (
            "physical row stride differs from width",
            lambda m: row_case(m, 7, 33, torch.float32, strided=True),
        )
    )
    cases.append(
        (
            "large positive/negative constant rows",
            lambda m: row_case(m, 3, 33, torch.float32, extreme=True),
        )
    )
    return cases
