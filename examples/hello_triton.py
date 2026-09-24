"""First run: compare one small GPU kernel against PyTorch."""

import torch

from kernel_portfolio import add


def main():
    torch.manual_seed(2026)
    # 257 elements deliberately leaves a one-element tail at BLOCK=256.
    x = torch.randn(257, device="cuda")
    y = torch.randn_like(x)
    result = add(x, y)
    torch.testing.assert_close(result, x + y)
    print(f"Correct on {torch.cuda.get_device_name()}; first 5 values: {result[:5].tolist()}")


if __name__ == "__main__":
    main()
