import torch
import triton
import os

def test_matmul():
    try:
        @torch.compile
        def matmul(a, b):
            return torch.matmul(a, b)

        a = torch.randn(128, 128, device='cuda')
        b = torch.randn(128, 128, device='cuda')
        
        # Warmup
        c = matmul(a, b)
        print("Success: matmul executed")
    except Exception as e:
        import traceback
        lines = traceback.format_exc().splitlines()
        print(f"FAILED_EXCEPTION_START: {lines[0]}")
        print(f"FAILED_EXCEPTION_END: {lines[-1]}")

if __name__ == '__main__':
    test_matmul()
