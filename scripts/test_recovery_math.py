"""Small tensor regression test for self-recovery posture geometry."""

import importlib.util
import itertools
from pathlib import Path

import torch


_MODULE_PATH = Path(__file__).parents[1] / "src/go2_recovery/recovery_math.py"
_SPEC = importlib.util.spec_from_file_location("go2_recovery_math", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(_MODULE)
upright_error_squared = _MODULE.upright_error_squared


def main() -> None:
    gravity = torch.tensor(
        (
            (0.0, 0.0, -1.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
        )
    )
    error = upright_error_squared(gravity)
    expected = torch.tensor((0.0, 4.0, 2.0, 2.0))
    torch.testing.assert_close(error, expected)
    assert error[1] > error[2] > error[0]
    print(f"upright posture regression passed: {error.tolist()}")
    torch.manual_seed(42)
    matrices, _ = torch.linalg.qr(torch.randn(1000, 3, 3))
    margins = torch.full((1000,), 0.02)
    height = _MODULE.reset_clearance_height(matrices, margins)
    corners = torch.tensor(list(itertools.product((-0.45, 0.45), (-0.30, 0.30), (-0.55, 0.18))))
    world_z = torch.einsum("ni,ki->nk", matrices[:, 2, :], corners) + height[:, None]
    torch.testing.assert_close(world_z.min(dim=1).values, margins, atol=2e-7, rtol=1e-5)
    print("Rotated envelope clearance passed for 1000 rotations (not a mesh-bound validation).")


if __name__ == "__main__":
    main()
