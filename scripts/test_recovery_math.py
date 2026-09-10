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
    # Contact count alone cannot distinguish these five physically different
    # postures. Reject both crossed feet and crossed knees, plus a single
    # severely twisted joint that would be hidden by averaging over 12 joints.
    feet = torch.tensor([[[.20, .14, -.31], [.20, -.14, -.31],
                          [-.20, .14, -.31], [-.20, -.14, -.31]]]).repeat(5, 1, 1)
    knees = feet.clone()
    delta = torch.zeros(5, 12)
    feet[1, :2, 1] *= -1
    knees[2, 0, 1] = -.08
    delta[3, 0] = .9
    feet[4, 1, 0] = -.10
    assert _MODULE.normal_stance_geometry(feet, knees, delta).tolist() == [True, False, False, False, False]
    # Real failing frame from seed20260918/model3498, despite 4 contacts.
    feet[0, 0, 1], feet[0, 1, 1] = -.1774, .0715
    assert not _MODULE.normal_stance_geometry(feet, knees, delta).any()
    print("Crossed feet/knees, wrong fore-aft placement and outlier joint rejected.")
    aligned = torch.tensor([[[.20, .16, -.32], [.20, -.16, -.32],
                            [-.20, .16, -.32], [-.20, -.16, -.32]]])
    wide = aligned.clone()
    # Measured final lateral locations from the 2849 fore-aft video.
    wide[0, :, 1] = torch.tensor([.3033, -.1671, .1747, -.1923])
    wide.requires_grad_(True)
    score = _MODULE.stance_alignment_penalty(wide)
    assert _MODULE.stance_alignment_penalty(aligned).item() == 0
    assert score.item() > 0.25
    mirrored = wide[:, [1, 0, 3, 2]] * wide.new_tensor([1., -1., 1.])
    torch.testing.assert_close(score, _MODULE.stance_alignment_penalty(mirrored))
    score.sum().backward()
    assert wide.grad[0, 0, 1] > 0  # Gradient descent brings the over-wide FL foot inward.
    print("Measured over-wide stance penalized, mirror invariant, with an inward correction gradient.")
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
