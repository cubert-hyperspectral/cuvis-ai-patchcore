"""PatchCoreDetector exports to ONNX (dynamo, opset 18) within the exporter's default tolerance.

The detector's distance (``_nearest_distance``) is written so the reported value never depends on
how a backend implements ``torch.cdist``'s matmul-expansion cancellation: the matmul form only
picks the nearest coreset row, and the distance itself is the exact difference-form norm on fp32
rows. That keeps a traced (ONNX) run within atol 1e-5 / rtol 1e-4 of the eager run on a realistic
fitted coreset, with no extra tolerance needed at export time.
"""

from __future__ import annotations

import pytest
import torch

from cuvis_ai_patchcore.node.patchcore import PatchCoreDetector

pytestmark = pytest.mark.unit
pytest.importorskip("onnxscript")
ort = pytest.importorskip("onnxruntime")

C, H, W = 6, 12, 10
FIT_KW = {
    "input_channels": C,
    "coreset_size": 50,
    "stride": 2,
    "bank_stride": 1,
    "pool_size": 3,
    "max_bank_size": 400,
    "chunk_size": 16,  # small chunks so the export exercises the loop more than once
}


def _stream(n_cubes: int = 5, seed: int = 0) -> list[dict[str, torch.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    scale = torch.tensor([1.0, 4.0, 0.5, 20.0, 2.0, 8.0])
    offset = torch.tensor([0.0, 10.0, -3.0, 100.0, 1.0, -5.0])
    return [{"cube": torch.rand(1, H, W, C, generator=g) * scale + offset} for _ in range(n_cubes)]


def _fitted(**overrides) -> PatchCoreDetector:
    node = PatchCoreDetector(**{**FIT_KW, **overrides})
    node.statistical_initialization(iter(_stream()))
    return node


class _ScoresOnly(torch.nn.Module):
    """Wraps the node so the export has a single tensor output: the dynamo exporter traces a
    plain module, and the port dict the node itself returns is not part of what is compared."""

    def __init__(self, node: PatchCoreDetector, with_reference: bool) -> None:
        super().__init__()
        self.node = node
        self.with_reference = with_reference

    def forward(self, cube: torch.Tensor, reference: torch.Tensor | None = None) -> torch.Tensor:
        kwargs = {"reference": reference} if self.with_reference else {}
        return self.node(cube=cube, **kwargs)["scores"]


def _export_and_run(module: torch.nn.Module, example_inputs: tuple, tmp_path, name: str):
    onnx_path = tmp_path / f"{name}.onnx"
    torch.onnx.export(
        module.eval(),
        example_inputs,
        str(onnx_path),
        dynamo=True,
        opset_version=18,
        input_names=["cube", "reference"] if len(example_inputs) > 1 else ["cube"],
        output_names=["scores"],
    )
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feeds = {
        inp.name: tensor.numpy()
        for inp, tensor in zip(session.get_inputs(), example_inputs, strict=True)
    }
    (onnx_out,) = session.run(["scores"], feeds)
    return onnx_out


def test_export_matches_eager_within_default_tolerance(tmp_path):
    node = _fitted()
    cube = torch.rand(2, H, W, C, generator=torch.Generator().manual_seed(21))
    wrapper = _ScoresOnly(node, with_reference=False)

    with torch.no_grad():
        eager = wrapper(cube)
    onnx_out = _export_and_run(wrapper, (cube,), tmp_path, "patchcore_no_reference")

    torch.testing.assert_close(torch.from_numpy(onnx_out), eager, atol=1e-5, rtol=1e-4)


def test_export_with_reference_port_matches_eager_within_default_tolerance(tmp_path):
    node = _fitted()
    cube = torch.rand(2, H, W, C, generator=torch.Generator().manual_seed(22))
    reference = torch.zeros(2, 20, 17, 3)
    wrapper = _ScoresOnly(node, with_reference=True)

    with torch.no_grad():
        eager = wrapper(cube, reference)
    onnx_out = _export_and_run(wrapper, (cube, reference), tmp_path, "patchcore_with_reference")

    torch.testing.assert_close(torch.from_numpy(onnx_out), eager, atol=1e-5, rtol=1e-4)
    assert onnx_out.shape == (2, 20, 17, 1)


def test_export_tags_the_node_as_onnx_exportable():
    from cuvis_ai_schemas.enums import NodeTag

    assert NodeTag.ONNX in PatchCoreDetector(**FIT_KW)._tags
