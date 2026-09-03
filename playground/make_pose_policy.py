"""Build the identity 'pose policy' ONNX for the Lab-2 slider scene.

obs[1, 14] -> actions[1, 14], y = x. In the browser the 14-wide observation IS the
slider block, and with `use_default_offset=True` the action is an offset from the
scene's keyframe pose, so each slider moves one joint. Opset 18 = Pollen parity.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from microduck_sim.obs import NUM_JOINTS


def build(out_path: Path) -> Path:
    x = helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, NUM_JOINTS])
    y = helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, NUM_JOINTS])
    eye = helper.make_tensor(
        "eye", TensorProto.FLOAT, [NUM_JOINTS, NUM_JOINTS],
        np.eye(NUM_JOINTS, dtype=np.float32).flatten().tolist(),
    )
    node = helper.make_node("MatMul", ["obs", "eye"], ["actions"])
    graph = helper.make_graph([node], "identity_pose", [x], [y], initializer=[eye])
    model = helper.make_model(
        graph, producer_name="microduck-course",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    return out_path


if __name__ == "__main__":
    default = Path(__file__).resolve().parents[1] / "assets" / "checkpoints" / "identity_pose.onnx"
    print(build(Path(sys.argv[1]) if len(sys.argv) > 1 else default))
