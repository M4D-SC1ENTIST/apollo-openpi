import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_zktp_example() -> dict:
    """Creates a random input example for the ZKTP policy."""
    return {
        "observation/state": np.random.rand(8),  # 7 joints + 1 gripper
        "observation/wrist_image": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "prompt": "open the cabinet on the right",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class ZKTPInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For ZKTP dataset, we have wrist camera images, robot state (7 joints + 1 gripper), and natural language tasks.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Parse wrist image to uint8 (H,W,C) format
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                # ZKTP only has wrist camera, so we use it as the base image and left wrist
                "base_0_rgb": wrist_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad right wrist image with zeros since ZKTP doesn't have it
                "right_wrist_0_rgb": np.zeros_like(wrist_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # For ZKTP, the task description is stored in "task" field
        if "task" in data:
            inputs["prompt"] = data["task"]
        elif "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class ZKTPOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back to the dataset specific format. It is
    used for inference only.

    For ZKTP dataset, we return 8 actions (7 joint commands + 1 gripper command).
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first 8 actions for ZKTP (7 joints + 1 gripper)
        # since we padded actions above to fit the model action dimension
        return {"actions": np.asarray(data["actions"][:, :8])}
