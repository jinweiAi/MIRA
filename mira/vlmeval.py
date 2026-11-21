"""Optional VLMEvalKit model adapter. VLMEvalKit is installed separately.

Usage from a VLMEvalKit checkout:
  from functools import partial
  from vlmeval.config import supported_VLM
  from mira.vlmeval import MIRAModel
  supported_VLM["MIRA"] = partial(MIRAModel, checkpoint="runs/stage2/final.pt")
  # Then use VLMEvalKit's run.py or dataset.evaluate with its native protocols.
"""
import os
from vlmeval.vlm.base import BaseModel
from .inference import generate, load_mira

class MIRAModel(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, checkpoint=None, model=None, device="cuda:0",
                 max_new_tokens=16, max_pixels=None, **kwargs):
        super().__init__()
        checkpoint = checkpoint or os.environ.get("MIRA_CHECKPOINT")
        if not checkpoint:
            raise ValueError("Set MIRA_CHECKPOINT or pass checkpoint")
        model = model or os.environ.get("MIRA_BASE_MODEL")
        self.model, self.processor, self.decoder, self.latents, self.config = load_mira(
            checkpoint, model, device, max_pixels)
        self.max_new_tokens = max_new_tokens

    def generate_inner(self, message, dataset=None):
        # Preserve VLMEvalKit's image/text order in the Qwen user content.
        content = []
        for item in message:
            if item["type"] == "image":
                content.append({"type": "image", "image": item["value"]})
            elif item["type"] == "text":
                content.append({"type": "text", "text": item["value"]})
            else:
                raise ValueError(f"Unsupported input type: {item['type']}")
        row = {"id": "vlmeval", "images": [x["value"] for x in message if x["type"] == "image"],
               "question": "\n".join(x["value"] for x in message if x["type"] == "text"),
               "_content": content}
        if not row["images"]:
            raise ValueError("MIRA requires at least one image")
        return generate(self.model, self.processor, self.decoder, self.latents,
                        self.config, row, self.max_new_tokens)["prediction"]
