from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class CLIPScorer:
    def __init__(self, model_name: str, *, device: str = "cuda") -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def image_embedding(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        embedding = self.model.get_image_features(**{k: v.to(self.device) for k, v in inputs.items()})
        return F.normalize(embedding.float(), dim=-1)[0]

    @torch.no_grad()
    def text_embedding(self, text: str) -> torch.Tensor:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        embedding = self.model.get_text_features(**{k: v.to(self.device) for k, v in inputs.items()})
        return F.normalize(embedding.float(), dim=-1)[0]

    def score(self, source: Image.Image, edited: Image.Image, source_prompt: str, target_prompt: str) -> dict[str, float]:
        source_image = self.image_embedding(source)
        edited_image = self.image_embedding(edited)
        source_text = self.text_embedding(source_prompt)
        target_text = self.text_embedding(target_prompt)
        image_direction = F.normalize((edited_image - source_image).unsqueeze(0), dim=-1)[0]
        text_direction = F.normalize((target_text - source_text).unsqueeze(0), dim=-1)[0]
        return {
            "cliptxt_target": float(torch.dot(edited_image, target_text).item()),
            "clipimg": float(torch.dot(source_image, edited_image).item()),
            "clipdir": float(torch.dot(image_direction, text_direction).item()),
        }


class DINOScorer:
    def __init__(self, model_name: str, *, device: str = "cuda") -> None:
        from transformers import AutoImageProcessor, AutoModel

        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def embedding(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        output = self.model(**{k: v.to(self.device) for k, v in inputs.items()})
        return F.normalize(output.last_hidden_state[:, 0].float(), dim=-1)[0]

    def similarity(self, source: Image.Image, edited: Image.Image) -> float:
        return float(torch.dot(self.embedding(source), self.embedding(edited)).item())


class LPIPSScorer:
    def __init__(self, *, net: str = "alex", device: str = "cuda") -> None:
        import lpips

        self.device = torch.device(device)
        self.model = lpips.LPIPS(net=net).to(self.device).eval()

    @staticmethod
    def _tensor(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)

    @torch.no_grad()
    def distance(self, source: Image.Image, edited: Image.Image) -> float:
        return float(self.model(self._tensor(source).to(self.device), self._tensor(edited).to(self.device)).item())
