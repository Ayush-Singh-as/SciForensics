from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from model import Model, distance


@dataclass
class PairwiseInput:
    image_path: Path
    rgb_image: np.ndarray
    gray_image: np.ndarray
    bgr_image: np.ndarray
    tensor: torch.Tensor


@dataclass
class GlobalSimilarityResult:
    distance: float
    similarity_score: float
    suspicious: bool
    trigger_local: bool
    left_heatmap: Optional[np.ndarray]
    right_heatmap: Optional[np.ndarray]


class PairwiseEmbeddingModel:
    def __init__(
        self,
        weights_path: Optional[str | Path] = None,
        input_size: int = 128,
        same_threshold: float = 1.1,
        local_trigger_threshold: float = 2.0,
        device: Optional[str] = None,
    ) -> None:
        base_dir = Path(__file__).resolve().parent
        self.weights_path = Path(weights_path) if weights_path else base_dir / "models" / "weights.pth"
        self.input_size = input_size
        self.same_threshold = same_threshold
        self.local_trigger_threshold = local_trigger_threshold
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Model(nin=True).to(self.device)
        state_dict = torch.load(self.weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def load_pair(self, left_path: str | Path, right_path: str | Path) -> tuple[PairwiseInput, PairwiseInput]:
        return self._load_image(left_path), self._load_image(right_path)

    def compare(
        self,
        left_path: str | Path,
        right_path: str | Path,
        generate_heatmaps: bool = True,
    ) -> GlobalSimilarityResult:
        left, right = self.load_pair(left_path, right_path)
        with torch.no_grad():
            left_embedding = self.model(left.tensor.to(self.device))
            right_embedding = self.model(right.tensor.to(self.device))
            pair_distance = float(distance(left_embedding, right_embedding).item())

        similarity_score = float(torch.sigmoid(torch.tensor(1.0 - pair_distance)).item())
        suspicious = pair_distance <= self.same_threshold
        trigger_local = pair_distance <= self.local_trigger_threshold

        left_heatmap = None
        right_heatmap = None
        if generate_heatmaps and trigger_local:
            left_heatmap, right_heatmap = self.localize(left, right)

        return GlobalSimilarityResult(
            distance=pair_distance,
            similarity_score=similarity_score,
            suspicious=suspicious,
            trigger_local=trigger_local,
            left_heatmap=left_heatmap,
            right_heatmap=right_heatmap,
        )

    def localize(self, left: PairwiseInput, right: PairwiseInput) -> tuple[np.ndarray, np.ndarray]:
        self.model.zero_grad(set_to_none=True)
        left_embedding, left_activations = self._forward_with_activations(left.tensor.to(self.device))
        right_embedding, right_activations = self._forward_with_activations(right.tensor.to(self.device))
        pair_distance = torch.sum(torch.abs(left_embedding - right_embedding), dim=-1).mean()
        pair_distance.backward()

        left_heatmap = self._overlay_heatmap(left.bgr_image, left_activations, left_activations.grad)
        right_heatmap = self._overlay_heatmap(right.bgr_image, right_activations, right_activations.grad)
        self.model.zero_grad(set_to_none=True)
        return left_heatmap, right_heatmap

    def _load_image(self, image_path: str | Path) -> PairwiseInput:
        image_path = Path(image_path)
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        display = cv2.resize(gray, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        normalized = display.astype(np.float32) / 255.0
        normalized = (normalized - 0.5) / 0.5
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)

        return PairwiseInput(
            image_path=image_path,
            rgb_image=rgb,
            gray_image=gray,
            bgr_image=bgr,
            tensor=tensor,
        )

    def _forward_with_activations(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        activations = self.model.features(tensor)
        activations.retain_grad()
        out = activations.view(activations.size(0), -1)
        out = self.model.fc1(out)
        out = self.model.relu(out)
        out = self.model.fc2(out)
        return out, activations

    def _overlay_heatmap(
        self,
        base_bgr: np.ndarray,
        activations: torch.Tensor,
        gradients: Optional[torch.Tensor],
    ) -> np.ndarray:
        if gradients is None:
            return base_bgr.copy()

        pooled_gradients = torch.mean(gradients.detach(), dim=[0, 2, 3])
        weighted_activations = activations.detach().clone()[0]
        weighted_activations *= pooled_gradients[:, None, None]
        heatmap = torch.mean(weighted_activations, dim=0)
        heatmap = torch.relu(heatmap)
        max_value = float(torch.max(heatmap).item())
        if max_value > 0.0:
            heatmap = heatmap / max_value

        heatmap_np = heatmap.cpu().numpy()
        heatmap_np = cv2.resize(heatmap_np, (base_bgr.shape[1], base_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
        heatmap_np = np.uint8(np.clip(heatmap_np * 255.0, 0, 255))
        heatmap_color = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)
        return cv2.addWeighted(base_bgr, 0.6, heatmap_color, 0.4, 0.0)
