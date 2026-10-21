"""Microsoft TATR detection and structure checkpoints through Transformers."""

from typing import Protocol

from PIL import Image

from .config import ObjectSettings
from .models import Detection


class StructureBackend(Protocol):
    def detect(self, image: Image.Image) -> list[Detection]: ...
    def recognize(self, image: Image.Image) -> list[Detection]: ...


class TATR:
    def __init__(self, detection: ObjectSettings, structure: ObjectSettings):
        self.detection = detection
        self.structure = structure
        self._models: dict[tuple[str, str, str], tuple] = {}

    def _predict(self, image: Image.Image, spec: ObjectSettings) -> list[Detection]:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
        except ImportError as exc:
            raise RuntimeError("Install the 'models' extra to run TATR") from exc
        key = spec.model_id, spec.revision, spec.device
        if key not in self._models:
            processor = AutoImageProcessor.from_pretrained(
                spec.model_id,
                revision=spec.revision,
                cache_dir=spec.cache_dir,
            )
            model = (
                AutoModelForObjectDetection.from_pretrained(
                    spec.model_id,
                    revision=spec.revision,
                    use_safetensors=True,
                    use_pretrained_backbone=False,
                    cache_dir=spec.cache_dir,
                )
                .to(spec.device)
                .eval()
            )
            self._models[key] = processor, model
        processor, model = self._models[key]
        inputs = processor(images=image, return_tensors="pt").to(spec.device)
        with torch.inference_mode():
            output = model(**inputs)
        result = processor.post_process_object_detection(
            output,
            threshold=0.0,
            target_sizes=torch.tensor([image.size[::-1]], device=spec.device),
        )[0]
        objects = []
        for score, label, box in zip(
            result["scores"], result["labels"], result["boxes"], strict=True
        ):
            name = model.config.id2label[int(label)]
            values = tuple(float(v) for v in box.tolist())
            if name != "no object" and values[0] < values[2] and values[1] < values[3]:
                objects.append(Detection(label=name, score=float(score), bbox=values))
        return objects

    def detect(self, image: Image.Image) -> list[Detection]:
        return self._predict(image, self.detection)

    def recognize(self, image: Image.Image) -> list[Detection]:
        return self._predict(image, self.structure)
