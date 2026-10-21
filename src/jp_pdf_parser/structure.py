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
        self._models: dict[tuple[str, str, str, str | None], tuple] = {}

    def load(self, spec: ObjectSettings) -> tuple:
        try:
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
        except ImportError as exc:
            raise RuntimeError("Install the 'tatr' extra to run TATR") from exc
        key = spec.model_id, spec.revision, spec.device, str(spec.local_dir)
        if key not in self._models:
            if spec.local_dir is not None:
                from .checkpoints import verify_checkpoint

                verify_checkpoint(spec)
            source = str(spec.local_dir) if spec.local_dir is not None else spec.model_id
            options = {
                "local_files_only": spec.local_dir is not None,
                "cache_dir": spec.cache_dir,
                "revision": spec.revision,
            }
            processor = AutoImageProcessor.from_pretrained(
                source,
                use_fast=False,
                **options,
            )
            # The structure checkpoint stores a longest-edge-only size. Transformers
            # 4.57's slow DETR processor requires both keys; equal limits retain
            # the same aspect-preserving longest-edge resize. Keep upstream files intact.
            if set(processor.size) == {"longest_edge"}:
                edge = processor.size["longest_edge"]
                processor.size = {"shortest_edge": edge, "longest_edge": edge}
            model = (
                AutoModelForObjectDetection.from_pretrained(
                    source,
                    use_safetensors=True,
                    use_pretrained_backbone=False,
                    **options,
                )
                .to(spec.device)
                .eval()
            )
            self._models[key] = processor, model
        return self._models[key]

    def _predict(self, image: Image.Image, spec: ObjectSettings) -> list[Detection]:
        import torch

        processor, model = self.load(spec)
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
