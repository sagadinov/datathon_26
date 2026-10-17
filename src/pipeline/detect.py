"""Open-vocabulary object detection with one backend-agnostic interface (CPU-first).

Backends (all Apache-2.0, loaded offline from models/hf):
  * "gdino" -> IDEA-Research/grounding-dino-tiny  (text prompt "a. b. c.")
  * "owlv2" -> google/owlv2-base-patch16-ensemble (list of text queries)

Our canonical vocabulary (VOCAB) maps a class id to one or more prompt phrases; the
detector returns canonical ids, so the rule engine never sees backend-specific text.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch

import src.common  # noqa: F401  (sets HF offline env before transformers is imported)

log = logging.getLogger("pipeline.detect")

# canonical class -> prompt phrases (lowercase, no trailing period)
VOCAB: dict[str, list[str]] = {
    "person": ["person"],
    "cash": ["banknotes", "cash money", "stack of paper money"],
    "money_bag": ["green money bag", "cash bag"],
    "bag": ["handbag", "backpack", "shoulder bag"],
    "cup": ["cup", "coffee cup", "drinking glass"],
    "jacket": ["jacket", "coat"],
    "laptop": ["laptop"],
    "phone": ["mobile phone"],
    "safe": ["safe", "metal cabinet"],
    "safe_door_open": ["open safe door", "open cabinet door"],
    "desk": ["desk", "table"],
    "door": ["glass door"],
    "counter_machine": ["money counting machine"],
    "book": ["magazine", "book"],
}

MODEL_IDS = {
    "gdino": "IDEA-Research/grounding-dino-tiny",
    "owlv2": "google/owlv2-base-patch16-ensemble",
}


@dataclass
class Detection:
    label: str                        # canonical VOCAB key
    score: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels of the input frame
    phrase: str = ""                  # raw backend phrase (debugging only)

    @property
    def xyxy(self) -> np.ndarray:
        return np.array(self.box, dtype=np.float32)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _phrase_to_label(phrase: str, phrases: dict[str, str]) -> str:
    """Map a backend phrase (possibly merged/partial) to the canonical class with max word overlap."""
    if phrase in phrases:
        return phrases[phrase]
    words = set(phrase.split())
    best, best_n = "unknown", 0
    for p, label in phrases.items():
        n = len(words & set(p.split()))
        if n > best_n:
            best, best_n = label, n
    return best


class Detector:
    """Unified detector. `detect()` works on a BGR uint8 frame and returns canonical detections."""

    def __init__(self, backend: str = "gdino", device: str = "cpu", vocab: dict[str, list[str]] | None = None,
                 box_threshold: float = 0.25, text_threshold: float = 0.25, shortest_edge: int | None = None,
                 threads: int | None = None):
        self.backend, self.device = backend, device
        self.vocab = vocab or VOCAB
        self.box_threshold, self.text_threshold = box_threshold, text_threshold
        self.phrases: dict[str, str] = {p: k for k, ps in self.vocab.items() for p in ps}
        if threads and device == "cpu":
            torch.set_num_threads(threads)
        t0 = time.perf_counter()
        if backend == "gdino":
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(MODEL_IDS[backend])
            if shortest_edge:
                self.processor.image_processor.size = {"shortest_edge": shortest_edge,
                                                       "longest_edge": int(shortest_edge * 1.67)}
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_IDS[backend])
            self.text = ". ".join(self.phrases) + "."
        elif backend == "owlv2":
            from transformers import Owlv2ForObjectDetection, Owlv2Processor

            self.processor = Owlv2Processor.from_pretrained(MODEL_IDS[backend])
            self.model = Owlv2ForObjectDetection.from_pretrained(MODEL_IDS[backend])
            self.queries = list(self.phrases)
        else:
            raise ValueError(f"unknown backend {backend}")
        self.model.to(device).eval()
        log.info("loaded %s (%s) in %.1fs", backend, MODEL_IDS[backend], time.perf_counter() - t0)

    @torch.inference_mode()
    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        from PIL import Image

        h, w = image_bgr.shape[:2]
        pil = Image.fromarray(image_bgr[:, :, ::-1])
        if self.backend == "gdino":
            inputs = self.processor(images=pil, text=self.text, return_tensors="pt").to(self.device)
            out = self.model(**inputs)
            res = self.processor.post_process_grounded_object_detection(
                out, inputs.input_ids, threshold=self.box_threshold, text_threshold=self.text_threshold,
                target_sizes=[(h, w)])[0]
            labels = res.get("text_labels") or res.get("labels")
            return [Detection(_phrase_to_label(str(ph), self.phrases), float(s), tuple(b.tolist()), str(ph))
                    for ph, s, b in zip(labels, res["scores"], res["boxes"])]
        # owlv2: the processor pads the image to a square, so boxes live in a (side x side) frame
        side = max(h, w)
        inputs = self.processor(text=[self.queries], images=pil, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        res = self.processor.post_process_object_detection(
            out, threshold=self.box_threshold, target_sizes=torch.tensor([[side, side]]))[0]
        dets = []
        for idx, s, b in zip(res["labels"], res["scores"], res["boxes"]):
            phrase = self.queries[int(idx)]
            x1, y1, x2, y2 = (float(v) for v in b.tolist())
            box = (max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2))
            dets.append(Detection(self.phrases[phrase], float(s), box, phrase))
        return dets

    def detect_many(self, frames, log_every: int = 20) -> list[list[Detection]]:
        """Run `detect` over an iterable of frames (objects with .load()) with progress logging."""
        results, t0 = [], time.perf_counter()
        for i, f in enumerate(frames):
            results.append(self.detect(f.load()))
            if log_every and (i + 1) % log_every == 0:
                log.info("detect %d frames, %.2fs/frame", i + 1, (time.perf_counter() - t0) / (i + 1))
        return results


def draw(image_bgr: np.ndarray, dets: list[Detection], min_score: float = 0.0) -> np.ndarray:
    """Return a copy of the frame with boxes and 'label score' captions (debugging)."""
    import cv2

    img = image_bgr.copy()
    for d in dets:
        if d.score < min_score:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in d.box)
        color = _color(d.label)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cap = f"{d.label} {d.score:.2f}"
        (tw, th), _ = cv2.getTextSize(cap, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th - 4)), (x1 + tw + 2, y1), color, -1)
        cv2.putText(img, cap, (x1 + 1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return img


def _color(label: str) -> tuple[int, int, int]:
    rng = np.random.default_rng(abs(hash(label)) % (2 ** 32))
    return tuple(int(v) for v in rng.integers(60, 255, 3))
