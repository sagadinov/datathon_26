"""Open-vocabulary object detection on CPU with Grounding DINO tiny (Apache-2.0), offline.

Why not the stock HF post-processing: it merges tokens of several prompt phrases into one
label ("open safe door glass door"). Here every query box is scored per phrase (max sigmoid
over that phrase's tokens) and assigned to the single best phrase, then class-wise NMS.
The rule engine only ever sees canonical class ids from VOCAB, never raw text.
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
    "hand": ["hand"],
    "cash": ["banknotes", "cash", "stack of money"],
    "money_bag": ["money bag", "cash bag"],
    "bag": ["handbag", "backpack"],
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
MODEL_ID = "IDEA-Research/grounding-dino-tiny"


@dataclass
class Detection:
    label: str                                  # canonical VOCAB key
    score: float
    box: tuple[float, float, float, float]      # x1, y1, x2, y2 pixels in the input frame
    phrase: str = ""                            # winning prompt phrase (debugging)

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

    def to_list(self) -> list:
        return [self.label, round(self.score, 4), *[round(v, 1) for v in self.box], self.phrase]

    @classmethod
    def from_list(cls, row: list) -> "Detection":
        return cls(row[0], float(row[1]), tuple(float(v) for v in row[2:6]), row[6] if len(row) > 6 else "")


class Detector:
    """Grounding DINO with per-phrase scoring. `detect()` takes a BGR uint8 frame."""

    def __init__(self, vocab: dict[str, list[str]] | None = None, box_threshold: float = 0.2,
                 shortest_edge: int = 384, longest_edge: int = 640, nms_iou: float = 0.6,
                 device: str = "cpu", threads: int | None = None):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.vocab = vocab or VOCAB
        self.box_threshold, self.nms_iou, self.device = box_threshold, nms_iou, device
        if threads and device == "cpu":
            torch.set_num_threads(threads)
        t0 = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.processor.image_processor.size = {"shortest_edge": shortest_edge, "longest_edge": longest_edge}
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device).eval()
        self.phrases = [p.lower() for ps in self.vocab.values() for p in ps]
        self.phrase_label = {p.lower(): k for k, ps in self.vocab.items() for p in ps}
        self.text = ". ".join(self.phrases) + "."
        self.token_phrase = self._token_phrase_index()
        self.config = {"model": MODEL_ID, "shortest_edge": shortest_edge, "longest_edge": longest_edge,
                       "box_threshold": box_threshold, "nms_iou": nms_iou, "phrases": self.phrases}
        log.info("loaded %s in %.1fs (%d phrases)", MODEL_ID, time.perf_counter() - t0, len(self.phrases))

    def _token_phrase_index(self) -> torch.Tensor:
        """For each text token position, the phrase index it belongs to (-1 for separators/specials)."""
        spans, pos = [], 0
        for p in self.phrases:
            start = self.text.index(p, pos)
            spans.append((start, start + len(p)))
            pos = start + len(p)
        enc = self.processor.tokenizer(self.text, return_offsets_mapping=True)
        idx = []
        for (s, e) in enc["offset_mapping"]:
            k = -1
            if e > s:
                for j, (ps, pe) in enumerate(spans):
                    if ps <= s < pe:
                        k = j
                        break
            idx.append(k)
        return torch.tensor(idx)

    @torch.inference_mode()
    def detect_batch(self, images_bgr: list[np.ndarray]) -> list[list[Detection]]:
        from PIL import Image
        from torchvision.ops import batched_nms

        pils = [Image.fromarray(im[:, :, ::-1]) for im in images_bgr]
        inputs = self.processor(images=pils, text=[self.text] * len(pils), return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        probs = out.logits.sigmoid()                      # (B, Q, T)
        boxes_n = out.pred_boxes                          # (B, Q, 4) normalised cxcywh
        n_tok = min(probs.shape[-1], len(self.token_phrase))
        tok_phrase = self.token_phrase[:n_tok]
        results = []
        for b, im in enumerate(images_bgr):
            h, w = im.shape[:2]
            per_phrase = torch.stack([probs[b, :, :n_tok][:, tok_phrase == j].max(dim=1).values
                                      for j in range(len(self.phrases))], dim=1)   # (Q, P)
            best_s, best_p = per_phrase.max(dim=1)
            keep = best_s > self.box_threshold
            if keep.sum() == 0:
                results.append([])
                continue
            cx, cy, bw, bh = boxes_n[b][keep].unbind(-1)
            xyxy = torch.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], -1)
            xyxy[:, 0::2] = xyxy[:, 0::2].clamp(0, w)
            xyxy[:, 1::2] = xyxy[:, 1::2].clamp(0, h)
            scores, ph = best_s[keep], best_p[keep]
            cls = torch.tensor([list(self.vocab).index(self.phrase_label[self.phrases[int(j)]]) for j in ph])
            kept = batched_nms(xyxy, scores, cls, self.nms_iou)
            dets = [Detection(self.phrase_label[self.phrases[int(ph[i])]], float(scores[i]),
                              tuple(float(v) for v in xyxy[i].tolist()), self.phrases[int(ph[i])]) for i in kept]
            results.append(sorted(dets, key=lambda d: -d.score))
        return results

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        return self.detect_batch([image_bgr])[0]


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
    rng = np.random.default_rng(sum(ord(c) * (i + 1) for i, c in enumerate(label)) % (2 ** 32))
    return tuple(int(v) for v in rng.integers(60, 255, 3))
