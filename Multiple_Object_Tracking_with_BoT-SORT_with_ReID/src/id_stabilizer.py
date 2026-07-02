"""
Deep-ReID ID stabilizer.

Instead of color histograms, every person crop is turned into a 512-dim
appearance embedding by a pretrained OSNet ReID model. Embeddings are
compared with cosine similarity -> two different people almost never
match even if they wear the same color.

Logic:
  - keep a gallery: canonical_id -> {embedding, last_box, lost_frames}
  - for every NEW tracker id, compare its embedding to lost gallery
    entries. Strong cosine match -> remap new id back to the old one.
"""
import numpy as np
import torch
import torchreid


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class ReIDExtractor:
    """Wraps an OSNet model that maps a person crop -> 512-dim vector."""

    def __init__(self, device="cpu"):
        self.device = device
        self.model = torchreid.models.build_model(
            name="osnet_x1_0", num_classes=1000, pretrained=True)
        self.model.eval().to(device)
        # OSNet standard input: 256x128, ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    @torch.no_grad()
    def __call__(self, frame, xyxy):
        import cv2
        x1, y1, x2, y2 = map(int, xyxy)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (128, 256))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        crop = (crop - self.mean) / self.std
        t = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).to(self.device)
        feat = self.model(t).cpu().numpy()[0]
        feat = feat / (np.linalg.norm(feat) + 1e-6)   # unit length
        return feat


class IDStabilizer:
    def __init__(self, max_lost=150, sim_gate=0.75, iou_gate=0.20,
                 center_gate=150.0, device="cpu"):
        self.max_lost = max_lost
        self.sim_gate = sim_gate          # min cosine similarity to rescue an id
        self.iou_gate = iou_gate
        self.center_gate = center_gate
        self.extractor = ReIDExtractor(device=device)
        self.gallery = {}                 # canonical_id -> dict
        self.remap = {}                   # raw tracker id -> canonical id

    @staticmethod
    def _center(b):
        return ((b[0]+b[2])/2.0, (b[1]+b[3])/2.0)

    def update(self, frame, tracks):
        for g in self.gallery.values():
            g["lost"] += 1

        seen = set()
        out = {}
        for t in tracks:
            raw, box = t.track_id, t.xyxy

            if raw in self.remap:
                canon = self.remap[raw]
                out[raw] = canon
                self._refresh(canon, frame, box)
                seen.add(canon)
                continue

            emb = self.extractor(frame, box)
            canon = self._match_lost(box, emb, exclude=seen)
            if canon is None:
                canon = raw
            self.remap[raw] = canon
            out[raw] = canon
            self._refresh(canon, frame, box, emb)
            seen.add(canon)

        for cid in [c for c, g in self.gallery.items() if g["lost"] > self.max_lost]:
            del self.gallery[cid]
        for r in [r for r, c in self.remap.items() if c not in self.gallery]:
            del self.remap[r]
        return out

    def _match_lost(self, box, emb, exclude):
        if emb is None:
            return None
        cx, cy = self._center(box)
        best, best_sim = None, -1.0
        for cid, g in self.gallery.items():
            if cid in exclude or g["lost"] == 0:
                continue
            dist = np.hypot(cx - g["cx"], cy - g["cy"])
            if dist > self.center_gate and _iou(box, g["box"]) < self.iou_gate:
                continue
            if g["emb"] is None:
                continue
            sim = float(np.dot(emb, g["emb"]))     # cosine (both unit-norm)
            if sim >= self.sim_gate and sim > best_sim:
                best, best_sim = cid, sim
        return best

    def _refresh(self, canon, frame, box, emb=None):
        if emb is None:
            emb = self.extractor(frame, box)
        cx, cy = self._center(box)
        g = self.gallery.get(canon)
        if g is None:
            self.gallery[canon] = {"emb": emb, "box": box.copy(),
                                   "lost": 0, "cx": cx, "cy": cy}
        else:
            if emb is not None:
                if g["emb"] is None:
                    g["emb"] = emb
                else:
                    g["emb"] = 0.9 * g["emb"] + 0.1 * emb   # slow adapt
                    g["emb"] /= (np.linalg.norm(g["emb"]) + 1e-6)
            g["box"], g["lost"], g["cx"], g["cy"] = box.copy(), 0, cx, cy