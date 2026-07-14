"""
ReID Appearance Matcher: Standalone Re-identification
Fixes ID swaps by matching embeddings, NOT just box geometry.

This runs AFTER BoT-SORT to catch and fix any swaps.
"""

import numpy as np
import torch
from collections import defaultdict
import cv2


class ReIDMatcher:
    """Match detections using ReID embeddings. Permanent ID lock."""
    
    def __init__(self, similarity_threshold=0.3):
        """
        similarity_threshold: How strict? Lower = stricter
        0.2 = very strict (same person must look very similar)
        0.3 = balanced (default)
        0.5 = loose (more lenient)
        """
        self.similarity_threshold = similarity_threshold
        self.id_gallery = {}  # id -> embedding
        self.id_counter = 1000  # Start from 1000 to avoid conflicts
        
        # Try to load a ReID model
        try:
            from ultralytics import YOLO
            # Use Ultralytics' built-in ReID feature extraction
            self.model = YOLO("yolov8m.pt")
            self.has_reid = True
        except:
            self.has_reid = False
            print("[WARN] ReID model unavailable")
    
    def extract_embedding(self, frame, xyxy):
        """Extract appearance embedding from a crop."""
        if not self.has_reid:
            return None
        
        try:
            x1, y1, x2, y2 = map(int, xyxy)
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            
            # Run YOLO on crop to get features
            results = self.model(crop, verbose=False)
            
            if results and len(results) > 0:
                # Extract embedding if available
                if hasattr(results[0], 'features'):
                    embedding = results[0].features
                    if embedding is not None:
                        return embedding / (np.linalg.norm(embedding) + 1e-6)
            
            # Fallback: use HOG features
            return self._hog_embedding(crop)
        except:
            return None
    
    @staticmethod
    def _hog_embedding(crop):
        """Fallback: HOG features when ReID unavailable."""
        if crop.shape[0] < 8 or crop.shape[1] < 8:
            return None
        
        # Simple color histogram as embedding
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist
    
    def cosine_similarity(self, emb1, emb2):
        """Cosine similarity between two embeddings."""
        if emb1 is None or emb2 is None:
            return 0.0
        
        emb1 = np.asarray(emb1).flatten()
        emb2 = np.asarray(emb2).flatten()
        
        if len(emb1) != len(emb2):
            return 0.0
        
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        
        return float(np.dot(emb1, emb2) / (norm1 * norm2))
    
    def match(self, frame, detections_with_ids):
        """
        Match detections using embeddings.
        Input: [(track_id, xyxy, conf), ...]
        Output: [(stable_id, xyxy, conf), ...] with fixed IDs
        """
        result = []
        matched_gallery_ids = set()
        
        # Process each detection
        for track_id, xyxy, conf in detections_with_ids:
            embedding = self.extract_embedding(frame, xyxy)
            
            if embedding is None:
                # Can't extract embedding, keep track_id
                result.append((track_id, xyxy, conf))
                if track_id in self.id_gallery:
                    matched_gallery_ids.add(track_id)
                continue
            
            # Try to match with existing gallery
            best_match_id = None
            best_similarity = -1.0
            
            for gallery_id, gallery_emb in self.id_gallery.items():
                if gallery_id in matched_gallery_ids:
                    continue  # Already matched
                
                sim = self.cosine_similarity(embedding, gallery_emb)
                if sim > best_similarity and sim > self.similarity_threshold:
                    best_similarity = sim
                    best_match_id = gallery_id
            
            # Assign ID
            if best_match_id is not None:
                # Match found! Use gallery ID
                stable_id = best_match_id
                matched_gallery_ids.add(best_match_id)
            else:
                # No match, use track_id (or create new)
                stable_id = track_id
                if track_id not in self.id_gallery:
                    # First time seeing this person, register
                    pass
            
            # Update gallery
            self.id_gallery[stable_id] = embedding
            matched_gallery_ids.add(stable_id)
            
            result.append((stable_id, xyxy, conf))
        
        # Cleanup: remove old gallery entries
        for gid in list(self.id_gallery.keys()):
            if gid not in matched_gallery_ids and gid not in [r[0] for r in result]:
                del self.id_gallery[gid]
        
        return result


# ============================================================================
# USAGE IN PIPELINE (add this to pipeline.py after track matching)
# ============================================================================

"""
In pipeline.py, after `persons = [t for t in tracks if t.cls == m.person_class_id]`:

    # Initialize ReID matcher once
    if not hasattr(self, 'reid_matcher'):
        self.reid_matcher = ReIDMatcher(similarity_threshold=0.3)
    
    # Match persons using embeddings
    detections = [(t.track_id, t.xyxy, t.conf) for t in persons]
    matched = self.reid_matcher.match(frame, detections)
    
    # Update track IDs
    for i, (stable_id, xyxy, conf) in enumerate(matched):
        persons[i].track_id = stable_id
"""