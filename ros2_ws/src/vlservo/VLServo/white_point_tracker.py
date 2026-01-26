import numpy as np
import cv2

from .gui_rotation import compute_display_offsets, resolve_rotation_degrees


class WhitePointTracker:
    """Lucas-Kanade + template matching tracker to keep the selected pixel attached to the object.

    Lucas-Kanade optical flow gives smooth per-frame motion, while template matching
    handles re-initialization when the point leaves the image and later re-enters.
    """
    def __init__(self, init_px: int, init_py: int, template_size: int = 41,
                 search_radius: int = 40, min_match_score: float = 0.9,
                 reacquire_score: float = 0.93, max_lost_frames: int = 5):
        self.px = int(init_px)
        self.py = int(init_py)
        self.template_size = int(max(11, template_size | 1))  # ensure odd size >= 11
        self.search_radius = int(max(8, search_radius))
        self.min_match_score = float(min_match_score)
        self.reacquire_score = float(max(reacquire_score, self.min_match_score))
        self.max_lost_frames = int(max(1, max_lost_frames))
        self.template = None
        self.prev_gray = None
        self.prev_pt = None
        self.initialized = False
        self.visible = False
        self.lost_counter = 0

    @staticmethod
    def _to_gray(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    def _extract_patch(self, gray, cx, cy, size):
        h, w = gray.shape[:2]
        half = size // 2
        cx = int(round(cx)); cy = int(round(cy))
        x0 = max(0, cx - half); x1 = min(w, cx + half + 1)
        y0 = max(0, cy - half); y1 = min(h, cy + half + 1)
        patch = gray[y0:y1, x0:x1]
        # Pad bottom/right edges if necessary to keep a consistent template size
        if patch.shape[0] != size or patch.shape[1] != size:
            pad_b = max(0, size - patch.shape[0])
            pad_r = max(0, size - patch.shape[1])
            patch = cv2.copyMakeBorder(patch, 0, pad_b, 0, pad_r, cv2.BORDER_REPLICATE)
        return patch

    def _local_search_region(self, gray):
        h, w = gray.shape[:2]
        half_t = self.template_size // 2
        cx = int(round(self.px))
        cy = int(round(self.py))
        x0 = max(0, cx - self.search_radius - half_t)
        y0 = max(0, cy - self.search_radius - half_t)
        x1 = min(w, cx + self.search_radius + half_t + 1)
        y1 = min(h, cy + self.search_radius + half_t + 1)
        # Expand if the window is smaller than the template
        if (x1 - x0) < self.template_size:
            deficit = self.template_size - (x1 - x0)
            x0 = max(0, x0 - deficit // 2)
            x1 = min(w, x1 + deficit - deficit // 2)
        if (y1 - y0) < self.template_size:
            deficit = self.template_size - (y1 - y0)
            y0 = max(0, y0 - deficit // 2)
            y1 = min(h, y1 + deficit - deficit // 2)
        return gray[y0:y1, x0:x1], x0, y0

    def _match_template(self, gray, full_search=False):
        if self.template is None:
            return None, None, None
        if full_search:
            search = gray
            x0, y0 = 0, 0
        else:
            search, x0, y0 = self._local_search_region(gray)
        if search.shape[0] < self.template_size or search.shape[1] < self.template_size:
            return None, None, None
        res = cv2.matchTemplate(search, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val is None:
            return None, None, None
        half_t = self.template_size // 2
        cx = x0 + max_loc[0] + half_t
        cy = y0 + max_loc[1] + half_t
        return cx, cy, max_val

    def initialize(self, color_image):
        gray = self._to_gray(color_image)
        self.template = self._extract_patch(gray, self.px, self.py, self.template_size)
        self.prev_gray = gray.copy()
        self.prev_pt = np.array([[[float(self.px), float(self.py)]]], dtype=np.float32)
        self.initialized = True
        self.visible = True
        self.lost_counter = 0

    def _set_new_position(self, px, py, gray):
        h, w = gray.shape[:2]
        px = int(np.clip(round(px), 0, w - 1))
        py = int(np.clip(round(py), 0, h - 1))
        self.px, self.py = px, py
        self.prev_pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
        self.prev_gray = gray.copy()
        self.visible = True
        return self.px, self.py

    def update(self, color_image):
        if not self.initialized:
            self.initialize(color_image)
            self.lost_counter = 0
            return self.px, self.py, True

        gray = self._to_gray(color_image)
        h, w = gray.shape[:2]
        found = False
        new_px, new_py = self.px, self.py

        # 1) Lucas-Kanade optical flow
        if self.visible and self.prev_gray is not None and self.prev_pt is not None:
            next_pt, status, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_pt, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )
            if status is not None and int(status.ravel()[0]) == 1:
                cand = next_pt[0, 0]
                if np.all(np.isfinite(cand)):
                    cx, cy = float(cand[0]), float(cand[1])
                    if 0 <= cx < w and 0 <= cy < h and self.template is not None:
                        patch = self._extract_patch(gray, cx, cy, self.template_size)
                        res = cv2.matchTemplate(patch, self.template, cv2.TM_CCOEFF_NORMED)
                        _, score, _, _ = cv2.minMaxLoc(res)
                        if score is not None and score >= self.min_match_score:
                            new_px, new_py = cx, cy
                            found = True

        # 2) Template matching fallback (local search only)
        if not found and self.template is not None:
            cx, cy, score = self._match_template(gray, full_search=False)
            if score is not None and score >= self.reacquire_score and cx is not None and cy is not None:
                new_px, new_py = cx, cy
                found = True

        # 3) Update or accumulate failure count
        if found:
            self.lost_counter = 0
            return self._set_new_position(new_px, new_py, gray) + (True,)

        self.lost_counter = getattr(self, "lost_counter", 0) + 1
        if self.lost_counter >= self.max_lost_frames:
            self.visible = False
            self.prev_pt = None
            self.prev_gray = None
        return self.px, self.py, False

    def get_display_offsets(self, width, height, rotation_deg=None):
        """Return GUI-aligned offsets for the tracked point."""
        if width is None or height is None:
            raise ValueError("Image width/height are required for display offsets")
        rot = resolve_rotation_degrees(rotation_deg)
        return compute_display_offsets(self.px, self.py, width, height, rotation_deg=rot)
