import cv2
import numpy as np
from PySide6.QtCore import Qt

from core.matcher import run_matching_pipeline
from utils.converters import cv_to_pixmap


class MatchingController:
    def __init__(self, ui, model):
        self.ui = ui
        self.model = model
        self._connect_signals()

    # -------------------------
    # Signals
    # -------------------------
    def _connect_signals(self):
        self.ui.btnMatchSSD.clicked.connect(self.run_match_ssd)
        self.ui.btnMatchNCC.clicked.connect(self.run_match_ncc)

    # -------------------------
    # Load images from UI
    # -------------------------
    def _get_image_from_label(self, label):
        path = label.property("imagePath")
        if not path:
            return None
        return cv2.imread(path)

    def _load_match_images(self):
        img1 = self._get_image_from_label(self.ui.lblMatchImage1)
        img2 = self._get_image_from_label(self.ui.lblMatchImage2)

        if img1 is None or img2 is None:
            self.ui.statusbar.showMessage("Please load both images.", 3000)
            return None, None

        return img1, img2

    # -------------------------
    # Keypoint conversion
    # -------------------------
    def _to_image_point(self, kp):
        octave, _, i, j = kp
        scale = 2 ** octave
        return int(j * scale), int(i * scale)

    # -------------------------
    # Colors (Visualization only)
    # -------------------------
    def _match_color(self, index):
        palette = [
            (0, 255, 0),
            (0, 255, 255),
            (255, 0, 0),
            (255, 0, 255),
            (0, 128, 255),
            (255, 255, 0),
            (0, 165, 255),
            (255, 128, 0),
        ]
        return palette[index % len(palette)]

    # -------------------------
    # Drawing only (NO logic here)
    # -------------------------
    def _draw_matches(self, img1, img2, kp1, kp2, matches):
        threshold = self.ui.matchThreshold.value()
        matches = [m for m in matches if m[2] <= threshold]

        max_matches = max(1, min(int(self.ui.matchCount.value()), len(matches)))
        matches = matches[:max_matches]

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        output = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        output[:h1, :w1] = img1
        output[:h2, w1:w1 + w2] = img2

        for idx, (i, j, _) in enumerate(matches):
            color = self._match_color(idx)

            pt1 = self._to_image_point(kp1[i])
            pt2 = self._to_image_point(kp2[j])

            pt2_shifted = (pt2[0] + w1, pt2[1])

            cv2.circle(output, pt1, 3, color, -1)
            cv2.circle(output, pt2_shifted, 3, color, -1)
            cv2.line(output, pt1, pt2_shifted, color, 2)

        return output

    # -------------------------
    # Main pipeline call
    # -------------------------
    def _run_matching(self, method):
        img1, img2 = self._load_match_images()
        if img1 is None:
            return

        matches, kp1, kp2, elapsed = run_matching_pipeline(img1, img2, method)

        if not matches:
            self.ui.statusbar.showMessage("No matches found.", 3000)
            return

        result = self._draw_matches(img1, img2, kp1, kp2, matches)

        pixmap = cv_to_pixmap(result)
        if pixmap:
            self.ui.lblMatchResult.setPixmap(
                pixmap.scaled(
                    self.ui.lblMatchResult.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

        self.ui.lblTimeValueSingle.setText(f"{elapsed:.3f} S")
        self.ui.statusbar.showMessage(f"Matching completed using {method.upper()}", 3000)

    # -------------------------
    # Buttons
    # -------------------------
    def run_match_ssd(self):
        self._run_matching("ssd")

    def run_match_ncc(self):
        self._run_matching("ncc")