import cv2
import numpy as np
from corner_detector import corner_detector


class CornerDetectionController:
    def __init__(self, main_controller):
        """
        Args:
            main_controller: instance of MainController, used to access
                             the shared model and the display helpers.
        """
        self.main_ctrl = main_controller

    # ------------------------------------------------------------------
    # Public entry-point – called by MainController.apply_harris()
    # ------------------------------------------------------------------

    def apply(
        self,
        k: float = 0.04,
        window_size: int = 5,
        threshold: float = 3500.0,
        lambda_minus_flag: bool = False,
    ):
        """Run corner detection on the currently loaded image and display
        the annotated result in the processed-image panel.

        Args:
            k               : Harris sensitivity constant (ignored when
                              lambda_minus_flag is True).
            window_size     : Side length of the sliding window (pixels).
            threshold       : Minimum corner response score to keep.
            lambda_minus_flag: If True, use the λ⁻ (min-eigenvalue)
                              criterion instead of the Harris R score.
        """
        model = self.main_ctrl.model

        if model.original_image is None:
            self.main_ctrl.ui.statusbar.showMessage(
                "No image loaded. Please open an image first.", 3000
            )
            return

        # corner_detector expects a single-channel (grayscale) image
        gray = self._to_grayscale(model.original_image)

        corner_list, output_img = corner_detector(
            gray,
            k=k,
            window_size=window_size,
            threshold=threshold,
            lambda_minus_flag=lambda_minus_flag,
        )

        # Show the annotated image in the processed panel
        self.main_ctrl.display_processed_image(output_img)

        method = "λ⁻ (Min-Eigenvalue)" if lambda_minus_flag else "Harris"
        self.main_ctrl.ui.statusbar.showMessage(
            f"{method} corner detection done – {len(corner_list)} corner(s) found.",
            4000,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_grayscale(cv_img: np.ndarray) -> np.ndarray:
        """Return a uint8 grayscale image.

        corner_detector does cv2.cvtColor(input, COLOR_GRAY2RGB) internally,
        which requires uint8.  np.gradient works fine on uint8 too, so we
        keep it as uint8 rather than float32.
        """
        if cv_img.ndim == 3:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv_img.copy()
        # Ensure uint8 — required by COLOR_GRAY2RGB inside corner_detector
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return gray