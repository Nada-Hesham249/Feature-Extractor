import cv2
from PySide6.QtWidgets import QFileDialog
from PySide6.QtCore import Qt
from utils.converters import cv_to_pixmap
from corner_detection_controller import CornerDetectionController


class MainController:
    def __init__(self, ui, model, window):
        self.ui = ui
        self.model = model
        self.window = window

        # Instantiate sub-controllers
        self.corner_ctrl = CornerDetectionController(self)

        self._connect_signals()

    def _connect_signals(self):
        # File menu
        self.ui.actionOpen_Image.triggered.connect(self.load_image)
        self.ui.actionSave_Result.triggered.connect(self.save_result)
        self.ui.actionExit.triggered.connect(self.window.close)

        # Top-bar buttons
        self.ui.btnUploadOriginal.clicked.connect(self.load_image)
        self.ui.btnReset.clicked.connect(self.reset_view)

        # Harris / corner detection – "Apply" button in the
        # "Extract Unique Features" tab
        self.ui.btnApplyHarris.clicked.connect(self.apply_harris)

        # Match image uploads
        self.ui.btnUploadMatchImage1.clicked.connect(
            lambda: self.load_match_image(slot=1)
        )
        self.ui.btnUploadMatchImage2.clicked.connect(
            lambda: self.load_match_image(slot=2)
        )

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.window, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_path:
            return

        self.model.original_image = cv2.imread(file_path)
        self.model.processed_image = None

        self.display_original_image(self.model.original_image)

        self.ui.lblProcessed.clear()
        self.ui.lblProcessed.setText("Ready for processing.")
        self.ui.statusbar.showMessage(f"Loaded: {file_path}", 3000)

    # ------------------------------------------------------------------
    # Corner / Harris detection
    # ------------------------------------------------------------------

    def apply_harris(self):
        """Read current UI parameters and delegate to CornerDetectionController."""
        # --- read parameters from the UI if the widgets exist,
        #     otherwise fall back to sensible defaults ---
        k            = self._spin_value("spinK",          default=0.04)
        window_size  = self._spin_value("spinWindowSize", default=5)
        threshold    = self._spin_value("spinThreshold",  default=3500.0)

        # Check box / toggle for λ⁻ mode (optional widget)
        lambda_minus = False
        if hasattr(self.ui, "chkLambdaMinus"):
            lambda_minus = self.ui.chkLambdaMinus.isChecked()

        self.corner_ctrl.apply(
            k=k,
            window_size=int(window_size),
            threshold=float(threshold),
            lambda_minus_flag=lambda_minus,
        )

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def display_original_image(self, cv_img):
        self.display_image_on_label(cv_img, self.ui.lblOriginal)

    def display_processed_image(self, cv_img):
        self.model.processed_image = cv_img
        self.display_image_on_label(cv_img, self.ui.lblProcessed)

    def display_image_on_label(self, cv_img, label):
        if cv_img is None:
            return
        pixmap = cv_to_pixmap(cv_img)
        if pixmap:
            scaled = pixmap.scaled(
                label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            label.setPixmap(scaled)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def reset_view(self):
        self.model.original_image  = None
        self.model.processed_image = None
        self.ui._reset_view_labels()
        self.ui.statusbar.showMessage("View reset.", 2000)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _spin_value(self, widget_name: str, default):
        """Return the value of a spin-box / double-spin-box widget if it
        exists on the UI, otherwise return *default*."""
        widget = getattr(self.ui, widget_name, None)
        if widget is not None:
            return widget.value()
        return default