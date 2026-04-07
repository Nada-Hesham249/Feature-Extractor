import sys

from PySide6.QtWidgets import QApplication, QMainWindow

from models.image_model import ImageModel
from views.ui_main_window import Ui_MainWindow
from PySide6.QtCore import QLocale
from controllers.main_controller import MainController

QLocale.setDefault(QLocale(QLocale.English, QLocale.UnitedStates))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Setup for the UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()

    # 2. Create the Model
    model = ImageModel()
    controller = MainController(window.ui, model, window)

    window.show()
    sys.exit(app.exec())
