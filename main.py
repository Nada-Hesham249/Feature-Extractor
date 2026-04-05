import sys

from PySide6.QtWidgets import QApplication, QMainWindow

from views.ui_main_window import Ui_MainWindow
from PySide6.QtCore import QLocale

QLocale.setDefault(QLocale(QLocale.English, QLocale.UnitedStates))


def main() -> int:
	app = QApplication(sys.argv)

	window = QMainWindow()
	ui = Ui_MainWindow()
	ui.setupUi(window)

	window.show()
	return app.exec()


if __name__ == "__main__":
	raise SystemExit(main())
