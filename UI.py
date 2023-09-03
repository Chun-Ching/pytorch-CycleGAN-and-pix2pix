import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QLineEdit
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import cv2
import numpy as np
import torch
import ESRGAN.RRDBNet_arch as arch
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image
from PIL.ImageQt import ImageQt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load the ESRGAN model
        model_path = './ESRGAN/models/RRDB_ESRGAN_x4.pth'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        esrgan_model = arch.RRDBNet(3, 3, 64, 23, gc=32)
        esrgan_model.load_state_dict(torch.load(model_path), strict=True)
        esrgan_model.eval()
        esrgan_model = esrgan_model.to(device)

        self.esrgan_model = esrgan_model
        self.device = device

        # Create the main window
        self.setWindowTitle("Super Resolution")
        self.setGeometry(100, 100, 800, 600)

        # Create a label to display the image
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.image_label)

        # Create a button to open the file dialog
        self.button = QPushButton("Open Image", self)
        self.button.clicked.connect(self.open_image_dialog)

        # Create a line edit to display the selected image path
        self.image_path_edit = QLineEdit(self)
        self.image_path_edit.setReadOnly(True)

        # Create a vertical layout
        layout = QVBoxLayout()
        layout.addWidget(self.button)
        layout.addWidget(self.image_path_edit)
        layout.addWidget(self.image_label)

        # Create a widget and set the layout
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def process_image(self, input_image):
        # Convert PIL image to tensor
        input_tensor = ToTensor()(input_image).unsqueeze(0).to(self.device)

        # Perform super-resolution
        with torch.no_grad():
            output_tensor = self.esrgan_model(input_tensor).clamp(0, 1)

        # Convert tensor to PIL image
        output_image = ToPILImage()(output_tensor.squeeze().cpu())

        return output_image

    def display_image(self, image):
        # Convert PIL image to QPixmap
        image = ImageQt.toqpixmap(image)

        # Scale the image to fit the label
        scaled_image = image.scaled(self.image_label.size(), Qt.KeepAspectRatio)

        # Set the scaled image to the label
        self.image_label.setPixmap(scaled_image)

    def process_input_image(self, image_path):
        input_image = Image.open(image_path).convert("RGB")
        output_image = self.process_image(input_image)
        self.display_image(output_image)

    def open_image_dialog(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            self.image_path_edit.setText(file_path)
            self.process_input_image(file_path)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
