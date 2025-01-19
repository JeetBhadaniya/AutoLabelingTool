import sys
import os
import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QToolBar,
    QFileDialog, QVBoxLayout, QWidget, QAction, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt
from segment_anything import SamPredictor, sam_model_registry

# Load SAM model with GPU support
MODEL_TYPE = "vit_b"  # Options: 'vit_b', 'vit_l', 'vit_h'
# Replace with your model's checkpoint path
MODEL_PATH = "sam_vit_b_01ec64.pth"  

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize the SAM model
sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_PATH).to(device)
predictor = SamPredictor(sam)


class AutoLabelingTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

        # Image and mask state
        self.image = None
        self.segmented_image = None
        self.mask_overlay = None
        self.is_drawing = False
        self.brush_size = 10
        self.draw_mode = None  # Modes: 'draw', 'erase', 'auto', 'rectangle', 'circle', 'polygon'
        self.start_point = None  # Used for rectangle and circle drawing
        self.polygon_points = []  # Used for polygon tool

        # State for circle manipulation
        self.selected_shape = None  # Track selected shape properties (e.g., {'type': 'circle', 'center': (x, y), 'radius': r})
        self.is_moving = False
        self.is_resizing = False

        # File handling
        self.image_list = []
        self.current_image_index = 0
        self.save_folder = None
    
    def initUI(self):
        self.setWindowTitle("Auto Labeling Tool - Instance Segmentation")
        self.setGeometry(100, 100, 1200, 800)

        # Main layout
        layout = QVBoxLayout()

        # Toolbar
        toolbar = QToolBar("Toolbar")
        self.addToolBar(toolbar)

        load_images_action = QAction("Load Images", self)
        load_images_action.triggered.connect(self.load_images)
        toolbar.addAction(load_images_action)

        save_folder_action = QAction("Select Save Folder", self)
        save_folder_action.triggered.connect(self.select_save_folder)
        toolbar.addAction(save_folder_action)

        reset_action = QAction("Reset", self)
        reset_action.triggered.connect(self.reset_segmentation)
        toolbar.addAction(reset_action)

        # Add new shape tools
        rect_action = QAction("Rectangle", self)
        rect_action.triggered.connect(lambda: self.set_draw_mode("rectangle"))
        toolbar.addAction(rect_action)

        circle_action = QAction("Circle", self)
        circle_action.triggered.connect(lambda: self.set_draw_mode("circle"))
        toolbar.addAction(circle_action)

        polygon_action = QAction("Polygon", self)
        polygon_action.triggered.connect(lambda: self.set_draw_mode("polygon"))
        toolbar.addAction(polygon_action)

        # Add existing functional tools
        add_action = QAction("Add", self)
        add_action.triggered.connect(lambda: self.set_draw_mode("draw"))
        toolbar.addAction(add_action)

        erase_action = QAction("Erase", self)
        erase_action.triggered.connect(lambda: self.set_draw_mode("erase"))
        toolbar.addAction(erase_action)

        auto_action = QAction("Auto", self)
        auto_action.triggered.connect(lambda: self.set_draw_mode("auto"))
        toolbar.addAction(auto_action)

        # Image Display
        self.imageView = QLabel()
        self.imageView.setAlignment(Qt.AlignCenter)
        self.imageView.setMouseTracking(True)
        self.imageView.mousePressEvent = self.on_mouse_press
        self.imageView.mouseMoveEvent = self.on_mouse_move
        self.imageView.mouseReleaseEvent = self.on_mouse_release
        layout.addWidget(self.imageView)

        # Set central widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    # def initUI(self):
    #     self.setWindowTitle("Auto Labeling Tool - Instance Segmentation")
    #     self.setGeometry(100, 100, 1200, 800)

    #     # Main layout
    #     layout = QVBoxLayout()

    #     # Toolbar
    #     toolbar = QToolBar("Toolbar")
    #     self.addToolBar(toolbar)

    #     load_images_action = QAction("Load Images", self)
    #     load_images_action.triggered.connect(self.load_images)
    #     toolbar.addAction(load_images_action)

    #     save_folder_action = QAction("Select Save Folder", self)
    #     save_folder_action.triggered.connect(self.select_save_folder)
    #     toolbar.addAction(save_folder_action)

    #     help_action = QAction("Help", self)
    #     help_action.triggered.connect(self.show_help)
    #     toolbar.addAction(help_action)

    #     # Image Display
    #     self.imageView = QLabel()
    #     self.imageView.setAlignment(Qt.AlignCenter)
    #     self.imageView.setMouseTracking(True)
    #     self.imageView.mousePressEvent = self.on_mouse_press
    #     self.imageView.mouseMoveEvent = self.on_mouse_move
    #     self.imageView.mouseReleaseEvent = self.on_mouse_release
    #     layout.addWidget(self.imageView)

    #     # Set central widget
    #     container = QWidget()
    #     container.setLayout(layout)
    #     self.setCentralWidget(container)
    
    def set_draw_mode(self, mode):
        """Set the current drawing mode."""
        self.draw_mode = mode
        print(f"Switched to {mode.capitalize()} Mode")

    def reset_segmentation(self):
        """Reset the segmentation and overlay."""
        if self.image is not None:
            self.segmented_image = None
            self.mask_overlay = np.zeros(self.image.shape[:2], dtype=np.uint8)  # Single-channel mask
            print("Segmentation reset.")
            self.update_display()

    def load_images(self):
        """Load all images from a selected folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.image_list = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.image_list.sort()
            self.current_image_index = 0
            self.load_image()

    def select_save_folder(self):
        """Select a folder to save annotations."""
        self.save_folder = QFileDialog.getExistingDirectory(self, "Select Save Folder")
        print(f"Save folder set to: {self.save_folder}")

    def load_image(self):
        """Load the current image."""
        if not self.image_list or self.current_image_index >= len(self.image_list):
            print("No images to load.")
            return

        file_path = self.image_list[self.current_image_index]
        self.image = cv2.imread(file_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.segmented_image = None  # Reset segmented image
        self.mask_overlay = np.zeros(self.image.shape[:2], dtype=np.uint8)  # Single-channel mask
        print(f"Loaded image: {file_path}")
        self.display_image(self.image)

    def display_image(self, img):
        """Display the given image in the GUI, resized to fit within the screen."""
        max_width = self.imageView.width()
        max_height = self.imageView.height()

        height, width, _ = img.shape
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h)
        self.image_scale = scale  # Save scale for coordinate mapping
        new_width = int(width * scale)
        new_height = int(height * scale)

        resized_image = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Convert the resized image to QImage
        bytes_per_line = 3 * new_width
        q_img = QImage(resized_image.data, new_width, new_height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # Display the image in the QLabel
        self.imageView.setPixmap(pixmap)
        self.imageView.setAlignment(Qt.AlignCenter)

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for modes and navigation."""
        super().keyPressEvent(event)
        if event.key() == Qt.Key_Enter and self.draw_mode == "polygon":
            self.close_polygon()
        elif event.key() == Qt.Key_A:
            self.draw_mode = "auto"
            print("Switched to Auto-Segmentation Mode")
        elif event.key() == Qt.Key_D:
            self.draw_mode = "draw"
            print("Switched to Draw Mode")
        elif event.key() == Qt.Key_E:
            self.draw_mode = "erase"
            print("Switched to Erase Mode")
        elif event.key() == Qt.Key_B:
            self.brush_size += 2
            print(f"Increased brush size: {self.brush_size}")
        elif event.key() == Qt.Key_N:
            self.brush_size = max(2, self.brush_size - 2)
            print(f"Decreased brush size: {self.brush_size}")
        elif event.key() == Qt.Key_Left:
            self.current_image_index = max(0, self.current_image_index - 1)
            self.load_image()
        elif event.key() == Qt.Key_Right:
            self.current_image_index = min(len(self.image_list) - 1, self.current_image_index + 1)
            self.load_image()
    
    def map_mouse_to_image(self, event):
        """Map mouse coordinates from the GUI to the original image dimensions."""
        # Get QLabel dimensions
        label_width = self.imageView.width()
        label_height = self.imageView.height()

        # Calculate scaling factors
        img_height, img_width, _ = self.image.shape
        scale_w = label_width / img_width
        scale_h = label_height / img_height
        scale = min(scale_w, scale_h)

        # Calculate padding
        pad_x = (label_width - int(img_width * scale)) // 2
        pad_y = (label_height - int(img_height * scale)) // 2

        # Map mouse position to image coordinates
        x = int((event.pos().x() - pad_x) / scale)
        y = int((event.pos().y() - pad_y) / scale)

        # Clamp coordinates to image dimensions
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))

        return x, y

    def on_mouse_press(self, event):
        """Handle mouse press events."""
        if self.image is None:
            return

        # Map mouse coordinates to original image dimensions
        x, y = self.map_mouse_to_image(event)

        if event.button() == Qt.LeftButton:
            if self.draw_mode == "circle" and self.selected_shape:
                # Check if the click is inside or near the circle
                dist = int(((x - self.selected_shape["center"][0])**2 + (y - self.selected_shape["center"][1])**2)**0.5)
                if abs(dist - self.selected_shape["radius"]) < 10:
                    self.is_resizing = True
                elif dist < self.selected_shape["radius"]:
                    self.is_moving = True
                else:
                    # Deselect if click is outside the shape
                    self.selected_shape = None
            elif self.draw_mode == "circle" and self.selected_shape is None:
                # Start drawing a new circle
                self.start_point = (x, y)
                self.selected_shape = {"type": "circle", "center": (x, y), "radius": 0}


    def on_mouse_move(self, event):
        """Handle mouse move events."""
        if self.image is None:
            return

        # Map mouse coordinates to original image dimensions
        x, y = self.map_mouse_to_image(event)

        if self.start_point and self.draw_mode == "circle" and not (self.is_moving or self.is_resizing):
            # Update the radius of the circle while drawing
            radius = int(((x - self.start_point[0])**2 + (y - self.start_point[1])**2)**0.5)
            self.selected_shape["radius"] = radius
            self.temp_overlay = self.mask_overlay.copy()
            cv2.circle(self.temp_overlay, self.selected_shape["center"], radius, 255, -1)
            self.update_display(temp_overlay=True)
        elif self.is_moving:
            # Clear the old circle and update the mask
            self.mask_overlay = np.zeros_like(self.mask_overlay)  # Clear previous overlay
            dx = x - self.selected_shape["center"][0]
            dy = y - self.selected_shape["center"][1]
            self.selected_shape["center"] = (self.selected_shape["center"][0] + dx, self.selected_shape["center"][1] + dy)
            cv2.circle(self.mask_overlay, self.selected_shape["center"], self.selected_shape["radius"], 255, -1)
            self.update_display()
        elif self.is_resizing:
            # Clear the old circle and update the radius
            self.mask_overlay = np.zeros_like(self.mask_overlay)  # Clear previous overlay
            radius = int(((x - self.selected_shape["center"][0])**2 + (y - self.selected_shape["center"][1])**2)**0.5)
            self.selected_shape["radius"] = radius
            cv2.circle(self.mask_overlay, self.selected_shape["center"], radius, 255, -1)
            self.update_display()


    def on_mouse_release(self, event):
        """Handle mouse release events."""
        if self.image is None:
            return

        # Map mouse coordinates to original image dimensions
        x, y = self.map_mouse_to_image(event)

        if self.draw_mode == "circle":
            if self.is_moving or self.is_resizing:
                # Finalize the manipulation (move/resize)
                self.is_moving = False
                self.is_resizing = False
            elif self.start_point:
                # Finalize the drawing of a new circle
                radius = int(((x - self.start_point[0])**2 + (y - self.start_point[1])**2)**0.5)
                self.selected_shape = {
                    "type": "circle",
                    "center": self.start_point,
                    "radius": radius
                }
                cv2.circle(self.mask_overlay, self.selected_shape["center"], radius, 255, -1)
                self.start_point = None
                self.temp_overlay = None
                self.update_display()
            else:
                # Deselect the circle if mouse release is outside any shape
                self.selected_shape = None


    def modify_mask(self, x, y):
        """Modify the mask using brush or eraser."""
        if self.segmented_image is None:
            print("No mask to modify!")
            return

        color = 1 if self.draw_mode == "draw" else 0
        cv2.circle(self.segmented_image, (x, y), self.brush_size, color, -1)
        cv2.circle(self.mask_overlay, (x, y), self.brush_size, (255 if color == 1 else 0), -1)
        self.update_display()

    def perform_segmentation(self, point):
        """Perform segmentation at the given point."""
        predictor.set_image(self.image)
        input_points = np.array([point], dtype=np.float32)
        input_labels = np.array([1])  # Foreground label

        masks, _, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False
        )
        self.segmented_image = masks[0].astype(np.uint8)
        self.mask_overlay = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.mask_overlay[self.segmented_image > 0] = 255
        self.update_display()

    def update_display(self, temp_overlay=False):
        """Update the displayed image with the segmentation overlay."""
        if temp_overlay and self.temp_overlay is not None:
            overlay = self.image.copy()
            overlay[self.temp_overlay > 0] = [255, 0, 0]  # Red overlay for the mask
        else:
            overlay = self.image.copy()
            overlay[self.mask_overlay > 0] = [255, 0, 0]  # Red overlay for the mask

        blended = cv2.addWeighted(self.image, 0.7, overlay, 0.3, 0)
        self.display_image(blended)


    def save_annotation(self):
        """Save the segmented mask to the save folder."""
        if self.segmented_image is None or not self.save_folder:
            return

        file_name = os.path.basename(self.image_list[self.current_image_index])
        save_path = os.path.join(self.save_folder, f"mask_{file_name}")
        cv2.imwrite(save_path, (self.segmented_image * 255).astype(np.uint8))
        print(f"Annotation saved to: {save_path}")
        
    def close_polygon(self):
        """Complete the polygon by connecting the last point to the first."""
        if len(self.polygon_points) > 2:
            pts = np.array(self.polygon_points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(self.mask_overlay, [pts], 255)
            self.polygon_points.clear()
            self.update_display()

    def show_help(self):
        """Show help instructions."""
        QMessageBox.information(self, "Help", "Shortcuts:\n"
                                              "A: Auto-Segmentation Mode\n"
                                              "D: Draw Mode\n"
                                              "E: Erase Mode\n"
                                              "B: Increase Brush Size\n"
                                              "N: Decrease Brush Size\n"
                                              "Left/Right: Navigate Images\n"
                                              "Rectangle: Draw a rectangular mask\n"
                                              "Circle: Draw a circular mask\n"
                                              "Polygon: Draw a polygon mask (Press Enter to close)\n"
                                              "Reset: Clear current mask.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = AutoLabelingTool()
    mainWin.show()
    sys.exit(app.exec_())
