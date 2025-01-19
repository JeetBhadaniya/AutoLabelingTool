# AutoLabelingTool

## Overview

AutoLabelingTool is an interactive and user-friendly image annotation and segmentation tool. It combines manual shape-drawing functionalities with automated mask generation using the **Segment Anything Model (SAM)**. The tool is designed for educational purposes and can be easily extended or modified for advanced use cases.

---

## Features

### Manual Annotation
- **Draw Shapes**:
  - Circle, Rectangle, and Polygon drawing with live previews.
  - Modify shapes dynamically by moving or resizing them.
- **Brush Tools**:
  - Add or erase parts of a mask with adjustable brush sizes.
- **Shape Visibility**:
  - Each shape's control points, center, and borders are clearly highlighted for easy editing.

### Automatic Masking
- **Powered by SAM**:
  - Automatically generate segmentation masks with precision.
  - Easily replace SAM with other segmentation models for custom use.

### Mask Management
- Save masks in memory for seamless navigation between images.
- Reload saved masks when revisiting an image.
- Export masks to a specified folder for further use.

### Multi-Image Support
- Load multiple images from a folder for annotation.
- Navigate between images without losing progress.

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/JeetBhadaniya/AutoLabelingTool.git
cd AutoLabelingTool
```
### 2. Install Dependencies
Make sure you have Python installed on your system. Then, install the required packages:
```bash
pip install -r requirements.txt
```
### 3. Download the SAM Model Checkpoint
Download the SAM checkpoint file **"sam_vit_b_01ec64.pth"** from the official SAM repository.
Place the file in the project directory or specify its path in the code.

### 4. Run the Application
Run the following command to start the application:
```bash
python auto_labeling.py
```

## Usage Guide

### Starting the Tool
1. **Launch the Application**:  
   Run the following command in your terminal to start the tool:
   ```bash
   python auto_labeling_tool.py
2. **Load Images**:
    Use the toolbar to select a folder containing the images you want to annotate.

3. **Set Save Folder**:
    Choose a directory where your annotated masks will be saved.


## Tool Features

The **AutoLabelingTool** offers a comprehensive set of features to simplify and enhance the image annotation process. Below is an overview of the key features:

---

### **1. Shape Drawing**
- **Circle**:
  - Click and drag to draw a circle with a live preview.
  - Move the circle by dragging its center after drawing.
  - Resize the circle by dragging its edges.
  - Center and edges are highlighted for clarity.
- **Rectangle**:
  - Click and drag to draw a rectangle with a live preview.
  - Resize or move the rectangle after it is drawn.
- **Polygon**:
  - Click to add vertices for a custom polygon shape.
  - Press `Enter` to close and finalize the polygon.
  - Visible control points allow easy modification.

---

### **2. Brush Tools**
- **Draw Mode**:
  - Add areas to the mask by painting directly on the image.
  - Adjustable brush size for precision.
- **Erase Mode**:
  - Remove unwanted areas from the mask.
  - Brush size adjustments also apply here.

---

### **3. Automatic Mask Generation**
- **Powered by SAM (Segment Anything Model)**:
  - Click anywhere on the image to automatically generate a mask.
  - Can be replaced with custom segmentation models for advanced use cases.

---

### **4. Multi-Image Management**
- **Batch Processing**:
  - Load all images from a folder for sequential annotation.
- **Navigation**:
  - Use the `Left` and `Right` arrow keys to switch between images.
- **Mask Persistence**:
  - Masks are saved in memory when navigating between images.
  - Revisit an image to continue editing its mask.

---

### **5. Mask Export and Saving**
- Export all masks to a user-specified folder.
- Masks are saved with filenames corresponding to their respective images.
- Ensure continuity and reuse with reloaded masks.

---

### **6. Live Feedback**
- Shapes and masks are displayed in real-time during drawing and editing.
- Enhanced visibility for control points, shape edges, and centers.

---

### **7. Reset and Undo**
- Easily reset the current mask using the "Reset" option.
- Start fresh on the current image without affecting others.

---

### **8. User-Friendly Toolbar**
- A well-organized toolbar provides easy access to:
  - Drawing tools: Circle, Rectangle, Polygon
  - Brush tools: Add (Draw) and Erase
  - Automatic Mask Generation
  - Navigation and Reset options
- Help section for quick reference to shortcuts and usage.

---

### **9. Keyboard Shortcuts**
- Speed up your workflow with intuitive keyboard shortcuts for drawing modes, brush adjustments, and navigation.

--- 

The **AutoLabelingTool** combines manual control with automation to offer a flexible and efficient annotation experience.
