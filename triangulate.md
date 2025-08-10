# Image Triangulation Script (with Face Detection)

This Python script transforms a standard image into a "low-poly" or triangulated version. It analyzes an image, breaks it down into a mesh of triangles, and then recolors them to create stylized artwork.

A key feature of this script is its ability to **detect human faces** and apply a finer-grained, more detailed triangulation to those areas, while using larger triangles for the background and other parts of the image.

The script generates two distinct output images:

1.  **Average Color Version**: Each triangle is filled with the average color of the pixels it covered in the original image. This results in a detailed, mosaic-like effect.
2.  **Clustered Color Version**: The script first identifies a small, representative palette of colors from the image. Each triangle is then filled with the color from this palette that is closest to its original average color. This creates a more graphic, posterized look with a limited color scheme.

---

## Requirements

The script relies on several popular Python libraries and an OpenCV data file.

* **Python Libraries**:
    * OpenCV-Python
    * NumPy
    * SciPy
    * Scikit-learn
* **OpenCV Data File**:
    * A Haar Cascade XML file for face detection (e.g., `haarcascade_frontalface_default.xml`).

---

## Installation

1.  **Install Python Libraries**: Before running the script, you need to install the required libraries. You can install them all with a single command using `pip`:
    ```bash
    pip install opencv-python numpy scipy scikit-learn
    ```

2.  **Download the Haar Cascade File**: To use the face detection feature, you must download a pre-trained classifier.
    * **Download Link**: [haarcascade_frontalface_default.xml](https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml)
    * **Placement**: Save this XML file in the same directory where you saved the Python script.

---

## Usage

Run the script from your command line, providing the path to your input image as the main argument.

### Basic Command (Without Face Detection)

```bash
python triangulate.py path/to/your/image.jpg
