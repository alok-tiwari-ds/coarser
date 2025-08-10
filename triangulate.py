import cv2
import numpy as np
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans
import argparse
import os
import sys

def process_image(
    image_path: str,
    max_points: int = 750,
    num_colors: int = 12,
    min_distance: int = None,
    haar_cascade_path: str = None,
    face_points: int = 500,
    face_min_dist: int = 5
):
    """
    Processes an image by triangulating it, with special attention to adding
    more detail to human faces if a Haar Cascade is provided.

    Args:
        image_path (str): The full path to the input image.
        max_points (int): Max feature points for the overall image.
        num_colors (int): The number of distinct colors for the final palette.
        min_distance (int, optional): Min distance between general feature points.
        haar_cascade_path (str, optional): Path to the Haar Cascade XML for face detection.
        face_points (int): Max feature points to add specifically to faces.
        face_min_dist (int): Min distance between feature points on faces.
    """
    # --- 1. Load Image ---
    print(f"⏳ Loading image: {os.path.basename(image_path)}...")
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not open or find the image at '{image_path}'", file=sys.stderr)
        sys.exit(1)

    height, width = original_image.shape[:2]
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    all_points = []

    # --- 2. Detect Faces and Add Dense Points (if cascade is provided) ---
    if haar_cascade_path:
        print(f"🔎 Loading face detector: {os.path.basename(haar_cascade_path)}...")
        face_cascade = cv2.CascadeClassifier(haar_cascade_path)
        if face_cascade.empty():
            print(f"Error: Could not load Haar Cascade from '{haar_cascade_path}'", file=sys.stderr)
        else:
            faces = face_cascade.detectMultiScale(gray_image, 1.1, 5)
            print(f"🙂 Found {len(faces)} face(s). Adding extra detail.")
            for (x, y, w, h) in faces:
                face_roi = gray_image[y:y+h, x:x+w]
                # Find dense points within the face region
                face_feature_points = cv2.goodFeaturesToTrack(
                    image=face_roi,
                    maxCorners=face_points,
                    qualityLevel=0.01,
                    minDistance=face_min_dist
                )
                if face_feature_points is not None:
                    # Offset points to their correct position in the full image
                    face_feature_points[:, 0, 0] += x
                    face_feature_points[:, 0, 1] += y
                    all_points.append(face_feature_points)

    # --- 3. Find General Feature Points for the Whole Image ---
    print("🔎 Finding general feature points for the whole image...")
    if min_distance is None:
        min_dist_heuristic = int((width + height) / 70)
        print(f"ℹ️ Minimum distance between points calculated automatically: {min_dist_heuristic}px")
    else:
        min_dist_heuristic = min_distance
        print(f"ℹ️ Using user-specified minimum distance: {min_dist_heuristic}px")

    general_points = cv2.goodFeaturesToTrack(
        image=gray_image,
        maxCorners=max_points,
        qualityLevel=0.01,
        minDistance=min_dist_heuristic
    )
    if general_points is not None:
        all_points.append(general_points)
    
    if not all_points:
        print("Error: Could not find any feature points in the image.", file=sys.stderr)
        sys.exit(1)

    # Combine all points (general + face points) into one array
    points = np.vstack(all_points).astype(int).reshape(-1, 2)

    # Add the four corners of the image to ensure the triangulation covers the entire area.
    corner_points = np.array([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])
    points = np.vstack([points, corner_points])

    # --- 4. Perform Triangulation ---
    print(f"⏳ Performing triangulation on {len(points)} points...")
    delaunay = Delaunay(points)
    triangles = delaunay.simplices
    print(f"✔️ Generated {len(triangles)} triangles.")

    # --- 5. Extract and Cluster Colors ---
    print("🎨 Extracting and clustering triangle colors...")
    avg_colors = []
    triangle_to_avg_color = {}

    for i, triangle in enumerate(triangles):
        vertices = points[triangle]
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [vertices], 255)
        mean_color = cv2.mean(original_image, mask=mask)[:3]
        avg_colors.append(mean_color)
        triangle_to_avg_color[i] = mean_color

    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
    kmeans.fit(avg_colors)
    palette = kmeans.cluster_centers_

    triangle_to_clustered_color = {}
    for i, label in enumerate(kmeans.labels_):
        triangle_to_clustered_color[i] = palette[label]
    print(f"🎨 Palette of {num_colors} distinct colors created.")

    # --- 6. Generate and Save Output Images ---
    print("🖼️ Generating output images...")
    output_avg_color = np.zeros_like(original_image)
    output_clustered_color = np.zeros_like(original_image)

    for i, triangle in enumerate(triangles):
        vertices = points[triangle]
        avg_color = tuple(map(int, triangle_to_avg_color[i]))
        cv2.fillPoly(output_avg_color, [vertices], avg_color)
        clustered_color = tuple(map(int, triangle_to_clustered_color[i]))
        cv2.fillPoly(output_clustered_color, [vertices], clustered_color)

    base_name, ext = os.path.splitext(image_path)
    output_filename_avg = f"{base_name}_triangulated_average{ext}"
    output_filename_clustered = f"{base_name}_triangulated_clustered{ext}"

    cv2.imwrite(output_filename_avg, output_avg_color)
    cv2.imwrite(output_filename_clustered, output_clustered_color)

    print("\n✅ Success! Output images saved:")
    print(f"  - {os.path.basename(output_filename_avg)}")
    print(f"  - {os.path.basename(output_filename_clustered)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        A Python script to create a triangulated, low-polygon version of an image.
        It can be configured to add extra detail to human faces.
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("image_path", type=str, help="Path to the input image file.")
    
    # General arguments
    parser.add_argument("-p", "--points", type=int, default=750, help="Maximum number of points for the general image (default: 750).")
    parser.add_argument("-c", "--colors", type=int, default=12, help="Number of colors in the final palette (default: 12).")
    parser.add_argument("-d", "--min-dist", type=int, default=None, help="Minimum distance in pixels between general points. Overrides automatic calculation.")
    
    # Face-specific arguments
    parser.add_argument("--haar-cascade", type=str, default=None, help="Path to the Haar Cascade XML file for face detection.")
    parser.add_argument("--face-points", type=int, default=500, help="Maximum number of points to add to each detected face (default: 500).")
    parser.add_argument("--face-min-dist", type=int, default=5, help="Minimum distance in pixels between points on faces (default: 5).")

    args = parser.parse_args()
    process_image(
        args.image_path,
        args.points,
        args.colors,
        args.min_dist,
        args.haar_cascade,
        args.face_points,
        args.face_min_dist
    )

