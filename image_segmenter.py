import numpy as np
import pandas as pd
import os
import cv2 
import csv
from pathlib import Path

from src.image_ops import manual_closing_grayscale, my_gaussian_blur, my_sobel_edge, get_binary_mask
from src.edge_ops import non_maximum_suppression
from src.hough import hough_line_transform, find_peaks, get_lines
from src.geometry import clip_lines, find_document_corners, clean_intersections, filter_and_cluster_lines
from src.visualization import draw_validated_grid_pil
from src.io import fetch_image_paths

DF_HEADERS = ('filename', 'x1', 'y1', 'x2', 'y2')
CSV_HEADERS = ('filename', 'x1', 'y1', 'x2', 'y2')

def main():
    data_path = './data'
    paths = fetch_image_paths(data_path)

    with open('lines_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADERS)

    if not paths:
        print("No images found.")
        return

    for i in range(len(paths)):
        img_real = cv2.imread(paths[i])
        img_gray = cv2.imread(paths[i], cv2.IMREAD_GRAYSCALE)
        
        path = Path(paths[i])
        file_name = path.name

        img_blurred = my_gaussian_blur(img_gray)

        img_closed = manual_closing_grayscale(img_blurred, kernel_size=15)

        # Hold this for post-processing
        binary_mask = get_binary_mask(img_closed, threshold=230)

        # Edge detection logic
        magnitude, grad_x, grad_y = my_sobel_edge(img_closed)
        nms_res = non_maximum_suppression(magnitude, grad_x, grad_y)
        img_edges = np.zeros_like(nms_res, dtype=np.uint8)
        img_edges[nms_res >= 30] = 255
        
        # Hough line logic
        accumulator, rhos, thetas = hough_line_transform(img_edges)
        peaks = find_peaks(accumulator, rhos, thetas, 50, 50) 
        lines = get_lines(peaks)
        
        if not lines:
            print("No lines found.")
            return

        # Post Processing (Pandas)
        df = pd.DataFrame(data=lines, columns=DF_HEADERS[1:])

        df['x_spread'] = (df['x1'] - df['x2']).abs()
        df['y_spread'] = (df['y1'] - df['y2']).abs()
        horizontal_lines = df[df['x_spread'] > df['y_spread']]
        vertical_lines = df[df['y_spread'] > df['x_spread']]

        # Filter lines by white pixel covered ratio and drop clusters
        final_horizontal_lines = filter_and_cluster_lines(
            horizontal_lines[['x1', 'y1', 'x2', 'y2']].values.tolist(),
            binary_mask,
            'horizontal',
            min_gap=25
        )
        final_vertical_lines = filter_and_cluster_lines(
            vertical_lines[['x1', 'y1', 'x2', 'y2']].values.tolist(),
            binary_mask,
            'vertical',
            min_gap=25
        )

        final_horizontal_lines = pd.DataFrame(final_horizontal_lines, columns=['x1', 'y1', 'x2', 'y2'])
        final_vertical_lines = pd.DataFrame(final_vertical_lines, columns=['x1', 'y1', 'x2', 'y2'])

        intersections = find_document_corners(final_horizontal_lines, final_vertical_lines)
        intersections = clean_intersections(binary_mask, intersections)

        # Saving the final outlined image with intersections
        out_dir = 'annotated_images'
        os.makedirs(out_dir, exist_ok=True)
        img_outlined = draw_validated_grid_pil(
            img_real,
            binary_mask,
            final_horizontal_lines[['x1', 'y1', 'x2', 'y2']],
            final_vertical_lines[['x1', 'y1', 'x2', 'y2']],
            intersections
        )  # Could change return from nparray to the actual image to save it
        img_name = os.path.splitext(file_name)[0]
        new_img_name = f"{img_name}_annotated.jpg"
        save_path = os.path.join(out_dir, new_img_name)
        cv2.imwrite(save_path, img_outlined)

        # Saving the final outputs to csv
        combined_lines = pd.concat([final_horizontal_lines, final_vertical_lines])
        final_df = combined_lines[['x1', 'y1', 'x2', 'y2']].copy()
        final_df = clip_lines(final_df, img_real)
        final_df.insert(0, 'filename', file_name)
        final_df.to_csv(
            'lines_data.csv',
            mode='a',
            header=False,
            index=False
        )

if __name__ == "__main__":
    main()