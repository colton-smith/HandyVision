""" reverse_images.py

Read all images in input folder, reflect in X, write to input_folder/flipped
"""
import sys
import os
import cv2 as cv


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Invalid number of args. Provide ONLY input folder path and output folder path")
        exit()
    
    folder = sys.argv[1]
    folder = os.path.normpath(folder)
    output_folder = os.path.join(folder, "flipped")

    output_folder_exists = os.path.exists(output_folder)
    if not output_folder_exists:
        os.makedirs(output_folder)

    print(f"--- Input folder: {folder}")
    print(f"--- Output folder: {output_folder}")

    # Load all filenames in the folder 
    input_files = []
    for _, _, files in os.walk(folder):
        for f in files:
            input_files.append(f)
        break

    # Reverse each of the input files
    for fin in input_files: 
        input_path = os.path.join(folder, fin)
        output_path = os.path.join(output_folder, fin)
    
        print(f"Input image: {input_path}")
        print(f"Output image: {output_path}")
        print("---")

        img_in = cv.imread(input_path, cv.IMREAD_UNCHANGED)
        img_out = cv.flip(img_in, 1)
        cv.imwrite(output_path, img_out)
