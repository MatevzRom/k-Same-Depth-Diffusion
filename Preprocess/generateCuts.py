import os
from PIL import Image
from tqdm import tqdm

def generateCuts(input_folder, scale, crop_box):
    output_folder = r'./temp/cuts'  # Path to the folder where cropped images will be saved

    # Make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Define the crop coordinates: [100:470, 140:384]
    if crop_box == [-1,-1,-1,-1]:
        crop_box = (35*4, 25*4, 96*4, 118*4)  # (left, upper, right, lower) (140, 100, 384, 472)
        crop_box = tuple(int(v * scale) for v in crop_box)
    else:
        crop_box = list(crop_box)
    for filename in tqdm(os.listdir(input_folder), desc= " Generating Cuts"):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Only process image files
            img_path = os.path.join(input_folder, filename)
            # Create the new filename with .png extension
            output_img_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.png')

            # Open the image
            with Image.open(img_path) as img:
                # Crop the image
                cropped_img = img.crop(crop_box)
                
                # Save the cropped image as PNG
                cropped_img.save(output_img_path, format='PNG')
    print("Image cropping process completed!")