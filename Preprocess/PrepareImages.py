import cv2
import os
import numpy as np
from tqdm import tqdm
from PIL import Image

def load_images_from_folder(folder,colored):
    images = []
    images_index_dict = {}
    
    for filename in tqdm(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder,filename)) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if not colored:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if img is not None:
            images.append(img)
            
            images_index_dict[filename.split(".")[0]] = len(images)-1
    return (images, images_index_dict)

def load_images_from_folder(folder, colored, filter_filenames=None):
    images = []
    images_index_dict = {}
    
    for filename in tqdm(os.listdir(folder)):
        # Skip files not in filter list (if filter is active)
        if filter_filenames and filename not in filter_filenames:
            continue

        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            continue  # Skip unreadable image
        
        # Convert to RGB or Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if not colored:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        images.append(img)
        images_index_dict[filename.split(".")[0]] = len(images) - 1

    return images, images_index_dict

def resize_and_save_images(input_folder, output_folder, batch_size=100, target_size=(512, 512)):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get the list of image filenames in the input folder
    image_filenames = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Process images in batches
    for i in tqdm(range(0, len(image_filenames), batch_size), desc="Resizing images", unit="batch"):
        batch_filenames = image_filenames[i:i + batch_size]
        batch_images = []

        # Load and resize each image in the batch
        for filename in batch_filenames:
            img = cv2.imread(os.path.join(input_folder, filename))            
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                resized_img = np.array(Image.fromarray(img_rgb).resize(target_size, Image.BICUBIC))
                
                batch_images.append(resized_img)

        # Save the resized images to the output folder
        for resized_img, filename in zip(batch_images, batch_filenames):
            # Ensure filename has .png extension
            base, ext = os.path.splitext(filename)
            if ext.lower() != ".png":
                filename = base + ".png"

            output_path = os.path.join(output_folder, filename)
            
            cv2.imwrite(output_path, cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR))