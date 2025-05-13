from transformers import pipeline
from PIL import Image
import os
import glob
from tqdm import tqdm

def generateDepth(input_folder):
    # Load pipeline
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf")
    print("Pipeline loaded :)")

    # Initialize image lists and paths
    image_list = []
    image_names_list = []

    result_folder_depth = os.path.join("./temp/", "depth")
    os.makedirs(result_folder_depth, exist_ok=True)

    batch_size = 1000  # Set batch size to 1000
    counter = 0
    ccounter = 0
    print("Starting to read images")

    # Loop through images in the specified directory
    for file_path in tqdm(glob.glob(input_folder+"/*"), desc="Generating depth images"):
        im = Image.open(file_path)
        
        # Get file name
        file_name = os.path.basename(file_path).split(".")[0]
        image_list.append(im)
        
        save_path_depth = os.path.join(result_folder_depth, file_name + ".png")
        image_names_list.append(save_path_depth)

        # If batch size is reached, process and save the images
        if counter >= batch_size:
            depth_list = pipe(image_list)  # Run depth estimation on batch
            for index, item in enumerate(depth_list):
                depth = item["depth"]
                depth.save(image_names_list[index])  # Save depth map
            print(f"Saved batch {ccounter}")
            
            # Reset for next batch
            image_list = []
            image_names_list = []
            counter = 0
            ccounter += 1
            continue

        counter += 1
        

    # Final batch processing if not processed in the loop (even if it's smaller than 1000)
    if image_list:
        depth_list = pipe(image_list)
        for index, item in enumerate(depth_list):
            depth = item["depth"]
            depth.save(image_names_list[index])
        print("Saved final batch")