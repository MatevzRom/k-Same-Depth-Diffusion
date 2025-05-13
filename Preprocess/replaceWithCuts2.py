import os
import json
from PIL import Image
from tqdm import tqdm
from . import PrepareImages
from .k_anonymity_depth import recalculate_depth
from .CustomClustering import Point

def replaceWithCuts2(json_path, scale, crop_box):
    image1_path = "./temp/datasetResized"
    facecut_path = "./temp/cuts"
    destination = "./temp/k-anonymity/"
    os.makedirs(destination, exist_ok=True)

# Load cluster definitions from JSON
    print("Loading cluster JSON...")
    with open(json_path, 'r') as f:
        loaded_clusters_dict = json.load(f)
    print("JSON loaded.")

    clusters = {
        int(key): [Point.from_dict(point_data) for point_data in points]
        for key, points in loaded_clusters_dict.items()
    }
    print("Clusters reconstructed.")
    # crop_coords = (100, 140, 472, 384)  # (y1, x1, y2, x2)
    if crop_box == [-1,-1,-1,-1]:
        crop_box = (35*4, 25*4, 96*4, 118*4)  # (left, upper, right, lower) (140, 100, 384, 472)
        crop_box = tuple(int(v * scale) for v in crop_box)
    else:
        crop_box = list(crop_box)

    # Process clusters one by one
    for cluster_id, cluster in tqdm(clusters.items(), desc="Processing clusters"):
        image_names = [point.img_num for point in cluster]  # <-- FIXED HERE
        filtered_filenames = [name + ".png" for name in image_names]  # Only PNGs

        # Load selected images from both folders
        imagesCut, imagesCut_index_dict = PrepareImages.load_images_from_folder(
            facecut_path, colored=True, filter_filenames=filtered_filenames
        )
        images1, images1_index_dict = PrepareImages.load_images_from_folder(
            image1_path, colored=True, filter_filenames=filtered_filenames
        )

        # Recalculate depth for this cluster
        recalculate_depth(imagesCut, imagesCut_index_dict, {cluster_id: cluster})

        # Replace cropped region and save
        for name in image_names:
            if name not in imagesCut_index_dict or name not in images1_index_dict:
                continue

            orig_img = images1[images1_index_dict[name]]
            facecut_img = imagesCut[imagesCut_index_dict[name]]

            orig_img[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]] = facecut_img

            save_path = os.path.join(destination, name + ".png")
            Image.fromarray(orig_img).save(save_path)
        
