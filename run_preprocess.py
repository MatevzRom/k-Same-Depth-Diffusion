import argparse
import os
from Preprocess.PrepareImages import resize_and_save_images
from Preprocess.generateCuts import generateCuts
from Preprocess.generateDepth import generateDepth
from Preprocess.find_clusters import findClusters
from Preprocess.replaceWithCuts2 import replaceWithCuts2

# ADD FACE BOX RESHAPE

def main():
    parser = argparse.ArgumentParser(description="Preprocess image dataset.")
    parser.add_argument('--input', type=str, default="./testSet", help='Path to input image folder')
    parser.add_argument('--genRes', type=int, default= 512, help='Resolution for generated images')
    parser.add_argument('--clustRes', type=int, default= 512, help='Image resolution used in clustering')
    parser.add_argument('--minClustSize', type=int, default= 3, help='Minimum cluster size')
    parser.add_argument('--maxClustSize', type=int, default= 4, help='Maximum cluster size')
    parser.add_argument('--faceBox', type=int, default= 4, help='Face box coordinates ')
    parser.add_argument('--crop_box', type=int, default = [-1, -1, -1, -1], nargs=4, metavar=('x1', 'y1', 'x2', 'y2'), help='Crop box in format: x1 y1 x2 y2')

    args = parser.parse_args()

    input_folder = args.input
    output_folder = r"./temp/datasetResized"
    json_path = './temp/testJson.json'

    crop_box = args.crop_box
    scale = args.genRes / 512

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    resize_and_save_images(input_folder, output_folder, batch_size=100, target_size=(args.genRes, args.genRes))
    if args.genRes != args.clustRes:
        output_folder_clust = r"./temp/datasetResizedClust/"
        os.makedirs(output_folder_clust, exist_ok=True)
        resize_and_save_images(input_folder, output_folder_clust, batch_size=100, target_size=(args.clustRes, args.clustRes))
    else:
        output_folder_clust = output_folder
    
    generateCuts(output_folder, scale, crop_box)

    generateDepth(output_folder)

    print(f"Finding clusters and saving to {json_path}")
    findClusters(output_folder_clust, json_path, args.minClustSize, args.maxClustSize)

    replaceWithCuts2(json_path, scale, crop_box)

    print(f"✅ Preprocessing complete. Results saved to ./temp/k-anonymity and {json_path}")

if __name__ == "__main__":
    main()
