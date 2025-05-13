from tqdm import tqdm
import json
from . import PrepareImages
from . import CustomClustering

def findClusters(input_folder, json_path, min_cluster_size, max_cluster_size):
    imagesCut, imagesCut_index_dict = PrepareImages.load_images_from_folder(input_folder,True)

    clusters = CustomClustering.custom_clustering_perpixel2(imagesCut,imagesCut_index_dict, min_cluster_size, max_cluster_size)
    clusters_dict = {key: [point.to_dict() for point in value] for key, value in tqdm(clusters.items(),desc = "converting to dict for json")}
    
    with open(json_path, 'w') as f:
        json.dump(clusters_dict, f)
