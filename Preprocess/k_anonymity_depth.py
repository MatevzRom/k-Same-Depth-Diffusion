import numpy as np
from .CustomClustering import Point
from tqdm import tqdm

def recalculate_depth(images_depth,images_depth_index_dict, clusters: list[Point]):
    for key in tqdm(clusters, desc = "recalculating depth:"):
        img_cluster = []

        # Calculate cluster mean in uint8
        for point in clusters[key]:
            img_cluster.append(images_depth[images_depth_index_dict[point.img_num]])
        img_cluster = np.array(img_cluster)
        avg = np.mean(img_cluster,axis=0)
        avg_uint8 = np.clip(avg, 0, 255).astype(np.uint8)

        # Swap depth for average
        for point in clusters[key]:
            images_depth[images_depth_index_dict[point.img_num]] = avg_uint8
    return