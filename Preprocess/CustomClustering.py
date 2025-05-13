import numpy as np
from tqdm import tqdm
from numba import jit, prange

class Point:
    def __init__(self, img_num, param_list):
        self.img_num = img_num
        self.param_list = param_list
        self.merged = False
    
    def __repr__(self):
        return f"Point(img_num={self.img_num}, param_list={self.param_list})"
    
    # Method to convert the object to a dictionary
    def to_dict(self):
        return {
            'img_num': self.img_num,
            'param_list': self.param_list,
            'merged': self.merged
        }

    # Static method to create an instance from a dictionary
    @staticmethod
    def from_dict(data):
        param_list = np.array(data['param_list'])
        point = Point(data['img_num'], param_list)
        point.merged = data.get('merged', False)
        return point

@jit(nopython=True, parallel=True)
def compute_distance_matrix(param_lists):
    n = len(param_lists)
    dist_matrix = np.zeros((n, n), dtype=np.float32)
    
    for i in prange(n):  # Use prange for parallel loops
        for j in range(i + 1, n):
            dist = np.linalg.norm(param_lists[i] - param_lists[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist  # Symmetric

    return dist_matrix

def custom_clustering_perpixel2(images: list[np.ndarray], images_dict, min_cluster_size=10, max_cluster_size=15):
    # Create list of Points with flattened images
    points = [Point(img_num=img_name, param_list=images[idx].flatten()) for img_name, idx in tqdm(images_dict.items(), desc="Create list of Points with flattened images")]

    # Extract param_list for all points
    param_lists = np.array([point.param_list for point in tqdm(points, desc="Extract param_list for all points")], dtype=np.float32)

    print("calculating cdist, could take time")
    distance_matrix = compute_distance_matrix(param_lists)

    clusters = {}
    used = set()
    cluster_id = 0

    for i in tqdm(range(len(points)), desc="finding closest images"):
        if i in used:
            continue

        # Start a new cluster
        base_point = points[i]
        cluster = [base_point]
        used.update([i])

        distances = distance_matrix[i]
        closest_indices = np.argsort(distances)[1:]  # skip self

        temp = []
        for idx in closest_indices:
            if len(cluster) >= max_cluster_size:
                break

            candidate_point = points[idx]

            if idx not in used:
                cluster.append(candidate_point)
                temp.append(idx)
        

        if len(cluster) >= min_cluster_size:
            clusters[cluster_id] = cluster
            used.update(temp)
            cluster_id += 1
        
        else:
            clusters[0] = clusters[0] + cluster
            used.update(temp)
    # After the main loop:
    remaining_indices = [i for i in range(len(points)) if i not in used]
    if remaining_indices:
        cluster = []
        for idx in remaining_indices:
            point = points[idx]
            cluster.append(point)

        if len(cluster) > 0:
            clusters[cluster_id] = cluster
    # Replace the param_list of all points with empty lists
    for cluster in clusters.values():
        for point in cluster:
            point.param_list = []

    return clusters
