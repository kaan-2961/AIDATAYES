import numpy as np
import elkai
from sklearn.cluster import KMeans
import math
import random
import copy
import matplotlib.cm as cm
import pandas as pd

# Global parameters
speed_km_per_hr = 35
service_time_hr = 0.05
tmax = 3
hiring_cost_per_cluster = 50
distance_cost_per_km = 2
max_merge_attempts_per_cluster = 20
max_iterations = 100

# Utility functions
def haversine(coord1, coord2):
    """
    Calculate the Haversine distance (in km) between two (lat, lon) points.
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.asin(math.sqrt(a))

class Cluster:
    def __init__(self, data_points, cluster_id=None, color=None):
        self.data_points = np.array(data_points)
        self.id = cluster_id
        self.time = None
        self.cost = None
        self.tour = None
        self.total_distance = None
        self.merge_attempts_remaining = max_merge_attempts_per_cluster
        self.attempts_left = 4
        self.nearest_cluster_ids = None
        self.color = color if color is not None else np.random.rand(3,)

    def centroid(self):
        return np.mean(self.data_points, axis=0) if self.data_points.size else np.array([0.0,0.0])

    def set_tsp(self, tour, dist, time):
        self.tour = tour
        self.total_distance = dist
        self.time = time
        self.cost = hiring_cost_per_cluster + dist * distance_cost_per_km

    def calculate_nearest_clusters(self, clusters, k=3):
        dists = [(haversine(self.centroid(), c.centroid()), c.id) for c in clusters if c.id != self.id]
        dists.sort(key=lambda x: x[0])
        self.nearest_cluster_ids = [cid for _,cid in dists[:k]]

# Core TSP solver

def solve_tsp(points):
    n = len(points)
    if n < 2:
        return None, 0.0, 0.0
    if n == 2:
        d = haversine(points[0], points[1])
        t = d/speed_km_per_hr + 2*service_time_hr
        return [0,1], d, t
    # distance matrix in meters
    M = np.zeros((n,n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i!=j:
                M[i,j] = int(haversine(points[i], points[j]) * 1000)
    tour = elkai.solve_int_matrix(M)
    d = sum(haversine(points[tour[i]], points[tour[i+1]]) for i in range(len(tour)-1))
    t = d/speed_km_per_hr + n*service_time_hr
    return tour, d, t

# Merge iterations
def get_valid(clusters):
    return [c for c in clusters if c.time is not None and c.time <= tmax]

def run_merges(clusters):
    random.shuffle(clusters)
    for base in clusters[:]:
        if base.merge_attempts_remaining <= 0 or base.attempts_left <= 0:
            continue
        base.calculate_nearest_clusters(clusters)
        for cid in (base.nearest_cluster_ids or []):
            cand = next((c for c in clusters if c.id == cid), None)
            if cand:
                merged_pts = np.vstack((base.data_points, cand.data_points))
                tour, d, t = solve_tsp(merged_pts.tolist())
                if tour and t <= tmax:
                    base.data_points = merged_pts
                    base.set_tsp(tour, d, t)
                    clusters.remove(cand)
                    break
        base.attempts_left -= 1
    return get_valid(clusters)

# Single-run optimizer
def optimize_clustering(points_list, run_id=1):
    # initialize clusters from point lists
    clusters = []
    for i, pts in enumerate(points_list):
        cl = Cluster(pts, cluster_id=f"{run_id}_{i}")
        tour, d, t = solve_tsp(pts)
        if tour and t <= tmax:
            cl.set_tsp(tour, d, t)
            clusters.append(cl)
    current = clusters
    for _ in range(max_iterations):
        current = run_merges(current)
    final = get_valid(current)
    init_metrics = (len(clusters), sum(c.cost for c in clusters))
    final_metrics = (len(final), sum(c.cost for c in final))
    return init_metrics, final_metrics, final

# Best-of-N wrapper

def optimize_clustering_best(points_list, first_runs=15, second_runs=10):
    """
    1) Run optimize_clustering first_runs times, pick best
    2) Take best clusters, run optimize_clustering second_runs times on them
    """
    # First phase
    best_cost1 = float('inf')
    best_clusters1 = None
    for rid in range(1, first_runs+1):
        _, fm, clusters1 = optimize_clustering(points_list, run_id=rid)
        if fm[1] < best_cost1:
            best_cost1 = fm[1]
            best_clusters1 = clusters1
    # Prepare second phase input
    pts2 = [c.data_points.tolist() for c in best_clusters1]
    # Second phase
    best_cost2 = float('inf')
    best_clusters2 = None
    for rid in range(1, second_runs+1):
        _, fm2, clusters2 = optimize_clustering(pts2, run_id=rid)
        if fm2[1] < best_cost2:
            best_cost2 = fm2[1]
            best_clusters2 = clusters2
    # Final metrics
    final_metrics = (len(best_clusters2), best_cost2)
    return best_clusters2, final_metrics

# End of module
