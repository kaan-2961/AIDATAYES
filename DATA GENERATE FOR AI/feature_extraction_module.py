import math
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import squareform
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import skew

# ── Haversine distance ────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    """
    Compute great-circle distance between two (lat, lon) points in km.
    """
    φ1, λ1 = map(math.radians, (lat1, lon1))
    φ2, λ2 = map(math.radians, (lat2, lon2))
    dφ = φ2 - φ1
    dλ = λ2 - λ1
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def haversine_matrix(coords):
    """
    Build full pairwise Haversine distance matrix for coords (array of [lat, lon]).
    """
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = haversine(coords[i,0], coords[i,1], coords[j,0], coords[j,1])
            D[i,j] = D[j,i] = d
    return D

# ── Feature extractor ─────────────────────────────────────────────────────────
def compute_features_with_depot(coords):
    """
    coords: list or array of (lat, lon) pairs where coords[0] is the depot.
    Returns dict of 22 features.
    """
    coords = np.array(coords)
    depot  = coords[0]
    cust   = coords[1:]
    n = len(cust)

    # 1) Node count
    node_count = n

    # 2–3) Convex hull of customers (lon,lat) for planar area & perimeter
    pts2d = cust[:, [1, 0]]
    hull = ConvexHull(pts2d)
    area = hull.volume       # 2D area
    perimeter = hull.area    # 2D perimeter

    # 4) Aspect ratio (width/height)
    w = pts2d[:,0].ptp()
    h = pts2d[:,1].ptp()
    aspect = w/h if h else 0

    # 5) Convex hull area ratio
    bbox_area = w*h
    ch_area_ratio = area/bbox_area if bbox_area else 0

    # 6–7) Std dev in lon/lat degrees
    std_x = np.std(pts2d[:,0])
    std_y = np.std(pts2d[:,1])

    # 8–9) Haversine dist from centroid
    cen_lat, cen_lon = cust.mean(axis=0)
    d_cent = [haversine(lat, lon, cen_lat, cen_lon) for lat, lon in cust]
    avg_cent = np.mean(d_cent)
    max_cent = np.max(d_cent)

    # 10) Full pairwise matrix
    D = haversine_matrix(cust)
    pair_d = D[np.triu_indices(n, k=1)]

    # 11) Density
    density = n/area if area else 0

    # 12–13) Min/Max neighbor dist
    with np.errstate(invalid='ignore'):
        nn = np.where(D==0, np.nan, D)
    row_min = np.nanmin(nn, axis=1)
    min_nn = np.nanmin(row_min)
    max_nn = np.nanmax(row_min)

    # 14) Avg pairwise dist
    avg_pw = np.mean(pair_d)

    # 15) MST length
    mst = minimum_spanning_tree(D)
    mst_len = mst.sum()

    # 16) MST diameter
    dist_mst = dijkstra(mst, directed=False)
    mst_diam = np.nanmax(dist_mst[np.isfinite(dist_mst)])

    # 17) Silhouette score over k=2..min(10,n)
    best_sil = -1
    for k in range(2, min(10, n)):
        labels = KMeans(n_clusters=k, random_state=0).fit_predict(cust)
        sil = silhouette_score(D, labels, metric='precomputed')
        best_sil = max(best_sil, sil)

    # 18) Entropy of distances
    probs = pair_d/pair_d.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-12))

    # 19) Variance of pairwise
    var_pw = np.var(pair_d)

    # 20) Skewness of pairwise
    skew_pw = skew(pair_d)

    # 21) MST/TSP ratio (naive TSP = sum each row's 2nd smallest edge)
    tsp_est = np.sum(np.sort(D, axis=1)[:,1])
    mst_tsp = mst_len/tsp_est if tsp_est else 0

    # 22–23) Depot distances
    d_dep = [haversine(depot[0], depot[1], lat, lon) for lat, lon in cust]
    avg_dep = np.mean(d_dep)
    max_dep = np.max(d_dep)

    return {
        'Node Count':         node_count,
        'Area':               area,
        'Aspect Ratio':       aspect,
        'Convex Hull Area Ratio': ch_area_ratio,
        'Std Dev X':          std_x,
        'Std Dev Y':          std_y,
        'Avg Dist from Centroid': avg_cent,
        'Max Dist from Centroid': max_cent,
        'Density':            density,
        'Min NN Dist':        min_nn,
        'Max NN Dist':        max_nn,
        'Avg Pairwise Dist':  avg_pw,
        'MST Length':         mst_len,
        'MST Diameter':       mst_diam,
        'Silhouette Score':   best_sil,
        'Entropy of Distances': entropy,
        'Variance of Pairwise Distances': var_pw,
        'Skewness of Pairwise Distances': skew_pw,
        'MST/TSP Ratio':      mst_tsp,
        'Avg Depot Dist':     avg_dep,
        'Max Depot Dist':     max_dep
    }
