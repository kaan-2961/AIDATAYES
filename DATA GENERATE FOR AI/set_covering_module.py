import math
import random
import numpy as np
from sklearn.cluster import KMeans
from ortools.linear_solver import pywraplp
import elkai

# ── Global parameters ─────────────────────────────────────────────────────────
TIME_LIMIT    = 3      # max allowed tour time (hours)
SPEED         = 35     # km per hour
SERVICE_TIME  = 0.05   # hours per stop

# ── Utility functions ─────────────────────────────────────────────────────────
def haversine(coord1, coord2):
    """
    Calculate the great‑circle distance (km) between two (lat, lon) pairs.
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371.0
    φ1, λ1 = math.radians(lat1), math.radians(lon1)
    φ2, λ2 = math.radians(lat2), math.radians(lon2)
    dφ = φ2 - φ1
    dλ = λ2 - λ1
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def tsp_solver_elkai(coordinates, customer_indices):
    """
    Solve TSP via elkai. Returns total route time (hrs) + service time.
    """
    if not customer_indices:
        return 0.0

    # round‑trip for single customer
    if len(customer_indices) == 1:
        c = coordinates[customer_indices[0]]
        d = haversine(coordinates[0], c)
        return (2*d)/SPEED + SERVICE_TIME

    # build distance matrix in meters
    idx = [0] + customer_indices
    n = len(idx)
    mat = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                mat[i][j] = int(haversine(coordinates[idx[i]], coordinates[idx[j]]) * 1000)
    # solve
    route = elkai.solve_int_matrix(mat)
    # compute actual float distance
    total_d = 0.0
    for a,b in zip(route, route[1:]):
        total_d += haversine(coordinates[idx[a]], coordinates[idx[b]])
    total_d += haversine(coordinates[idx[-1]], coordinates[idx[0]])
    # add service time
    return (total_d / SPEED) + SERVICE_TIME*len(customer_indices)


def compute_route_time_for_sequence(coordinates, sequence):
    """
    Compute route time (hrs) for a given customer‑index sequence,
    starting/ending at depot (0), including service time.
    """
    route = [0] + sequence + [0]
    t = 0.0
    for u, v in zip(route, route[1:]):
        t += haversine(coordinates[u], coordinates[v]) / SPEED
        if v != 0:
            t += SERVICE_TIME
    return t

# ── Cluster container ─────────────────────────────────────────────────────────
class Cluster:
    """
    Holds:
      - cluster_id: unique int
      - method:     string tag
      - nodes:      list of customer‑indices
      - route_time: float (hrs)
    """
    def __init__(self, cluster_id, method, nodes, route_time):
        self.cluster_id = cluster_id
        self.method     = method
        self.nodes      = nodes
        self.route_time = route_time

    def __repr__(self):
        return (f"Cluster(id={self.cluster_id}, method={self.method}, "
                f"nodes={self.nodes}, time={self.route_time:.2f})")

# ── Clustering methods ─────────────────────────────────────────────────────────
def kmeans_clusters(coordinates, k_range=(10, 35)):
    """
    Run k‑means for k in [10,35] by default, or override via k_range tuple (k_min, k_max).
    Splits clusters whose route time exceeds TIME_LIMIT.
    Returns: list of Cluster objects.
    """
    clusters, cid = [], 0
    cust = np.array(coordinates[1:])
    inds = np.arange(1, len(coordinates))
    for k in range(k_range[0], k_range[1] + 1):
        labels = KMeans(n_clusters=k, random_state=42).fit_predict(cust)
        for lbl in range(k):
            nodes = inds[labels == lbl].tolist()
            rt = tsp_solver_elkai(coordinates, nodes)
            if rt > TIME_LIMIT and len(nodes) > 1:
                # split into two subclusters
                sub_pts = np.array([coordinates[i] for i in nodes])
                sub_inds = np.array(nodes)
                sub_lbl = KMeans(n_clusters=2, random_state=42).fit_predict(sub_pts)
                for s in [0, 1]:
                    sub_nodes = sub_inds[sub_lbl == s].tolist()
                    sub_rt = tsp_solver_elkai(coordinates, sub_nodes)
                    clusters.append(Cluster(cid, f"kmeans_split_{k}", sub_nodes, sub_rt))
                    cid += 1
            else:
                clusters.append(Cluster(cid, f"kmeans_{k}", nodes, rt))
                cid += 1
    return clusters


def nn_clusters(coordinates):
    """
    Greedy nearest‑neighbor clustering until TIME_LIMIT.
    """
    clusters, cid = [], 0
    unassigned    = set(range(1, len(coordinates)))
    while unassigned:
        curr, improved = [], True
        while improved:
            best, best_t = None, None
            for c in list(unassigned):
                t = compute_route_time_for_sequence(coordinates, curr + [c])
                if t <= TIME_LIMIT and (best is None or t < best_t):
                    best, best_t = c, t
            if best is not None:
                curr.append(best)
                unassigned.remove(best)
            else:
                improved = False
        if curr:
            rt = tsp_solver_elkai(coordinates, curr)
            clusters.append(Cluster(cid, "nn", curr, rt))
            cid += 1
        else:
            break
    return clusters


def rand_nn_clusters(coordinates):
    """
    Randomized NN: pick among top‑3 feasible candidates each step.
    """
    clusters, cid = [], 0
    unassigned    = set(range(1, len(coordinates)))
    while unassigned:
        curr, improved = [], True
        while improved:
            cands = []
            for c in list(unassigned):
                t = compute_route_time_for_sequence(coordinates, curr + [c])
                if t <= TIME_LIMIT:
                    cands.append((c, t))
            if cands:
                cands.sort(key=lambda x: x[1])
                pick = random.choice(cands[:3])
                curr.append(pick[0])
                unassigned.remove(pick[0])
            else:
                improved = False
        if curr:
            rt = tsp_solver_elkai(coordinates, curr)
            clusters.append(Cluster(cid, "rand_nn", curr, rt))
            cid += 1
        else:
            break
    return clusters

# ── Set‑covering & cleanup ────────────────────────────────────────────────────
def set_covering(clusters, num_customers):
    """
    Solve minimum‑cluster covering IP.
    clusters: list of Cluster
    num_customers: int (# customers = len(coordinates)-1)
    Returns: list of selected Cluster.
    """
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise RuntimeError("SCIP solver not available")
    x = {c.cluster_id: solver.BoolVar(f"x_{c.cluster_id}") for c in clusters}
    for j in range(1, num_customers + 1):
        cov = [c for c in clusters if j in c.nodes]
        if not cov:
            raise ValueError(f"Customer {j} uncovered")
        solver.Add(sum(x[c.cluster_id] for c in cov) >= 1)
    solver.Minimize(sum(x[c.cluster_id] for c in clusters))
    if solver.Solve() != pywraplp.Solver.OPTIMAL:
        raise RuntimeError("No optimal set‑covering solution found")
    return [c for c in clusters if x[c.cluster_id].solution_value() > 0.5]


def remove_overlapping_nodes(clusters, coordinates):
    """
    Ensure each customer appears in at most one cluster (keep fastest).
    """
    node_map = {}
    for c in clusters:
        for n in c.nodes:
            node_map.setdefault(n, []).append(c)
    for n, lst in node_map.items():
        if len(lst) > 1:
            best = min(lst, key=lambda c: c.route_time)
            for c in lst:
                if c is not best and n in c.nodes:
                    c.nodes.remove(n)
                    c.route_time = (tsp_solver_elkai(coordinates, c.nodes)
                                    if c.method.startswith("kmeans")
                                    else compute_route_time_for_sequence(coordinates, c.nodes))
    return [c for c in clusters if c.nodes]


def get_final_cluster_coordinates(final_clusters, coordinates):
    """
    Returns a list of lists of (lat, lon) for each final cluster.
    """
    return [[coordinates[n] for n in c.nodes] for c in final_clusters]
