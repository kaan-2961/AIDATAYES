#!/usr/bin/env python3
# generate_500_node_dataset.py

import os
import sys
import random
import pickle
import pandas as pd

# ── ensure imports work from this directory ───────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# ── external libraries ────────────────────────────────────────────────────────
import osmnx as ox

# ── your helper modules ───────────────────────────────────────────────────────
from set_covering_module import (
    kmeans_clusters,
    nn_clusters,
    rand_nn_clusters,
    set_covering,
    remove_overlapping_nodes,
    get_final_cluster_coordinates,
)
from split_merge_module import optimize_clustering
from feature_extraction_module import compute_features_with_depot

# ── cache file for node pool ─────────────────────────────────────────────────
POOL_PKL = "istanbul_euro_pool.pkl"

def load_or_build_pool():
    """
    Load the cached list of (lat, lon) on Istanbul's European side,
    or download/filter and cache it on first run.
    """
    if os.path.exists(POOL_PKL):
        print("Loading cached pool of nodes…")
        with open(POOL_PKL, "rb") as f:
            pool = pickle.load(f)
    else:
        print("Downloading Istanbul drivable network…")
        G = ox.graph_from_place("Istanbul, Turkey", network_type="drive")
        nodes_gdf, _ = ox.graph_to_gdfs(G)
        euro = nodes_gdf[nodes_gdf["x"] <= 29.03]
        pool = list(zip(euro["y"], euro["x"]))
        print(f"Caching pool of {len(pool)} nodes to {POOL_PKL}…")
        with open(POOL_PKL, "wb") as f:
            pickle.dump(pool, f)
    return pool

def generate_dataset(n_customers=500, reps=5, output_csv="500_node_vrp_dataset.csv"):
    # 1) Load or build the pool of nodes once
    pool = load_or_build_pool()
    if len(pool) < n_customers + 1:
        raise RuntimeError(f"Only {len(pool)} nodes available; need {n_customers+1}.")

    rows = []
    for i in range(reps):
        print(f"\n--- Sample {i+1}/{reps} ---")
        # a) Sample depot + customers
        depot     = random.choice(pool)
        customers = random.sample([pt for pt in pool if pt != depot], n_customers)
        coords    = [depot] + customers

        # b) Feature extraction
        feats = compute_features_with_depot(coords)

        # c) Pre-cluster via set-covering
        cands  = kmeans_clusters(coords, k_range=(10, 35))
        cands += nn_clusters(coords)
        cands += rand_nn_clusters(coords)
        sel    = set_covering(cands, num_customers=len(coords)-1)
        clean  = remove_overlapping_nodes(sel, coords)

        # d) Prepare for split–merge
        nested = get_final_cluster_coordinates(clean, coords)

        # e) Run split–merge to get true k
        init_m, final_m, final_clusters = optimize_clustering(nested, run_id=i+1)
        k_true = final_m[0]  # only the cluster count

        print(f" → true k = {k_true}")

        # f) Record
        row = feats.copy()
        row["k_true"]      = k_true
        row["instance_id"] = f"{n_customers}_{i}"
        rows.append(row)

    # 3) Save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(df)} rows to '{output_csv}'.")

if __name__ == "__main__":
    # You can tweak these parameters:
    N_CUSTOMERS = 500
    REPS        = 700
    OUTPUT_CSV  = "500_node_vrp_dataset.csv"

    generate_dataset(N_CUSTOMERS, REPS, OUTPUT_CSV)
