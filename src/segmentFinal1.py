########################################################################
# LiDAR Project - Box Segment verification via hdbscan, bimodal split,
# and RANSAC model fitting in polar coodrinates
#########################################################################


import numpy as np
import hdbscan
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.exceptions import UndefinedMetricWarning
import matplotlib.pyplot as plt
import warnings
import time
from polar_segment_generator import generate_polar_segments

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)



# =======================================================
#                 CONTROL PANEL
#
# This dictionary contains all tunable parameters for the
# segmentation pipeline.
# =======================================================

PARAMS = {
# - RANSAC: min_samples, residual_threshold, max_trials, linearity_threshold

    "min_samples": 8, # adjust when when a point density is known

    "residual_threshold": 0.01, # max allowed distance from the line to count as an inlier.

    "max_trials": 200, # increasing will increase computation time but may improve results

    "linearity_threshold": 0.005, # Currently very low as my sample data is made by arcs

# - Segment filters: min/max length in meters

# - adust for expected box size
    "min_segment_length": 0.20,
    "max_segment_length": 0.9,

# - HDBSCAN: cluster granularity (min_cluster_size, min_samples)

    "hdbscan_min_cluster_size": 8, # matched to RANSAC min_samples

    "hdbscan_min_samples": 5, # smaller yields more clusters


# - Bimodal split: DBSCAN in theta-space (eps in radians, min_samples)

    "bimodal_eps_theta": 0.02, # adjust based on expected space between segments

    "bimodal_min_samples": 5,

# - Radius gate: will allow us to have a set distance range from LiDAR
    "radius_min": 1.0,
    "radius_max": 3.0,

# - Expected box Location (If we key of barcode detection)
    "angle_center_deg": -90,
    "angle_halfwidth_deg": 180,
}


# =======================================================
#                     TIMER WRAPPER
# =======================================================

def timed(label):
    class Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            print(f"\n--- {label} START ---")
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            end = time.perf_counter()
            print(f"--- {label} END: {end - self.start:.4f} s ---\n")
    return Timer()

# =======================================================
#                 BIMODAL SPLITTER
# theta = angle array in radians-defines where points are
# located relative to LiDAR
# rr = radius array in meters-defines how far points are from LiDAR
# P = PARAMS dictionary
# =======================================================

def split_cluster_if_bimodal(theta, rr, P):
    # Exit early if note enough points to make two clusters
    if len(theta) < 2 * P["bimodal_min_samples"]: 
        return [(theta, rr)]
    # Reshape theta array for DBSCAN. Makes it a 2D array with one column
    X = theta.reshape(-1, 1)
    #Density-Based Spatial Clustering
    db = DBSCAN(
        eps=P["bimodal_eps_theta"],
        min_samples=P["bimodal_min_samples"]
    )
    labels = db.fit_predict(X) #fit model and predict clusters

    # if 0 or 1 clusters found, return orignal data
    unique = sorted(set(labels)) 
    if len(unique) <= 1:
        return [(theta, rr)]

    # Build subclusters from DBSCAN results
    subclusters = []
    for lab in unique:
        if lab == -1: # noise labeled  -1 by DBSCAN
            continue
        # get points in this subcluster
        mask = labels == lab
        th_sub = theta[mask]
        rr_sub = rr[mask]
        # only keep subclusters with enough points
        if len(th_sub) >= P["bimodal_min_samples"]:
            subclusters.append((th_sub, rr_sub))
    # if no valid subclusters found, return original data
    if len(subclusters) <= 1:
        return [(theta, rr)]
    
    return subclusters # return the subclusters found

# =======================================================
#                     RANSAC
# theta = angle array in radians-defines where points are
# r - radius array
# P = PARAMS dicationary
# lablel = debug label for print statements
# =======================================================

def fit_RANSAC(theta, r, P, label=""):

    theta_std = np.std(theta) #standard deviation of angle values
    r_std = np.std(r) # standard deviation of radius values

# If angle variance is higher, use horizontal model, else vertical
    if theta_std > r_std:
        model_type = "Horizontal"
    else:
        model_type = "Vertical"

    print(f"\n[{label}] Model selected: {model_type.upper()}")

    # ---------------------------------------------------
    # Horizontal model
    # ---------------------------------------------------
    if model_type == "Horizontal":

        model = RANSACRegressor(
            estimator=LinearRegression(),
            min_samples=P["min_samples"],
            residual_threshold=P["residual_threshold"],
            max_trials=P["max_trials"],
            stop_probability=0.99
        )

        T = theta.reshape(-1, 1) # reshape for sklearn
        R = r
        model.fit(T, R) # fit Ransac model

        # Get inliers from model
        inliers = model.inlier_mask_
        if inliers.sum() < P["min_samples"]:
            print(f"[{label}] Rejected: inliers < min_samples")
            return None
        # r=a*theta + b. Get slope and intercept
        a = model.estimator_.coef_[0]
        b = model.estimator_.intercept_

        in_theta = theta[inliers]
        in_r = r[inliers]

        # calculate linearity score (Correlation coefficient of inliers. Closer to 1 is more linear)
        score = abs(np.corrcoef(in_theta, in_r)[0, 1])

        # Theta determines the endpoints of the segment in polar coordinates
        th_min = in_theta.min()
        th_max = in_theta.max()

        # radius calculated by endpoint thetas
        r1 = a * th_min + b
        r2 = a * th_max + b
        # convert to Cartesian coordinates for plotting and length
        x1 = r1 * np.cos(th_min)
        y1 = r1 * np.sin(th_min)
        x2 = r2 * np.cos(th_max)
        y2 = r2 * np.sin(th_max)

    # ---------------------------------------------------
    # Vertical model
    # May turn off if we only want horizontal segments
    # Similar to horizontal but swaps theta and r
    # ---------------------------------------------------
    else:

        model = RANSACRegressor(
            estimator=LinearRegression(),
            min_samples=P["min_samples"],
            residual_threshold=P["residual_threshold"],
            max_trials=P["max_trials"],
            stop_probability=0.99
        )

        R = r.reshape(-1, 1)
        T = theta
        model.fit(R, T)

        inliers = model.inlier_mask_
        if inliers.sum() < P["min_samples"]:
            print(f"[{label}] Rejected: inliers < min_samples")
            return None

        c = model.estimator_.coef_[0]
        d = model.estimator_.intercept_

        in_r = r[inliers]
        in_theta = theta[inliers]

        score = abs(np.corrcoef(in_r, in_theta)[0, 1])

        r_min = in_r.min()
        r_max = in_r.max()

        th1 = c * r_min + d
        th2 = c * r_max + d

        x1 = r_min * np.cos(th1)
        y1 = r_min * np.sin(th1)
        x2 = r_max * np.cos(th2)
        y2 = r_max * np.sin(th2)

        x1 = float(x1)
        y1 = float(y1)
        x2 = float(x2)
        y2 = float(y2)



    # ---------------------------------------------------
    # Compute length + diagnostics
    # ---------------------------------------------------
    length = np.hypot(x2 - x1, y2 - y1)
# Print Candidates for debugging
    print(f"[{label}] Candidate segment:")
    print(f"  model:       {model_type}")
    print(f"  inliers:     {inliers.sum()}")
    print(f"  linearity:   {score:.3f}")
    print(f"  length:      {length:.3f} m")
    print(f"  p1:          ({x1:.3f}, {y1:.3f})")
    print(f"  p2:          ({x2:.3f}, {y2:.3f})")

    if score < P["linearity_threshold"]:
        print("  REJECTED: linearity too low")
        return None

    if length < P["min_segment_length"]:
        print("  REJECTED: too short")
        return None

    if length > P["max_segment_length"]:
        print("  REJECTED: too long")
        return None

    print("  ACCEPTED")

    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    

    return {
        "orientation": model_type,
        "p1": (x1, y1),
        "p2": (x2, y2),
        "length": length,
        "linearity": score,
    }

# =======================================================
#                 FULL PIPELINE
# It Starts with 
# =======================================================

def main():
    with timed("TOTAL PIPELINE"):

        # --------------------------------------------------------------
        # Generate synthetic LiDAR-like polar data-replace with LiDAR input
        # --------------------------------------------------------------
        x, y, r, theta = generate_polar_segments()
        print("Generated polar-style segments.")

        P = PARAMS

        # --------------------------------------------------------------
        # HDBSCAN clustering in (theta, r) space-groups surface points together
        # -Supposedly better than DBSCAN for variablily dense data
        # -may go back to DBSCAN for speed when distance between boxes is known
        # --------------------------------------------------------------
        with timed("HDBSCAN clustering"):
            X = np.column_stack([theta, r])  # feature matrix for clustering
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=P["hdbscan_min_cluster_size"],
                min_samples=P["hdbscan_min_samples"]
            )
            labels = clusterer.fit_predict(X)
            unique_labels = sorted(set(labels))
            print(f"HDBSCAN clusters: {unique_labels}")

        # --------------------------------------------------------------
        # Bimodal splitting + gating
        # -HDBSCAN sometimes merges two surfaces if they are close.
        # -This splits clusters that have two distinct theta modes.
        # --------------------------------------------------------------

        with timed("Bimodal splitting"):
            polar_clusters = []

            # Precompute angular gate boundaries (constant)
            CENTER = np.radians(P["angle_center_deg"])
            HALF = np.radians(P["angle_halfwidth_deg"])

            for lab in unique_labels:
                if lab == -1:
                    continue  # skip noise cluster

                # Extract all points belonging to this HDBSCAN cluster
                mask = labels == lab
                theta_all = theta[mask]
                r_all = r[mask]

                # ---------------- Radius gate ----------------
                # Remove points too close or too far from LiDAR.
                mask_r = (r_all > P["radius_min"]) & (r_all < P["radius_max"])

                # ---------------- Angular gate ----------------
                #Restrict to expected angular region (e.g., barcode direction).
                mask_theta = (theta_all > CENTER - HALF) & (theta_all < CENTER + HALF)

                # ---------------- Combined gate ----------------
                mask_final = mask_r & mask_theta

                theta_gated = theta_all[mask_final]
                rr_gated = r_all[mask_final]

                # If nothing survives gating, skip this cluster
                if len(theta_gated) == 0:
                    continue

                # ---------------- Bimodal split ----------------
                subclusters = split_cluster_if_bimodal(theta_gated, rr_gated, P)

                # Store each subcluster separately for RANSAC
                for sub_th, sub_rr in subclusters:
                    polar_clusters.append((lab, sub_th, sub_rr))

        # --------------------------------------------------------------
        # RANSAC line fitting for each subcluster
        # --------------------------------------------------------------

        segments = []
        for idx, (lab, theta_sub, rr_sub) in enumerate(polar_clusters):
            with timed(f"RANSAC fit for Cluster {lab} sub {idx}"):

                seg = fit_RANSAC(
                    theta_sub,
                    rr_sub,
                    P,
                    label=f"Cluster {lab} sub {idx}"
                )

                if seg:
                    seg["cluster_label"] = lab
                    segments.append(seg)


        # ---------------- Plotting ----------------
        # with timed("Plotting"):
        #     plt.figure(figsize=(7, 7))
        #     plt.scatter(x, y, s=10, color="lightgray", alpha=0.5)

        #     colors = ["red", "green", "blue", "orange", "purple", "cyan"]
        #     for i, (lab, theta, rr) in enumerate(polar_clusters):
        #         xx = rr * np.cos(theta)
        #         yy = rr * np.sin(theta)
        #         plt.scatter(xx, yy, s=25, color=colors[i % len(colors)],
        #                     label=f"Cluster {lab} sub {i}")

        #     for seg in segments:
        #         x1, y1 = seg["p1"]
        #         x2, y2 = seg["p2"]
        #         plt.plot([x1, x2], [y1, y2], color="black", linewidth=3)

        #     plt.title("Final Segments After HDBSCAN + Bimodal Split + Dual RANSAC")
        #     plt.axis("equal")
        #     plt.grid(True)
        #     plt.legend()
        #     plt.show()

        # ---------------- Summary ----------------
        print("\n=== FINAL SEGMENTS ===")
        for i, seg in enumerate(segments, 1):
            print(f"\nSegment {i}:")
            print(f"  cluster_label: {seg['cluster_label']}")
            print(f"  orientation:    {seg['orientation']}")
            print(f"  length:        {seg['length']:.3f} m")
            print(f"  linearity:     {seg['linearity']:.3f}")
            x1, y1 = seg["p1"]
            x2, y2 = seg["p2"]
            print(f"  p1: ({x1:.3f}, {y1:.3f})")
            print(f"  p2: ({x2:.3f}, {y2:.3f})")

if __name__ == "__main__":
    main()