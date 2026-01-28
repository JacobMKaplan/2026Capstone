import numpy as np
import matplotlib.pyplot as plt
def generate_polar_segments():
    np.random.seed(42)

    all_r = []
    all_theta = []

    # Horizontal segments
    horizontal_defs = [
        (2.0, -113.5, -106.3),
        (2.0, -103.5, -96.3),
        (2.0, -93.5, -86.3)
    ]

    for r_base, th_start, th_end in horizontal_defs:
        theta_deg = np.linspace(th_start, th_end, 30)
        theta_rad = np.radians(theta_deg)
        theta_rad += np.radians(np.random.uniform(-0.2, 0.2, len(theta_rad)))
        r_vals = r_base + np.random.uniform(-0.01, 0.01, len(theta_rad))

        all_r.append(r_vals)
        all_theta.append(theta_rad)

    # Vertical segments
    vertical_defs = [-120.0, -85.0]

    for th_deg in vertical_defs:
        theta_rad = np.radians(th_deg)
        theta_vals = theta_rad + np.radians(np.random.uniform(-0.2, 0.2, 30))
        r_vals = np.linspace(1.75, 2.25, 30)
        r_vals += np.random.uniform(-0.01, 0.01, len(r_vals))

        all_r.append(r_vals)
        all_theta.append(theta_vals)

    # Noise
    noise_theta = np.radians(np.random.uniform(-135, -45, 20))
    noise_r = np.random.uniform(1.8, 2.2, 20)

    all_r.append(noise_r)
    all_theta.append(noise_theta)

    r_all = np.concatenate(all_r)
    theta_all = np.concatenate(all_theta)

    x = r_all * np.cos(theta_all)
    y = r_all * np.sin(theta_all)

    return x, y, r_all, theta_all

def plot_raw_data(x, y):
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=20, c='dodgerblue', alpha=0.7, edgecolors='k')

    plt.title("Raw Synthetic Polar Data (Cartesian)")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.axis("equal")
    plt.grid(True)
    plt.show()

x, y, r_all, theta_all = generate_polar_segments()

# Plot the raw Cartesian scatter
plot_raw_data(x, y)
