#%%
import numpy as np
import matplotlib.pyplot as plt

def plot_beam_pattern_db(steering_vector_func, weight_vector, num_points=360, db_floor=-40):
    """
    Plots the beam pattern in dB scale for a given steering vector function and weight vector.
    
    Parameters:
    - steering_vector_func: Function that returns the steering vector for a given angle (in radians).
    - weight_vector: Complex weight vector (numpy array).
    - num_points: Number of angle samples (default: 360).
    - db_floor: Minimum dB level to clip to (default: -40 dB).
    """
    theta = np.linspace(0, 2 * np.pi, num_points)
    pattern = np.zeros_like(theta, dtype=np.float64)

    for i, angle in enumerate(theta):
        a = steering_vector_func(angle)
        pattern[i] = np.abs(np.vdot(weight_vector, a))  # w^H * a(theta)

    pattern /= np.max(pattern)
    pattern_db = 20 * np.log10(pattern)
    pattern_db = np.clip(pattern_db, db_floor, None)

    # Normalize to [0, 1] for plotting
    pattern_db_plot = (pattern_db - db_floor) / -db_floor

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(theta, pattern_db_plot)
    ax.set_title("Beam Pattern (dB)", va='bottom')

    r_ticks = np.linspace(0, 1, 5)
    r_labels = [f"{int(db_floor * (1 - r))} dB" for r in r_ticks]
    ax.set_yticks(r_ticks)
    ax.set_yticklabels(r_labels)
    plt.savefig('C:\\Users\\Calder\\OneDrive - NTNU\\Documents\\Presentations\\ASA May 2025\\Figures\\0pattern.png')

    plt.show()


def ula_steering_vector(theta, num_elements=8, d=24, wavelength=147):
    k = 2 * np.pi / wavelength
    n = np.arange(num_elements)
    return np.exp(1j * k * d * n * np.cos(theta))

# Number of array elements
N = 30

# Define beamforming weights (e.g., steer to broadside: 90°, or θ = π/2)
target_angle = np.pi / 2
a_target = ula_steering_vector(target_angle, num_elements=N)
weights = a_target / np.linalg.norm(a_target)  # normalize

# Plot the beam pattern
plot_beam_pattern_db(lambda theta: ula_steering_vector(theta, num_elements=N), weights)

# %%
