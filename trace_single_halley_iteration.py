import numpy as np
import matplotlib.pyplot as plt


def af(z):
    return z**3 - 0.1


def df(z):
    return 3 * z**2


def ddf(z):
    return 6 * z


def trace_halleys_iteration(f, df, ddf, z0, max_iter=20):
    """Track the iteration path for a single starting point"""
    path = [z0]
    z = z0

    for _ in range(max_iter):
        fz = f(z)
        dfz = df(z)
        ddfz = ddf(z)

        if abs(fz) < 1e-6:
            break

        denominator = 2 * dfz**2 - fz * ddfz
        if abs(denominator) < 1e-15:
            break

        z = z - (2 * fz * dfz) / denominator
        path.append(z)

    return np.array(path, dtype=np.complex128)


# The three roots
alt_roots = [
    0.1 ** (1.0 / 3),
    0.1 ** (1.0 / 3) * np.exp(2j * np.pi / 3),
    0.1 ** (1.0 / 3) * np.exp(4j * np.pi / 3),
]

fig, ax = plt.subplots(figsize=(12, 12))

# Plot the three boundary rays
for angle in [np.pi / 3, np.pi, 5 * np.pi / 3]:  # 60°, 180°, 300°
    ray = np.array([0.1, 2.0]) * np.exp(1j * angle)
    ax.plot(
        ray.real,
        ray.imag,
        "--",
        color="gray",
        linewidth=2,
        alpha=0.5,
        label="Boundary ray" if angle == np.pi / 3 else "",
    )

# Plot roots
for i, root in enumerate(alt_roots):
    ax.plot(root.real, root.imag, "o", markersize=15, label=f"Root {i + 1}", zorder=5)
    # Draw line from origin to root
    ax.plot([0, root.real], [0, root.imag], ":", color="black", alpha=0.3)

# The special starting points (3rd, 7th, 11th)
special_angles = [np.pi / 3, np.pi, 5 * np.pi / 3]
radius = 1.5

for i, angle in enumerate(special_angles):
    z0 = radius * np.exp(1j * angle)
    path = trace_halleys_iteration(af, df, ddf, z0, max_iter=15)

    # Plot the path with emphasis
    ax.plot(
        path.real,
        path.imag,
        "o-",
        linewidth=2.5,
        markersize=6,
        label=f"Path from angle {int(np.degrees(angle))}°",
        zorder=4,
    )
    ax.plot(
        z0.real, z0.imag, "x", color="red", markersize=12, markeredgewidth=3, zorder=6
    )

    # Print distances to nearest two roots at each step
    print(f"\n=== Starting angle: {np.degrees(angle):.0f}° ===")
    for j, point in enumerate(path[:5]):  # First 5 iterations
        distances = sorted(
            [(abs(point - root), idx) for idx, root in enumerate(alt_roots)]
        )
        print(
            f"Step {j}: distance to closest root: {distances[0][0]:.6f}, "
            f"to 2nd closest: {distances[1][0]:.6f}, "
            f"ratio: {distances[1][0] / distances[0][0]:.4f}"
        )

ax.set_xlabel("Real", fontsize=12)
ax.set_ylabel("Imaginary", fontsize=12)
ax.set_title(
    "Iteration Paths Along Basin Boundaries\n(Starting exactly between roots)",
    fontsize=14,
)
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
ax.axis("equal")
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
plt.tight_layout()
plt.show()
