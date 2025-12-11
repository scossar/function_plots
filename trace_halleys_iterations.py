import numpy as np
import matplotlib.pyplot as plt


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
        # for debugging
        # print(f"z.real: {z.real}, z.imag: {z.imag}")
        path.append(z)

    return np.array(path, dtype=np.complex128)


# f(z) = z^3 - 0.1
def f(z):
    return z**3 - 0.1


def df(z):
    return 3 * z**2


def ddf(z):
    return 6 * z


roots = [
    0.1 ** (1.0 / 3),
    0.1 ** (1.0 / 3) * np.exp(2j * np.pi / 3),
    0.1 ** (1.0 / 3) * np.exp(4j * np.pi / 3),
]


# Plot the roots and some iteration paths
fig, ax = plt.subplots(figsize=(10, 10))

# Plot roots
for i, root in enumerate(roots):
    ax.plot(root.real, root.imag, "o", markersize=12, label=f"Root {i + 1}", zorder=5)

# Sample starting points from different angular sectors
colors = ["red", "blue", "green"]
angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
radius = 1.5
# angles:
# [0.         0.52359878 1.04719755 1.57079633 2.0943951  2.61799388
#  3.14159265 3.66519143 4.1887902  4.71238898 5.23598776 5.75958653]

# test a slice of angles
# angles = angles[:1]
print(angles)
for angle in angles:
    z0 = radius * np.exp(1j * angle)
    path = trace_halleys_iteration(f, df, ddf, z0, max_iter=15)

    # Determine which root it converges to
    final_z = path[-1]
    distances = [abs(final_z - root) for root in roots]
    root_idx = np.argmin(distances)

    # Plot the path
    ax.plot(
        path.real,
        path.imag,
        "o-",
        alpha=0.6,
        color=colors[root_idx],
        markersize=4,
        linewidth=1,
    )
    ax.plot(z0.real, z0.imag, "x", color="black", markersize=8, markeredgewidth=2)

ax.set_xlabel("Real", fontsize=12)
ax.set_ylabel("Imaginary", fontsize=12)
ax.set_title(
    "Iteration Paths from Different Starting Points\n(Same color = converge to same root)",
    fontsize=14,
)
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis("equal")
plt.tight_layout()
plt.show()
