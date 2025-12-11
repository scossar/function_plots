import numpy as np
import matplotlib.pyplot as plt


def halleys_roots_fractal(
    f,
    df,
    ddf,
    roots,
    width=10,
    height=10,
    xmin=-1.0,
    xmax=1.0,
    ymin=-1.0,
    ymax=1.0,
    max_iter=50,
):
    # create a grid of complex numbers
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # record which root a point converges to
    root_map = np.zeros((height, width), dtype=int)
    # record how many iterations it takes for a point to converge
    iter_map = np.zeros((height, width), dtype=int)

    for i in range(height):
        for j in range(width):
            z = Z[i, j]

            for iteration in range(max_iter):
                fz = f(z)
                dfz = df(z)
                ddfz = ddf(z)

                if abs(fz) < 1e-6:  # converged
                    distances = [abs(z - root) for root in roots]
                    root_idx = np.argmin(distances)
                    root_map[i, j] = root_idx
                    iter_map[i, j] = iteration
                    break

                denominator = 2 * dfz**2 - fz * ddfz
                if abs(denominator) < 1e-15:
                    print("Denominator approaching zero; breaking from loop")
                    break
                z = z - (2 * fz * dfz) / denominator

            else:  # for...else is legitimate Python; runs if a break statement isn't hit in the for loop
                # didn't converge
                root_map[i, j] = -1
                iter_map[i, j] = max_iter

    return root_map, iter_map, x, y


# f(z) = z^3 - 1
def f(z):
    return z**3 - 1


def df(z):
    return 3 * z**2


def ddf(z):
    return 6 * z


roots = [1, np.exp(2j * np.pi / 3), np.exp(4j * np.pi / 3)]


# an example with an alternate function; not called in the code below
# f(z) = z^3 - 0.1
def af(z):
    return z**3 - 0.1


# use alt_roots and af instead of roots and f to see the difference (it's not much different)
alt_roots = [
    0.1 ** (1.0 / 3),
    0.1 ** (1.0 / 3) * np.exp(2j * np.pi / 3),
    0.1 ** (1.0 / 3) * np.exp(4j * np.pi / 3),
]

root_map, iter_map, xs, ys = halleys_roots_fractal(
    af,
    df,
    ddf,
    alt_roots,
    xmin=-1000,
    xmax=1000,
    ymin=-1000,
    ymax=1000,
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

im1 = ax1.imshow(
    root_map, extent=(xs[0], xs[-1], ys[0], ys[-1]), cmap="viridis", origin="lower"
)
ax1.set_title("Basins of attraction")
ax1.set_xlabel("Real")
ax1.set_ylabel("Imaginary")
cbar1 = plt.colorbar(im1)  # could use cbar1.set_label("label")

im2 = ax2.imshow(
    iter_map, extent=(xs[0], xs[-1], ys[0], ys[-1]), cmap="viridis", origin="lower"
)
ax2.set_title("Convergence speed")
ax2.set_xlabel("Real")
ax2.set_ylabel("Imaginary")
cbar2 = plt.colorbar(im2)

plt.tight_layout()
plt.show()
