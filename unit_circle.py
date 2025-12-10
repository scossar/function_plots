import numpy as np
import matplotlib.pyplot as plt


def plot_unit_circle_with_angle(angle_rad):
    _, ax = plt.subplots(figsize=(6, 6))

    theta = np.linspace(0, 2 * np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    ax.plot(x, y, color="gray", linestyle="--", label="Unit Circle")

    x_angle = np.cos(angle_rad)
    y_angle = np.sin(angle_rad)

    # NOTE: lists of ax.plot args seem to work differently than I expected
    # first list is x values, second list is y values (?)
    ax.plot(
        [0, x_angle],
        [0, y_angle],
        color="red",
        linewidth=1,
    )

    ax.plot(
        [x_angle, x_angle],
        [y_angle, 0],
        color="green",
        linewidth=1,
    )

    ax.plot(
        [0, x_angle],
        [0, 0],
        color="blue",
        linewidth=1,
    )

    ax.plot(x_angle, y_angle, "ro", label=f"{x_angle},{y_angle}", markersize=2)
    ax.annotate(f"({x_angle:.4f},{y_angle:.4f})", xy=(x_angle, y_angle))

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_title(f"Unit Circle, angle: {np.degrees(angle_rad)}")
    ax.grid(True, linestyle=":", alpha=0.6)

    ax.axhline(y=0, color="k", linewidth=0.25)
    ax.axvline(x=0, color="k", linewidth=0.25)
    plt.show()


plot_unit_circle_with_angle(np.pi / 4)
