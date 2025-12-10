import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-2, 2, 100) + 1j * np.linspace(-2, 2, 100)[:, None]
w = z**2

plt.imshow(np.abs(w), extent=(-2, 2, -2, 2), origin="lower")
plt.colorbar(label="Magnitude")
plt.show()
