import matplotlib.pyplot as plt
import numpy as np


z = np.array([1 + 2j, -1 + 1j, 2 - 1j, -2 - 2j, 3 + 0j], dtype=np.complex64)


plt.scatter(z.real, z.imag)
plt.axhline(y=0, color="k", linewidth=0.5)
plt.axvline(x=0, color="k", linewidth=0.5)
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.grid(True, alpha=0.3)
plt.axis("equal")
plt.show()
