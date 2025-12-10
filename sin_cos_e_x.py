import numpy as np
import matplotlib.pyplot as plt


def sin_cos_exp(x):
    return np.sin(np.cos(np.exp(x)))


# finding the derivative of f(x) = sin(cos(e^x))
# Note that:
# d/dx[sin(x)] = cos(x)
# d/dx[cos(x)] = -sin(x)
# d/dx[e^x] = e^x
# think of it as three nested functions:
# sin(u) where u = cos(e^x); d/dx[sin(cos(e^x))] = cos(cos(e^x)) x d/dx[cos(e^x)]
# cos(v) where v = e^x; d/dx[cos(e^x)] = -sin(e^x) x d/dx[e^x]
# innermost e^x; d/dx[e^x] = e^x; (the derivative of e^x is e^x)
# f'(x) = cos(cos(e^x)) x -sin(e^x) x e^x
def sin_cos_exp_derivatives(x):
    expx = np.exp(x)
    return np.cos(np.cos(expx)) * -np.sin(expx) * expx


print("x=0.0", sin_cos_exp_derivatives(0.0))
# x=0.0 -0.7216061490634433
print("x=0.4", sin_cos_exp_derivatives(0.4))
# x=0.4 -1.4825498531537413


# I'd need to define lambda functions for f and df to use this function
def newtons_method_lambdas(f, df, x0, tolerance=1e-6, max_iter=100):
    xn = x0
    for n in range(max_iter):
        fxn = f(xn)
        if abs(fxn) < tolerance:
            print(f"Found solution {xn} after {n} iterations")
            return xn
        dfxn = df(xn)
        if dfxn == 0:
            print("Zero derivative. No solution found.")
            return None
        xn = xn - fxn / dfxn
    print("Exceeded max iterations. No solution found.")
    return None


def newtons_method(x0, tolerance=1e-6, max_iter=100):
    xn = x0
    for n in range(max_iter):
        fxn = sin_cos_exp(xn)
        if abs(fxn) < tolerance:
            print(f"Found solution {xn} after {n} iterations.")
            return xn
        dfxn = sin_cos_exp_derivatives(xn)
        if dfxn == 0:
            print("Zero derivative. No solution found.")
            return None
        xn = xn - fxn / dfxn
    print("Exceeded max iterations. No solution found.")
    return None


newtons_method(0.4)
newtons_method(1.5)

x_arr = np.linspace(-5, 5, 500)
y_arr = [sin_cos_exp(x) for x in x_arr]


plt.plot(x_arr, y_arr)
plt.axhline(y=0, color="k", linewidth=0.5)
plt.axvline(x=0, color="k", linewidth=0.5)
plt.grid(True, alpha=0.3)
plt.show()
