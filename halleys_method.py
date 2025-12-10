def halley_method(f, df, ddf, x0, tol=1e-10, max_iter=100):
    """
    Find root using Halley's method.

    Args:
    f: function
    df: first derivative
    ddf: second derivative
    x0: initial guess
    max_iter: maxumum iterations

    Returns:
    root approximation
    """
    x = x0
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        ddfx = ddf(x)

        # note: for complex numbers `abs()` returns the magnitude (distance from origin) in the complex plane
        if abs(fx) < tol:
            print(f"Root: {x}, Converged in {i} iterations")
            return x

        # Halley's formula
        denominator = 2 * dfx**2 - fx * ddfx
        if abs(denominator) < 1e-15:
            print("Denominator too small, stopping")
            return x
        x_new = x - (2 * fx * dfx) / denominator
        print(f"Iteration {i}: x = {x_new:.10f}, f(x) = {f(x_new):.2e}")

        x = x_new

    print("Max iterations reached")
    return x


# x0 = 4
x0 = 2 + 2j


# functions for sqrt(2)
def f(x):
    return x**2 - x0  # x^2 - a


# first derivative
def df(x):
    return 2 * x


# second derivative
def ddf(x):  # x isn't accessed, but the pattern might be useful for other formulas
    return 2


root = halley_method(f, df, ddf, x0=x0)
print(f"Approximate root: {root}")
print(f"Actual root: {x0**0.5}")
