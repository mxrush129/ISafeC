import numpy as np

def format_polynomial(coeffs):
    terms = []
    for i, c in enumerate(coeffs):
        power = len(coeffs) - i - 1
        if abs(c) < 1e-10:  # 忽略极小值
            continue
        if power == 0:
            term = f"{c:.4f}"
        else:
            term = f"{c:.4f} *" + ("x" if power == 1 else f"x**{power}")
        terms.append(term if i == 0 else f"{' + ' if c > 0 else ' - '}{abs(c):.4f} *" + ("" if power == 0 else "x" if power == 1 else f"x**{power}"))
    if terms and terms[-1][-1] == '*':
        terms[-1] = terms[-1][:-1]
    return "y = " + " ".join(terms) if terms else "y = 0"

def fit(x, y, k):
    coeff = np.polyfit(x, y, k)
    y_pred = np.polyval(coeff, x)
    mse = np.mean((y - y_pred)**2)

    print("系数:", [f"{c:.4f}" for c in coeff])
    print("拟合多项式:", format_polynomial(coeff))
    print("均方误差:", f"{mse:.14f}")

if __name__ == '__main__':
    x = np.linspace(-2, 2, 1000)
    f1 = (1 / (1 + np.sin(x)**2))
    f2 = (np.sin(x) / (1 + np.sin(x)**2))
    f3 = (np.sin(x) * np.cos(x) / (1 + np.sin(x)**2))
    f4 = (np.cos(x) / (1 + np.sin(x)**2))
    # fit(x, f1, 4)
    # fit(x, f2, 3)
    # fit(x, f3, 5)
    fit(x, f4, 4)