import numpy as np
import matplotlib as plt
# 수치미분
def numerical_diff(f, x):
    h = 1e-50
    return (f(x + h) - f(x)) / h

def numerical_diff(f, x):
    h = 1e-4 # 0.001
    return (f(x+h) - f(x-h)) / (2*h)

def function_1(x):
    return 0.01*x**2 + 0.1*x

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)

fig, ax = plt.subplots(figsize=(5,5))
ax.xlabel("x")
ax.ylabel("f(x)")
ax.plot(x, y)
plt.show()