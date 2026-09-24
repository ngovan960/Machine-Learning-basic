import numpy as np

def grad(x):
    return x*x-1

def cost(x):
    return 1/3*(x**3)-x

def myGD1(x0, eta):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x[-1])) < 1e-3:  # just a small number
            break
        x.append(x_new)
    return (x, it)

(x1, it1) = myGD1(0.5, .2)

print(f"x1: {x1[-1]}, cost: {cost(x1[-1])}, after:{it1}")
