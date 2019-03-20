import matplotlib.pyplot as plt

# sample points
X = [0, 6, 11, 14, 22]
Y = [1, 7, 12, 15, 21]


def best_fit(X, Y):
    """"solve for a and b
    """
    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)
    
    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2
    
    b = numer / denum
    a = ybar - b * xbar

    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))
    
    return a, b

# solution
a, b = best_fit(X, Y)
"""
best fit line:
y = 1.48 + 0.92x
"""

# plot points and fit line

plt.scatter(X, Y)
yfit = [a + b * xi for xi in X]
plt.plot(X, yfit)
plt.show()
