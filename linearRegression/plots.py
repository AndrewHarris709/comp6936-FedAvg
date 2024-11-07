import matplotlib.pyplot as plt

def plot_scatter_line(X, Y, w, b):
    plt.scatter(X, Y)
    plt.plot(X, w * X + b, color = "black", linewidth = 3)
    plt.show()