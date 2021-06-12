import numpy as np
import matplotlib.pyplot as plt

def average_position(p):
    s=np.zeros(shape=(2))
    n=0
    for e in p:
        s += np.array(e)
        n = n+1
        yield s/n

def plot_G_for_all_N(title, G, N):
    plt.figure(figsize=(12,12))
    plt.gca().set_aspect('equal')
    plt.title(title)
    for n in N:
        points=np.array([e for e in G(n)])
        plt.plot(points[:,0], points[:,1], linewidth=1)
    plt.show()