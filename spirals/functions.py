import numpy as np
import matplotlib.pyplot as plt
from .classes import VectorState, Vectors

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

def spiral(n, length=None, initial=None,  visited=None, filter=None, transform=None):

    if not initial:
        initial = VectorState.init(n)

    if not filter:
        filter = lambda s: True

    if not transform:
        transform = lambda s: s

    if not visited:
        visited = set()

    state = initial
    length = n ** 2 if length is None else length

    while length > 0:
        visited.add(state.id())
        if filter(state):
            length = length - 1
            yield transform(state)

        next = state.left()
        if next.stop().id() in visited:
            next = state.forward()

        state = next

def plot_spiral(n, s, plot=True, scatter=True, title=None):
    v = np.array([(np.array(p)) for p in s])
    fig = plt.figure(figsize=(6,6))
    ax=plt.axes()
    #ax.set_facecolor('#f1c738')
    if not title:
        title=f"n={n}"
    plt.title(title)
    plt.gca().set_aspect("equal")
    if plot:
        plt.plot(v[:,0], v[:,1])
    if scatter:
        plt.scatter(v[:,0], v[:,1], marker=".")
    plt.show()