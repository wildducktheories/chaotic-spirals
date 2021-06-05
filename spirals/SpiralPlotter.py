import math
import matplotlib.pyplot as plt
import numpy as np

class SpiralPlotter:

    def spiral(self, n, length=None, yield_forward_only=False):
        length = n ** 2 if length is None else length
        R = complex(np.cos(2*math.pi/n), np.sin(2*math.pi/n))
        LEFT    = [
                    [1, 1, 0],
                    [0, R, 0],
                    [0, 1, 0]
                  ]

        FORWARD = [
                    [1, 1, 0],
                    [0, 1, 0],
                    [0, 1, 0]
                  ]

        state = [
            complex(0,0), # position
            complex(1,0), # heading
            complex(0,0)  # previous heading
        ]

        visited = set()

        to_xy = lambda state: (round(state[0].real,8), round(state[0].imag, 8))

        while length > 0:
            xy = to_xy(state)
            visited.add(xy)
            if not yield_forward_only or (yield_forward_only and state[1]==state[2]):
                length = length - 1
                yield xy

            next = np.matmul(LEFT, state)
            if to_xy(np.matmul(LEFT, next)) in visited:
                next = np.matmul(FORWARD, state)

            state = next

    def plot_spiral(self, n, s, plot=True, scatter=True, title=None):
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