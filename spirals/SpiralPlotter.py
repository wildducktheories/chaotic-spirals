import math
import matplotlib.pyplot as plt
import numpy as np

class MatrixState:
    def init(n):
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
        return MatrixState(n, state, LEFT, FORWARD)


    def __init__(self, n, state, left, forward):
        self.n = n
        self._left    = left
        self._forward = forward
        self.state    = state

    def left(self):
        return MatrixState(self.n, np.matmul(self._left, self.state), self._left, self._forward)

    def forward(self):
        return MatrixState(self.n, np.matmul(self._forward, self.state), self._left, self._forward)

    def xy(self):
        return (round(self.state[0].real,10), round(self.state[0].imag, 10))

    def id(self):
        return self.xy()

    def heading(self):
        return self.state[1]

    def prev_heading(self):
        return self.state[2]


class SpiralPlotter:

    def init(self, n):
        return MatrixState.init(n)

    def spiral(self, n, length=None, yield_forward_only=False):
        length = n ** 2 if length is None else length
        state = self.init(n)
        visited = set()

        while length > 0:
            visited.add(state.id())
            if not yield_forward_only or (yield_forward_only and state.heading()==state.prev_heading()):
                length = length - 1
                yield state.xy()

            next = state.left()
            if next.left().id() in visited:
                next = state.forward()

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