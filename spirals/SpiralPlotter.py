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

class Vectors:
    """
        This class knows how to represent the n unit vectors of the form e^i(2*pi*k/n)
        for 0 <= k < n and the symmeties between them.

        It also knows how to map an n-dimensional vector space down into a cross
        product of two floor((n+1)/2) vector spaces by matching vectors
        which are additive inverses (n is even) or complex conjugates (n is odd).

        Finally, it knows how to project the vector space into a 2D space by
        multiplying the id vectors with cosine and cosine coefficients of the unit
        vectors and summing the results.

    """
    def __init__(self, n):
        m=int(n/2) if int(n%2) == 0 else int((n+1)/2)
        o=n%2
        self.coefficients=np.round(np.array([
            [np.cos(math.pi*2*i/n), np.sin(math.pi*2*i/n)] for i in range(0,m)
        ]), 8)
        self.m = m
        self.n = n
        self.o = o
        self.x = np.zeros(shape=(m,n),dtype=int)
        self.y = np.zeros(shape=(m,n),dtype=int)
        self.x[0:m,0:m]=np.identity(m, dtype=int)
        self.y[0:m,0:m]=np.identity(m, dtype=int)
        if o == 0:
            self.x[0:,self.m:]=-np.identity(m)
            self.y[0:,self.m:]=-np.identity(m)
        else:
            self.x[1:,self.m:]= np.fliplr(np.identity(m-1))
            self.y[1:,self.m:]=-np.fliplr(np.identity(m-1))
        self.reductions = [ self._reduce(self.coefficients[:,i]) for i in [0,1]]

    def _reduce(self, c):
        l = len(c)
        d = dict(zip(np.round(c,8), range(0, l)))
        m = np.identity(l,dtype=int)
        for x in range(0, l):
            a=np.abs(c[x])
            if a in d:
                i=d[a]
                m[x,x]=0
                m[i,x]=np.int32(np.sign(c[x]+0.1))
        return m


    def to_id(self, p):
        # return (
        #     tuple(np.matmul(self.reductions[0], np.matmul(self.x, p))),
        #     tuple(np.matmul(self.reductions[1], np.matmul(self.y, p)))
        # )
        return (
            tuple(np.matmul(self.x, p)),
            tuple(np.matmul(self.y, p))
        )

    def to_xy(self, p):
        return (
            np.round(np.sum(np.matmul(self.coefficients[:,0],np.matmul(self.x, p))),8),
            np.round(np.sum(np.matmul(self.coefficients[:,1],np.matmul(self.y, p))),8)
        )


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