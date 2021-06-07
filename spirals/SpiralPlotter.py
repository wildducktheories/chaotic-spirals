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
        self.x[0:m,0:m]=np.identity(m)
        self.y[0:m,0:m]=np.identity(m)
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
        m = np.identity(l)
        one_index = d[1.0] if 1.0 in d else None
        for x in range(0, l):
            a=np.abs(np.round(c[x],8))
            if np.abs(np.round(a-0.5, 8)) < 1e-8:
                if not one_index is None:
                    m[x,x] = 0
                    m[one_index,x] += np.round(c[x], 8)
            elif a < 1e-8:
                 m[x,x] = 0
            elif a in d:
                i=d[a]
                if not a < 1e-8:
                    m[x,x]=0
                    m[i,x]=np.sign(c[x]+0.1)
        return m

    def to_reduced_id(self, p):
        return (
            tuple(np.matmul(self.reductions[0], np.matmul(self.x, p))),
            tuple(np.matmul(self.reductions[1], np.matmul(self.y, p)))
        )


    def to_id(self, p):
        return (
            tuple(np.matmul(self.x, p)),
            tuple(np.matmul(self.y, p))
        )

    def to_xy(self, p):
        p=p-np.min(p) # sum of all vectors is zero
        return (
            np.round(np.sum(np.matmul(self.coefficients[:,0],np.matmul(self.x, p))),8),
            np.round(np.sum(np.matmul(self.coefficients[:,1],np.matmul(self.y, p))),8)
        )


class VectorState:
    """
        Represents a spiral state in terms of a linear combination of unit vectors of the form e^i*(2*pi*k/n)

        The main difference between this class and MatrixState is that state is maintained as vectors of integers
        rather than floats and floats are only used when converting a state into an xy coordinate.
    """
    def init(n):
        p=np.zeros(shape=(n), dtype=int)
        return VectorState(n, p, 0, None, Vectors(n))

    def __init__(self, n, p, d, prev_d, vectors):
        self.n = n
        self.p = (p - np.min(p[1:]))
        self.d = d
        self.prev_d = prev_d
        self.vectors = vectors

    def id(self):
        return self.vectors.to_id(self.p)

    def reduced_id(self):
        return self.vectors.to_reduced_id(self.p)

    def xy(self):
        return self.vectors.to_xy(self.p)

    def left(self):
        p=np.copy(self.p)
        p[self.d] = p[self.d]+1
        d = (self.d+1)%self.n
        return VectorState(self.n, p, d, self.d, self.vectors)

    def forward(self):
        p=np.copy(self.p)
        p[self.d] = p[self.d]+1
        return VectorState(self.n, p, self.d, self.d, self.vectors)

    def heading(self):
        return self.d

    def prev_heading(self):
        return self.prev_d

    def __repr__(self):
        return str(self)

    def __str__(self):
        left=self.left()
        forward=self.forward()
        return f"id={self.id()} reduced_id={self.reduced_id()} xy={self.xy()} p={self.p} d={self.d} prev_d={self.prev_d} left={left.id()}@{left.d} forward={forward.id()}@{forward.d}"

class SpiralPlotter:

    def spiral(self, n, length=None, initial=None,  filter=None, transform=None, id=None):

        if not initial:
            initial = VectorState.init(n)

        if not filter:
            filter = lambda s: True

        if not transform:
            transform = lambda s: s.xy()

        if not id:
            id = lambda s: s.reduced_id()

        state = initial
        visited = set()
        length = n ** 2 if length is None else length

        while length > 0:
            visited.add(id(state))
            if filter(state):
                length = length - 1
                yield transform(state)

            next = state.left()
            if id(next.left()) in visited:
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