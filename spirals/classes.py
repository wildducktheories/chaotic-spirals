import numpy as np
import math

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

    def stop(self):
        return StopState.init(self.forward())

class StopState:
    """
        This stage can never progress anywhere, but it can be used to report the final
        id and xy coordinate reached by a previous state.
    """
    def init(state):
        return StopState(state.n, state.id(), state.xy(), state.heading())

    def __init__(self, n, id, xy, heading):
        self.n = n
        self._id = id
        self._xy = xy
        self._heading = heading

    def id(self):
        return self._id

    def xy(self):
        return self._xy

    def left(self):
        return self

    def forward(self):
        return self

    def stop(self):
        return self

    def heading(self):
        return self._heading

    def prev_heading(self):
        return self._heading

class Vectors:
    """
        Instances of this class represent all the unit vectors that are rotated an integer multiple
        of 2pi/n displaced from the real axis for all.

        .coefficients
            contains a matrix whose elements represent the real and imaginary components of the 12 vectors

        .id_coefficients
            contains a matrix of coefficients which are not integer multiples of each other

        .id_reduction
            contains 2 matricies that collapses a n-element state vector into 2 smaller id
            vectors such that none of the coefficients are integer multiples of the other

        .to_id()
            maps a state vector containing counts of unit vectors to a tuple of two vectors
            whose elements correspond to the elements of id_coefficients

        .to_xy()
            maps a state vector containing counts of unit vectors to an xy coordinate

    """
    def __init__(self, n):
        PRECISION=1e-8

        self.n = n

        # self.coefficients
        #
        # Each row contains the magitudes of the real and imaginary components of the
        # b unit vectors that partition the unit circle into sectors of angle 2pi/n radians each
        # starting with the pure real unit vector (1+0i) as element 0 and
        # iterating in an anti-clockwise direction

        self.coefficients = np.round(np.array([ [math.cos(math.pi*2*i/n), math.sin(math.pi*2*i/n)] for i in range(0,n) ]), 12).transpose()

        # m is a 3-dimensional transformation matrix.
        #
        # m[i,:,:] corresponds to ith axis of the complex plane (0=real, 1=imaginary)
        # m[i,j,:] corresponds to the jth row of the output matrix
        # m[i.j,k] corresponds to the contribution of the kth row of the state vector to the jth row of the output matrix
        #
        # The 3-dimensional matrix contains 2 nxn matrices, one for each of the real and
        # imagiary axes. Each of the nxn matrices is initially equal to the nxn identity matrix.
        #
        # The matrices are progressively edited so that input vector elements whose
        # corresponding coeeficients are integer multiples of each other can be summed into
        # a single element of the output vector, effectively reducing the dimensionality of the
        # output vectors.
        #
        # The intent of building this matrix is to allow it to be used to map all state vectors
        # that map to the same XY coordinate to a unique state vector of reduced dimension, thereby
        # allowing identity matching to be done in precise integer vector space, rather than in the
        # imprecise domain of floating point vectors.
        #

        # make a copy because we need to mutate it
        _coefficients = np.copy(self.coefficients)

        m=np.zeros(shape=(2,n,n), dtype=int)
        for i in range(0,len(m)):
            m[i,0:n,0:n] = np.identity(n, dtype=int)

            c = _coefficients[i,:]
            a = np.abs(c)

            # if we have any coefficients whose absolute value is 0.5, then
            # then we multiply unit coefficients by 0.5 and adjust the
            # reduction matrix to compensate by muliplying the corresponding
            # element by 2.
            if len(np.where(np.abs(a-0.5) < PRECISION)[0]) > 0:
                for j in np.where(np.abs(a-1.0) < PRECISION)[0]:
                    c[j]=np.sign(c[j])*0.5
                    a[j]=np.abs(c[j])
                    m[i,j,j] = 2

            # next, we ensure that any input rows whose coefficients have the
            # same magnitude are summed into the same output row
            # this will leave some rows containing zeros that need to be
            # collapsed.
            for k, e in enumerate(a):
                j=np.where(np.abs(a-e)<PRECISION)[0][0]
                if j != k or e<PRECISION:
                    m[i,j,k]=int(np.sign(c[k]*c[j]))*np.abs(m[i,k,k])
                    m[i,k,k]=0

        # the collapse lambda reduces a matrix 'a' by eliding all rows
        # where the corresponding row of matrix 'm' contains only zeros
        collapse = lambda a : np.array([
            a[i][
                [not np.all(r) for r in m[i] == 0]
            ] for i in (0,1)
        ], dtype=object)

        #
        # we apply the collapse function to produce a matrix that can
        # reduces a state vector to a reduced state vector in which all
        # dimensions have
        #
        self.id_reduction = collapse(m)

        #
        # we apply the collapse function to coefficients vector to produce
        # the set of coefficients such that set(np.unique(mp.abs(self.coefficients[i]))) =
        #
        self.id_coefficients = collapse(_coefficients)


    def to_xy(self, p):
        # converts the id vectors into a position on a tuple describing a point on the x-y plane
        return tuple([ np.dot(self.id_coefficients[i], self.to_id(p)[i]) for i in (0,1) ])

    def to_id(self, p):
        # this exploits an identity that the sum of all unit vectors is 0
        p = p - np.min(p)

        # transforms the n-element state vactor into 2-tuple containing tuples of smaller dimension
        return tuple([tuple(np.matmul(r, p)) for r in self.id_reduction])


class VectorState:
    """
        Represents a spiral state in terms of a linear combination of unit vectors of the form e^i*(2*pi*k/n)

        The main difference between this class and MatrixState is that state is maintained as vectors of integers
        rather than floats and floats are only used when converting a state into an xy coordinate.
    """
    def init(n, v=None):
        p=np.zeros(shape=(n), dtype=int)
        if v==None:
            v = Vectors(n)
        return VectorState(n, p, 0, None, v)

    def __init__(self, n, p, d, prev_d, vectors):
        self.n = n
        self.p = p
        self.d = d
        self.prev_d = prev_d
        self.vectors = vectors

    def id(self):
        return self.vectors.to_id(self.p)

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

    def stop(self):
        return StopState.init(self.forward())

    def heading(self):
        return self.d

    def prev_heading(self):
        return self.prev_d

    def __repr__(self):
        return str(self)

    def __str__(self):
        left=self.left()
        forward=self.forward()
        return f"id={self.id()} xy={self.xy()} p={self.p} d={self.d} prev_d={self.prev_d} left={left.id()}@{left.d} forward={forward.id()}@{forward.d}"
