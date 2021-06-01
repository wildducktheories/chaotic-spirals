import math 
import numpy as np

from .SpiralPlotter import SpiralPlotter

class HiddenSolution(SpiralPlotter):
    def I(self, n):
        return (
            complex(0,0), # P
            complex(1,0), # D
            complex(np.cos(2*math.pi/n), np.sin(2*math.pi/n)), # R
            set() # V
        )

    def XY(self, p):
        return (round(p.real, 8), round(p.imag, 8))

    def SUCC_V(self, v, p):
        v.add(p)
        return v

    def SUCC_P(self, p, d):
        return p+d

    def SUCC_D(self, d, r):
        return d*r

    def C(self, q, v):
        return q in v
