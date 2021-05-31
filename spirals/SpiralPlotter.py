import math
import matplotlib.pyplot as plt
import numpy as np

class SpiralPlotter:

    def I(self, n):
        return ((0,0), None, None, None) # wrong

    def XY(self, p):
        return p # wrong

    def SUCC_V(self, v, xy):
        return v # wrong

    def SUCC_P(self, p, d):
        return p # wrong

    def SUCC_D(self, d, r):
        return d # wrong

    def C(self, q, v):
        return False # wrong

    def spiral(self, L, n):
        P,D,R,V=self.I(n)

        while L > 0:
            L = L -1

            xy = self.XY(P)
            V = self.SUCC_V(V, xy)
            yield xy

            P = self.SUCC_P(P, D)

            e = self.SUCC_D(D, R)
            q = self.SUCC_P(P, e)

            if not self.C(self.XY(q), V):
                D = e

    def plot_spiral(self, n, s):
        v = np.array([(np.array(p)) for p in s])
        fig = plt.figure(figsize=(6,6))
        ax=plt.axes()
        #ax.set_facecolor('#f1c738')
        plt.title(f"n={n}")
        plt.gca().set_aspect("equal")
        plt.plot(v[:,0], v[:,1])
        plt.scatter(v[:,0], v[:,1])
        plt.show()