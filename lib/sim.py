
import copy
import numpy as np

import simple, simd1, omp1, simdomp1

class World(object):
    """World is a structure used to hold all of our state:
    
    STATE OF THE WORLD: 

    N  : The number of bodies in the simulation. 

    m  : The mass of each body. 

    r  : The 2D coordinates of the each body. 
    v  : The 2D velocity of each body. 
    F  : The 2D force on each body. 

    TEMPORARY VARIABLES:

    Ft : The 2D force on each body for each thread. This is used for
         thread-local storage in parallel algorithms.
    s  : The 2D distances between two bodies.
    s3 : The 1D norm of the distance between two bodies cubed.
    """
    def __init__(self, N, threads=1, 
                 m_min=1, m_max=30.0, r_max=50.0, v_max=4.0, dt=1e-3):
        self.N = N
        self.threads = threads

        self.m = np.random.uniform(m_min, m_max, N)

        self.r = np.random.uniform(-r_max, r_max, (N, 2))
        self.v = np.random.uniform(-v_max, v_max, (N, 2))
        self.F = np.zeros_like(self.r)

        self.Ft = np.zeros((threads, N, 2))
        self.s = np.zeros_like(self.r)
        self.s3 = np.zeros_like(self.m)

        self.dt = dt


    def copy(self):
        return copy.deepcopy(self)
        

def compute_F(w):
    """Compute the force on each body in the world."""
    for i in xrange(w.N):
        w.s[:] = w.r - w.r[i]
        w.s3[:] = (w.s[:,0]**2 + w.s[:,1]**2)**1.5
        w.s3[i] = 1.0
        w.F[i] = (w.m[i] * w.m[:,None] * w.s / w.s3[:,None]).sum(0)


def evolve(w, steps):
    """Evolve the world through a given number of steps."""
    for _ in xrange(steps):
        compute_F(w)
        w.v += w.F * w.dt / w.m[:,None]
        w.r += w.v * w.dt


def evolve_c_simple(w, steps):
    """Evolve the world using the simple C module."""
    simple.evolve(w.threads, w.dt, steps, w.N, w.m, w.r, w.v, w.F)


def evolve_c_simd1(w, steps):
    """Evolve the world using the simd1 C module."""
    simd1.evolve(w.threads, w.dt, steps, w.N, w.m, w.r, w.v, w.F)


def evolve_c_omp1(w, steps):
    """Evolve the world using the omp1 C module."""
    omp1.evolve(w.threads, w.dt, steps, w.N, w.m, w.r, w.v, w.F, w.Ft)


def evolve_c_simdomp1(w, steps):
    """Evolve the world using the omp1 C module."""
    simdomp1.evolve(w.threads, w.dt, steps, w.N, w.m, w.r, w.v, w.F, w.Ft)
    
