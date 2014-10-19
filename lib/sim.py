
import copy
import numpy as np

import simple1, simple2, simd1, omp1, simdomp1


class World(object):
    """World is a structure that holds the state of N bodies and
    additional variables.
    
    threads : (int) The number of threads to use for multithreaded
              implementations.
    
    STATE OF THE WORLD: 

    N : (int) The number of bodies in the simulation.
    m : (1D ndarray) The mass of each body.
    r : (2D ndarray) The position of each body.
    v : (2D ndarray) The velocity of each body.
    F : (2D ndarray) The force on each body.

    TEMPORARY VARIABLES:
    
    Ft : (3D ndarray) A 2D force array for each thread's local storage.
    s  : (2D ndarray) The vectors from one body to all others. 
    s3 : (1D ndarray) The norm of each s vector. 

    NOTE: Ft is used by parallel algorithms for thread-local
          storage. s and s3 are only used by the python
          implementation.
    """
    def __init__(self, N, threads=1, 
                 m_min=1, m_max=30.0, r_max=50.0, v_max=4.0, dt=1e-3):
        self.threads = threads
        self.N  = N
        self.m  = np.random.uniform(m_min, m_max, N)
        self.r  = np.random.uniform(-r_max, r_max, (N, 2))
        self.v  = np.random.uniform(-v_max, v_max, (N, 2))
        self.F  = np.zeros_like(self.r)
        self.Ft = np.zeros((threads, N, 2))
        self.s  = np.zeros_like(self.r)
        self.s3 = np.zeros_like(self.m)
        self.dt = dt


    def copy(self):
        return copy.deepcopy(self)
        

def compute_F(w):
    """Compute the force on each body in the world, w."""
    for i in xrange(w.N):
        w.s[:] = w.r - w.r[i]
        w.s3[:] = (w.s[:,0]**2 + w.s[:,1]**2)**1.5
        w.s3[i] = 1.0 # This makes the self-force zero.
        w.F[i] = (w.m[i] * w.m[:,None] * w.s / w.s3[:,None]).sum(0)


def evolve(w, steps):
    """Evolve the world, w, through the given number of steps."""
    for _ in xrange(steps):
        compute_F(w)
        w.v += w.F * w.dt / w.m[:,None]
        w.r += w.v * w.dt


def evolve_c_simple1(w, steps):
    """Evolve the world using the simple1 C module."""
    simple1.evolve(w.threads, w.dt, steps, w.N, w.m, w.r, w.v, w.F)


def evolve_c_simple2(w, steps):
    """Evolve the world using the simple2 C module."""
    simple2.evolve(w.threads, w.dt, steps, w.N, w.m, w.r, w.v, w.F)


def evolve_c_simd1(w, steps):
    """Evolve the world using the simd1 C module."""
    simd1.evolve(w.threads, w.dt, steps, w.N, w.m, w.r, w.v, w.F)


def evolve_c_omp1(w, steps):
    """Evolve the world using the omp1 C module."""
    omp1.evolve(w.threads, w.dt, steps, w.N, w.m, w.r, w.v, w.F, w.Ft)


def evolve_c_simdomp1(w, steps):
    """Evolve the world using the simdomp1 C module."""
    simdomp1.evolve(w.threads, w.dt, steps, w.N, w.m, w.r, w.v, w.F, w.Ft)
    
