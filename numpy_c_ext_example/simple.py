
import numpy as np

dt = 0.1

def initialize(N, m_min=1, m_max=30.0, r_max=50.0, v_max=4.0):
    """Compute initial m, r, v, and F vectors."""
    m = np.random.uniform(m_min, m_max, N)
    r = np.random.uniform(-r_max, r_max, (N, 2))
    v = np.random.uniform(-v_max, v_max, (N, 2))
    F = np.zeros_like(r)
    return m, r, v, F


def compute_F(m, r, F):
    """Compute F inplace from m and r."""
    # Zero F. 
    F *= 0
    
    # Sum forces. 
    for i in xrange(len(m)):
        mi = m[i]
        ri = r[i]
        for j in xrange(i + 1, len(m)):
            mj = m[j]
            rj = r[j]
            rvec = rj - ri
            rnorm3 = (rvec[0]**2 + rvec[1]**2)**1.5 # Faster than linalg.norm.
            Fij = mi * mj * rvec / rnorm3
            F[i] += Fij
            F[j] -= Fij


def update_v(m, v, F):
    """Update v inplace: v = v + (F/m)*dt."""
    for i in xrange(len(m)):
        v[i] += (F[i] / m[i]) * dt


def update_r(r, v):
    """Update r inplace: r = r + v*dt."""
    r += v * dt


def evolve(steps, m, r, v, F):
    """Evolve the system in time over some number of steps."""
    for i in xrange(steps):
        compute_F(m, r, F); update_v(m, v, F); update_r(r, v)


def run_plot(ax, steps, m, r, v, F):
    """A function to produce an example plot."""
    rx = np.zeros((steps, (len(m))))
    ry = np.zeros((steps, (len(m))))
    
    for i in xrange(steps):
        compute_F(m, r, F); update_v(m, v, F); update_r(r, v)
        rx[i] = r[:,0]
        ry[i] = r[:,1]

    for i in xrange(rx.shape[1]):
        ax.plot(rx[:,i], ry[:,i], '.-')


if __name__ == "__main__":
    m, r, v, F = initialize(100)
    evolve(1000, m, r, v, F)
