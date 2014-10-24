
import os
import time
import multiprocessing as mp

import numpy as np

import lib


def test_fn(evolve_fn, name, steps=1000, dt=1e-3, bodies=101, threads=1):
    print "\n"

    # Test the speed of the evolution function. 
    w = lib.World(bodies, threads=threads, dt=dt)
    
    t0 = time.time()
    evolve_fn(w, steps)
    t1 = time.time()
    print "{0} ({1}): {2} steps/sec".format(
        name, threads, int(steps / (t1 - t0)))
    
    # Compare the evolution function to the pure python version. 
    w1 = lib.World(10, threads=threads, dt=dt)
    w2 = w1.copy()
    
    lib.evolve(w1, 1024)
    evolve_fn(w2, 1024)
    
    def f(name):
        wA = w1
        wB = w2
        dvmax = eval("np.absolute(wA.{0} - wB.{0}).max()".format(name))
        print("    max(delta {0}): {1:2.2}".format(name, dvmax))
        
    f("r")
    f("v")
    f("F")
    

if __name__ == "__main__":
    # Single CPU only tests. 
    test_fn(lib.evolve, "Python", steps=512)
    
    test_fn(lib.evolve_c_simple1, "C Simple 1", steps=32000)
    test_fn(lib.evolve_c_simple2, "C Simple 2", steps=32000)
    test_fn(lib.evolve_c_simd1, "C SIMD 1", steps=32000)
    
    # Multi-threaded tests. 
    threads = 0

    while True:

        threads += 1
        if threads > mp.cpu_count() + 2:
            break

        steps = threads * 32000

        test_fn(
            lib.evolve_c_omp1, "C OpenMP 1", steps=steps, threads=threads)
        test_fn(
            lib.evolve_c_simdomp1, "C SIMD OpenMP 1", 
            steps=steps, threads=threads)

    print "\n"

