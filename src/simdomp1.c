
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <x86intrin.h>
#include <omp.h>

// Forward declarations of our function.
static PyObject *evolve(PyObject *self, PyObject *args); 

// Boilerplate: function list.
static PyMethodDef methods[] = {
  { "evolve", evolve, METH_VARARGS, "Doc string."},
  { NULL, NULL, 0, NULL } /* Sentinel */
};

// Boilerplate: Module initialization.
PyMODINIT_FUNC initsimdomp1(void) {
  (void) Py_InitModule("simdomp1", methods);
  import_array();
}

/*****************************************************************************
 * compute_F                                                                 *
 *****************************************************************************/
static inline void compute_F(npy_int64 threads,
                             npy_int64 N,
                             npy_float64 *m,
                             __m128d *r,
                             __m128d *F, 
                             __m128d *Ft) {
  npy_int64 id, i, j, Nid;
  __m128d s, s2, tmp;
  npy_float64 s3;

#pragma omp parallel private(id, i, j, s, s2, s3, tmp, Nid) 
  {
    id = omp_get_thread_num();
    Nid = N * id; // Zero-index in thread-local array Ft. 
    
    // Zero out the thread-local force arrays. 
    for(i = 0; i < N; i++) {
      Ft[Nid + i] = _mm_set1_pd(0);
    }
    
    // Compute forces between pairs of bodies.
#pragma omp for schedule(dynamic, 8) 
    for(i = 0; i < N; ++i) {
      F[i] = _mm_set1_pd(0);
      
      for(j = i + 1; j < N; ++j) {
        
        s = r[j] - r[i];
        s2 = s * s;
        s3 = sqrt(s2[0] + s2[1]);
        s3 *= s3 * s3;
        
        tmp = s * m[i] * m[j] / s3;
        
        Ft[Nid + i] += tmp;
        Ft[Nid + j] -= tmp;
      }
    }
    
    // Sum the thread-local forces computed above.
#pragma omp for 
    for(i = 0; i < N; ++i) {
      for(id = 0; id < threads; ++id) {
        F[i] += Ft[N*id + i];
      }
    }
  }
}

/*****************************************************************************
 * evolve                                                                    *
 *****************************************************************************/
static PyObject *evolve(PyObject *self, PyObject *args) {
  // Variable declarations.
  npy_int64 N, threads, steps, step, i;
  npy_float64 dt;

  PyArrayObject *py_m, *py_r, *py_v, *py_F, *py_Ft;
  npy_float64 *m;
  __m128d *r, *v, *F, *Ft;

  // Parse arguments. 
  if (!PyArg_ParseTuple(args, "ldllO!O!O!O!O!",
                        &threads,
                        &dt,
                        &steps,
                        &N,
                        &PyArray_Type, &py_m,
                        &PyArray_Type, &py_r,
                        &PyArray_Type, &py_v,
                        &PyArray_Type, &py_F,
                        &PyArray_Type, &py_Ft)) {
    return NULL;
  }
  
  omp_set_num_threads(threads);
  
  // Get underlying arrays from numpy arrays. 
  m  = (npy_float64*)PyArray_DATA(py_m);
  r  = (__m128d*)PyArray_DATA(py_r);
  v  = (__m128d*)PyArray_DATA(py_v);
  F  = (__m128d*)PyArray_DATA(py_F);
  Ft = (__m128d*)PyArray_DATA(py_Ft);

  // Evolve the world. 
  for(step = 0; step < steps; ++step) {
    compute_F(threads, N, m, r, F, Ft);
    
#pragma omp parallel for private(i)
    for(i = 0; i < N; ++i) {
      v[i] += F[i] * dt / m[i];
      r[i] += v[i] * dt;
    }
  } 

  Py_RETURN_NONE; 
}
