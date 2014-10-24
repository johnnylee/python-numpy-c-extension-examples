
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <x86intrin.h>

// Forward declarations of our function.
static PyObject *evolve(PyObject *self, PyObject *args); 

// Boilerplate: function list.
static PyMethodDef methods[] = {
  { "evolve", evolve, METH_VARARGS, "Doc string."},
  { NULL, NULL, 0, NULL } /* Sentinel */
};

// Boilerplate: Module initialization.
PyMODINIT_FUNC initsimd1(void) {
  (void) Py_InitModule("simd1", methods);
  import_array();
}

/*****************************************************************************
 * compute_F                                                                 *
 *****************************************************************************/
static inline void compute_F(npy_int64 N,
                             npy_float64 *m,
                             __m128d *r,
                             __m128d *F) {
  npy_int64 i, j;
  __m128d s, s2, tmp;
  npy_float64 s3;
  
  // Set all forces to zero.
  for(i = 0; i < N; ++i) {
    F[i] = _mm_set1_pd(0);
  }
  
  // Compute forces between pairs of bodies. 
  for(i = 0; i < N; ++i) {
    for(j = i + 1; j < N; ++j) {
      s = r[j] - r[i];

      s2 = s * s;
      s3 = sqrt(s2[0] + s2[1]);
      s3 *= s3 * s3;

      tmp = s * m[i] * m[j] / s3;
      F[i] += tmp;
      F[j] -= tmp;
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

  PyArrayObject *py_m, *py_r, *py_v, *py_F;
  npy_float64 *m;
  __m128d *r, *v, *F;

  // Parse arguments. 
  if (!PyArg_ParseTuple(args, "ldllO!O!O!O!",
                        &threads,
                        &dt,
                        &steps,
                        &N,
                        &PyArray_Type, &py_m,
                        &PyArray_Type, &py_r,
                        &PyArray_Type, &py_v,
                        &PyArray_Type, &py_F)) {
    return NULL;
  }
  
  // Get underlying arrays from numpy arrays. 
  m = (npy_float64*)PyArray_DATA(py_m);
  r = (__m128d*)PyArray_DATA(py_r);
  v = (__m128d*)PyArray_DATA(py_v);
  F = (__m128d*)PyArray_DATA(py_F);
  
  // Evolve the world.
  for(step = 0; step < steps; ++step) {
    compute_F(N, m, r, F);
    
    for(i = 0; i < N; ++i) {
      v[i] += F[i] * dt / m[i];
      r[i] += v[i] * dt;
    }
  }

  Py_RETURN_NONE;
}
