
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

// Forward function declaration.
static PyObject *evolve(PyObject *self, PyObject *args); 

// Boilerplate: function list.
static PyMethodDef methods[] = {
  { "evolve", evolve, METH_VARARGS, "Doc string."},
  { NULL, NULL, 0, NULL } /* Sentinel */
};

// Boilerplate: Module initialization.
PyMODINIT_FUNC initsimple2(void) {
  (void) Py_InitModule("simple2", methods);
  import_array();
}

/*****************************************************************************
 * compute_F                                                                 *
 *****************************************************************************/
static inline void compute_F(npy_int64 N,
                             npy_float64 *m,
                             npy_float64 *r,
                             npy_float64 *F) {
  npy_int64 i, j, xi, yi, xj, yj;
  npy_float64 sx, sy, Fx, Fy, s3, tmp;
  
  // Set all forces to zero. 
  for(i = 0; i < N; ++i) {
    F[2*i] = F[2*i + 1] = 0;
  }
  
  // Compute forces between pairs of bodies.
  for(i = 0; i < N; ++i) {
    xi = 2*i;
    yi = xi + 1;
    for(j = i + 1; j < N; ++j) {
      xj = 2*j; 
      yj = xj + 1;

      sx = r[xj] - r[xi];
      sy = r[yj] - r[yi];

      s3 = sqrt(sx*sx + sy*sy);
      s3 *= s3*s3;

      tmp = m[i] * m[j] / s3;
      Fx = tmp * sx;
      Fy = tmp * sy;
      
      F[xi] += Fx;
      F[yi] += Fy;
      F[xj] -= Fx;
      F[yj] -= Fy;
    }
  }
}

/*****************************************************************************
 * evolve                                                                    *
 *****************************************************************************/
static PyObject *evolve(PyObject *self, PyObject *args) {
  // Declare variables. 
  npy_int64 N, threads, steps, step, i, xi, yi;
  npy_float64 dt;
  PyArrayObject *py_m, *py_r, *py_v, *py_F;
  npy_float64 *m, *r, *v, *F;

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
  r = (npy_float64*)PyArray_DATA(py_r);
  v = (npy_float64*)PyArray_DATA(py_v);
  F = (npy_float64*)PyArray_DATA(py_F);
  
  // Evolve the world. 
  for(step = 0; step < steps; ++step) {
    compute_F(N, m, r, F);
    
    for(i = 0; i < N; ++i) {
      xi = 2 * i;
      yi = xi + 1;

      v[xi] += F[xi] * dt / m[i];
      v[yi] += F[yi] * dt / m[i];
      
      r[xi] += v[xi] * dt;
      r[yi] += v[yi] * dt;
    }
  }

  Py_RETURN_NONE;
}
