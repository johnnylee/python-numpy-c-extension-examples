
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
 * Array access macros.                                                      *
 *****************************************************************************/
#define m(x0) (*(npy_float64*)((PyArray_BYTES(py_m) + \
                                (x0) * PyArray_STRIDES(py_m)[0])))
#define m_shape(i) (py_m->dimensions[(i)])

#define r(x0, x1) (*(npy_float64*)((PyArray_BYTES(py_r) + \
                                    (x0) * PyArray_STRIDES(py_r)[0] + \
                                    (x1) * PyArray_STRIDES(py_r)[1])))
#define rv(x0) (*(__m128d*)((PyArray_BYTES(py_r) +              \
                             (x0) * PyArray_STRIDES(py_r)[0])))
#define r_shape(i) (py_r->dimensions[(i)])

#define v(x0, x1) (*(npy_float64*)((PyArray_BYTES(py_v) + \
                                    (x0) * PyArray_STRIDES(py_v)[0] + \
                                    (x1) * PyArray_STRIDES(py_v)[1])))
#define vv(x0) (*(__m128d*)((PyArray_BYTES(py_v) +              \
                             (x0) * PyArray_STRIDES(py_v)[0])))
#define v_shape(i) (py_v->dimensions[(i)])

#define F(x0, x1) (*(npy_float64*)((PyArray_BYTES(py_F) + \
                                    (x0) * PyArray_STRIDES(py_F)[0] + \
                                    (x1) * PyArray_STRIDES(py_F)[1])))
#define Fv(x0) (*(__m128d*)((PyArray_BYTES(py_F) +              \
                             (x0) * PyArray_STRIDES(py_F)[0])))
#define F_shape(i) (py_F->dimensions[(i)])


/*****************************************************************************
 * compute_F                                                                 *
 *****************************************************************************/
static inline void compute_F(npy_int64 N,
                             PyArrayObject *py_m,
                             PyArrayObject *py_r,
                             PyArrayObject *py_F) {
  npy_int64 i, j;
  __m128d s, tmp;
  npy_float64 s3;
  
  for(i = 0; i < N; ++i) {
    Fv(i) = _mm_set1_pd(0); // Zero Fv(i). 
  }
  
  for(i = 0; i < N; ++i) {
    // Loop through all other particles to compute force. 
    for(j = i + 1; j < N; ++j) {
      s = rv(j) - rv(i);
      s3 = pow(s[0]*s[0] + s[1]*s[1], 1.5);
      tmp = s * m(i) * m(j) / s3;
      Fv(i) += tmp;
      Fv(j) -= tmp;
    }
  }
}

/*****************************************************************************
 * evolve                                                                    *
 *****************************************************************************/
static PyObject *evolve(PyObject *self, PyObject *args) {
  // Variable declarations.
  npy_int64 N, threads, steps;
  npy_float64 dt;

  PyArrayObject *py_m, *py_r, *py_v, *py_F;

  // Parse variables. 
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

  npy_int64 i, j; 

  for(i = 0; i < steps; ++i) {
    compute_F(N, py_m, py_r, py_F);
    
    for(j = 0; j < N; ++j) {
      vv(j) += Fv(j) * dt / m(j);
      rv(j) += vv(j) * dt;
    }
  }

  Py_RETURN_NONE; // Nothing to return. 
}
