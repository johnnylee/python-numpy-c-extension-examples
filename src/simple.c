
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

// Forward declarations of our function.
static PyObject *evolve(PyObject *self, PyObject *args); 

// Boilerplate: function list.
static PyMethodDef methods[] = {
  { "evolve", evolve, METH_VARARGS, "Doc string."},
  { NULL, NULL, 0, NULL } /* Sentinel */
};

// Boilerplate: Module initialization.
PyMODINIT_FUNC initsimple(void) {
  (void) Py_InitModule("simple", methods);
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
#define r_shape(i) (py_r->dimensions[(i)])

#define v(x0, x1) (*(npy_float64*)((PyArray_BYTES(py_v) + \
                                    (x0) * PyArray_STRIDES(py_v)[0] + \
                                    (x1) * PyArray_STRIDES(py_v)[1])))
#define v_shape(i) (py_v->dimensions[(i)])

#define F(x0, x1) (*(npy_float64*)((PyArray_BYTES(py_F) + \
                                    (x0) * PyArray_STRIDES(py_F)[0] + \
                                    (x1) * PyArray_STRIDES(py_F)[1])))
#define F_shape(i) (py_F->dimensions[(i)])


/*****************************************************************************
 * compute_F                                                                 *
 *****************************************************************************/
static inline void compute_F(npy_int64 N,
                             PyArrayObject *py_m,
                             PyArrayObject *py_r,
                             PyArrayObject *py_F) {
  npy_int64 i, j;
  npy_float64 sx, sy, s3, tmp;
  
  for(i = 0; i < N; ++i) {
    F(i, 0) = F(i, 1) = 0;
  }

  for(i = 0; i < N; ++i) {
    // Loop through all other particles to compute force. 
    for(j = i + 1; j < N; ++j) {
      sx = r(j, 0) - r(i, 0);
      sy = r(j, 1) - r(i, 1);

      s3 = sqrt(sx*sx + sy*sy);
      s3 *= s3 * s3;

      tmp = m(i) * m(j) / s3;

      F(i, 0) += tmp * sx;
      F(i, 1) += tmp * sy;

      F(j, 0) -= tmp * sx;
      F(j, 1) -= tmp * sy;
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
      v(j, 0) += F(j, 0) * dt / m(j);
      v(j, 1) += F(j, 1) * dt / m(j);
      
      r(j, 0) += v(j, 0) * dt;
      r(j, 1) += v(j, 1) * dt;
    }
  }

  Py_RETURN_NONE; // Nothing to return. 
}
