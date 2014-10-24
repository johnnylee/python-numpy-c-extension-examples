
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

// Forward function declaration.
static PyObject *evolve(PyObject *self, PyObject *args); 

// Boilerplate: method list.
static PyMethodDef methods[] = {
  { "evolve", evolve, METH_VARARGS, "Doc string."},
  { NULL, NULL, 0, NULL } /* Sentinel */
};

// Boilerplate: Module initialization.
PyMODINIT_FUNC initsimple1(void) {
  (void) Py_InitModule("simple1", methods);
  import_array();
}

/*****************************************************************************
 * Array access macros.                                                      *
 *****************************************************************************/
#define m(x0) (*(npy_float64*)((PyArray_DATA(py_m) +                \
                                (x0) * PyArray_STRIDES(py_m)[0])))
#define m_shape(i) (py_m->dimensions[(i)])

#define r(x0, x1) (*(npy_float64*)((PyArray_DATA(py_r) +                \
                                    (x0) * PyArray_STRIDES(py_r)[0] +   \
                                    (x1) * PyArray_STRIDES(py_r)[1])))
#define r_shape(i) (py_r->dimensions[(i)])

#define v(x0, x1) (*(npy_float64*)((PyArray_DATA(py_v) +                \
                                    (x0) * PyArray_STRIDES(py_v)[0] +   \
                                    (x1) * PyArray_STRIDES(py_v)[1])))
#define v_shape(i) (py_v->dimensions[(i)])

#define F(x0, x1) (*(npy_float64*)((PyArray_DATA(py_F) +              \
                                    (x0) * PyArray_STRIDES(py_F)[0] +   \
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
  npy_float64 sx, sy, Fx, Fy, s3, tmp;
  
  // Set all forces to zero. 
  for(i = 0; i < N; ++i) {
    F(i, 0) = F(i, 1) = 0;
  }
  
  // Compute forces between pairs of bodies.
  for(i = 0; i < N; ++i) {
    for(j = i + 1; j < N; ++j) {
      sx = r(j, 0) - r(i, 0);
      sy = r(j, 1) - r(i, 1);

      s3 = sqrt(sx*sx + sy*sy);
      s3 *= s3 * s3;

      tmp = m(i) * m(j) / s3;
      Fx = tmp * sx;
      Fy = tmp * sy;

      F(i, 0) += Fx;
      F(i, 1) += Fy;

      F(j, 0) -= Fx;
      F(j, 1) -= Fy;
    }
  }
}

/*****************************************************************************
 * evolve                                                                    *
 *****************************************************************************/
static PyObject *evolve(PyObject *self, PyObject *args) {
  // Declare variables. 
  npy_int64 N, threads, steps, step, i;
  npy_float64 dt;
  PyArrayObject *py_m, *py_r, *py_v, *py_F;

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

  // Evolve the world. 
  for(step = 0;  step< steps; ++step) {
    compute_F(N, py_m, py_r, py_F);
    
    for(i = 0; i < N; ++i) {
      v(i, 0) += F(i, 0) * dt / m(i);
      v(i, 1) += F(i, 1) * dt / m(i);
      
      r(i, 0) += v(i, 0) * dt;
      r(i, 1) += v(i, 1) * dt;
    }
  }

  Py_RETURN_NONE;
}
