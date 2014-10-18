
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <omp.h>

// Forward declarations of our function.
static PyObject *evolve(PyObject *self, PyObject *args); 

// Boilerplate: function list.
static PyMethodDef methods[] = {
  { "evolve", evolve, METH_VARARGS, "Doc string."},
  { NULL, NULL, 0, NULL } /* Sentinel */
};

// Boilerplate: Module initialization.
PyMODINIT_FUNC initomp1(void) {
  (void) Py_InitModule("omp1", methods);
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

#define Ft(x0, x1, x2) (*(npy_float64*)((PyArray_BYTES(py_Ft) +         \
                                         (x0) * PyArray_STRIDES(py_Ft)[0] + \
                                         (x1) * PyArray_STRIDES(py_Ft)[1] + \
                                         (x2) * PyArray_STRIDES(py_Ft)[2])))
#define Ft_shape(i) (py_Ft->dimensions[(i)])

/*****************************************************************************
 * compute_F                                                                 *
 *****************************************************************************/
static inline void compute_F(npy_int64 threads,
                             npy_int64 N,
                             PyArrayObject *py_m,
                             PyArrayObject *py_r,
                             PyArrayObject *py_F, 
                             PyArrayObject *py_Ft) {
  npy_int64 id, i, j;
  npy_float64 sx, sy, Fx, Fy, s3, tmp;

#pragma omp parallel for private(i, id)
  for(i = 0; i < N; i++) {
    for(id = 0; id < threads; ++id) {
      Ft(id, i, 0) = Ft(id, i, 1) = 0;
    }
  }

#pragma omp parallel for                        \
  private(id, i, j, sx, sy, Fx, Fy, s3, tmp)    \
  schedule(dynamic)

  for(i = 0; i < N; ++i) {

    F(i, 0) = F(i, 1) = 0;

    id = omp_get_thread_num();
    for(j = i + 1; j < N; ++j) {
      
      sx = r(j, 0) - r(i, 0);
      sy = r(j, 1) - r(i, 1);
      
      s3 = pow(sx*sx + sy*sy, 1.5);
      
      tmp = m(i) * m(j) / s3;
      
      Fx = tmp * sx;
      Fy = tmp * sy;
      
      Ft(id, i, 0) += Fx;
      Ft(id, i, 1) += Fy;
      
      Ft(id, j, 0) -= Fx;
      Ft(id, j, 1) -= Fy;
    }
  }
  
#pragma omp parallel for private(i, id)
  for(i = 0; i < N; ++i) {
    for(id = 0; id < threads; ++id) {
      F(i, 0) += Ft(id, i, 0);
      F(i, 1) += Ft(id, i, 1);
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

  PyArrayObject *py_m, *py_r, *py_v, *py_F, *py_Ft;

  // Parse variables. 
  if(!PyArg_ParseTuple(args, "ldllO!O!O!O!O!",
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

  npy_int64 step, i; 

  for(step = 0; step < steps; ++step) {

    compute_F(threads, N, py_m, py_r, py_F, py_Ft);
    
#pragma omp parallel for private(i)
    for(i = 0; i < N; ++i) {
      v(i, 0) += F(i, 0) * dt / m(i);
      v(i, 1) += F(i, 1) * dt / m(i);
      
      r(i, 0) += v(i, 0) * dt;
      r(i, 1) += v(i, 1) * dt;
    }
  }

  Py_RETURN_NONE; // Nothing to return. 
}
