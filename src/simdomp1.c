
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

#define Ft(x0, x1, x2) (*(npy_float64*)((PyArray_BYTES(py_Ft) +         \
                                         (x0) * PyArray_STRIDES(py_Ft)[0] + \
                                         (x1) * PyArray_STRIDES(py_Ft)[1] + \
                                         (x2) * PyArray_STRIDES(py_Ft)[2])))
#define Ftv(x0, x1) (*(__m128d*)((PyArray_BYTES(py_Ft) +         \
                                  (x0) * PyArray_STRIDES(py_Ft)[0] +    \
                                  (x1) * PyArray_STRIDES(py_Ft)[1])))

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
  __m128d s, tmp;
  npy_float64 s3;
  
  // Zero out the thread-local force arrays. 
#pragma omp parallel for private(i, id)
  for(i = 0; i < N; i++) {
    for(id = 0; id < threads; ++id) {
      Ftv(id, i) = _mm_set1_pd(0);
    }
  }

  // Compute the interaction forces.
#pragma omp parallel for                        \
  private(id, i, j, s, tmp, s3)                 \
  schedule(dynamic)

  for(i = 0; i < N; ++i) {
    
    Fv(i) = _mm_set1_pd(0);

    id = omp_get_thread_num();

    for(j = i + 1; j < N; ++j) {
      
      s = rv(j) - rv(i);
      s3 = pow(s[0]*s[0] + s[1]*s[1], 1.5);
      
      tmp = s * m(i) * m(j) / s3;
      
      Ftv(id, i) += tmp;
      Ftv(id, j) -= tmp;
    }
  }
  
  // Sum the thread-local forces computed above.
#pragma omp parallel for private(i, id)
  for(i = 0; i < N; ++i) {
    for(id = 0; id < threads; ++id) {
      Fv(i) += Ftv(id, i);
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

  npy_int64 i, j; 

  for(i = 0; i < steps; ++i) {

    compute_F(threads, N, py_m, py_r, py_F, py_Ft);
    
#pragma omp parallel for private(j)
    for(j = 0; j < N; ++j) {
      vv(j) += Fv(j) * dt / m(j);
      rv(j) += vv(j) * dt;
    }
  } 

  Py_RETURN_NONE; // Nothing to return. 
}
