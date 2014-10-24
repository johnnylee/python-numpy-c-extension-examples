
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
 * compute_F                                                                 *
 *****************************************************************************/
static inline void compute_F(npy_int64 threads,
                             npy_int64 N,
                             npy_float64 *m,
                             npy_float64 *r,
                             npy_float64 *F, 
                             npy_float64 *Ft) {
  npy_int64 id, i, j, xi, yi, xj, yj, Nid;
  npy_float64 sx, sy, Fx, Fy, s3, tmp;

#pragma omp parallel private(id, i, j, xi, yi, xj, yj, Nid, sx, sy, Fx, Fy, s3, tmp)
  {

    id = omp_get_thread_num();
    Nid = 2 * N * id; // Zero-index in thread-local array Ft. 
  
    // Zero out the thread-local force arrays. 
    for(i = 0; i < N; i++) {
      xi = 2*(N*id + i);
      yi = xi + 1;
      Ft[xi] = Ft[yi] = 0;
    }
    
    // Compute forces between pairs of bodies. 
#pragma omp for schedule(dynamic, 8)
    for(i = 0; i < N; ++i) {
      xi = 2*i;
      yi = xi + 1;
      
      F[xi] = F[yi] = 0;
      
      for(j = i + 1; j < N; ++j) {
        xj = 2*j;
        yj = xj + 1;
        
        sx = r[xj] - r[xi];
        sy = r[yj] - r[yi];
        
        s3 = sqrt(sx*sx + sy*sy);
        s3 *= s3 * s3;
        
        tmp = m[i] * m[j] / s3;
        Fx = tmp * sx;
        Fy = tmp * sy;
        
        Ft[Nid + xi] += Fx;
        Ft[Nid + yi] += Fy;
        Ft[Nid + xj] -= Fx;
        Ft[Nid + yj] -= Fy;
      }
    }
    
    // Sum the thread-local forces computed above.
#pragma omp for 
    for(i = 0; i < N; ++i) {
      xi = 2*i;
      yi = xi + 1;
      for(id = 0; id < threads; ++id) {
        xj = 2*(N*id + i);
        yj = xj + 1;
        F[xi] += Ft[xj];
        F[yi] += Ft[yj];
      }
    }
  }
}
  
/*****************************************************************************
 * evolve                                                                    *
 *****************************************************************************/
static PyObject *evolve(PyObject *self, PyObject *args) {
  // Variable declarations.
  npy_int64 N, threads, steps, step, i, xi, yi;
  npy_float64 dt;

  PyArrayObject *py_m, *py_r, *py_v, *py_F, *py_Ft;
  npy_float64 *m, *r, *v, *F, *Ft;

  // Parse arguments. 
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
  
  // Get underlying arrays from numpy arrays. 
  m  = (npy_float64*)PyArray_DATA(py_m);
  r  = (npy_float64*)PyArray_DATA(py_r);
  v  = (npy_float64*)PyArray_DATA(py_v);
  F  = (npy_float64*)PyArray_DATA(py_F);
  Ft = (npy_float64*)PyArray_DATA(py_Ft);

  // Evolve the world.
  for(step = 0; step < steps; ++step) {
    compute_F(threads, N, m, r, F, Ft);
    
#pragma omp parallel for private(i, xi, yi)
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
