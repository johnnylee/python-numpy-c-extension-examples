#!/usr/bin/env python

from distutils.core import setup, Extension

setup(name             = "numpy_c_ext_example",
      version          = "1.0",
      description      = "Example code for blog post.",
      author           = "J. David Lee",
      author_email     = "contact@crumpington.com",
      maintainer       = "contact@crumpington.com",
      url              = "https://www.crumpington.com",
      ext_modules      = [
          Extension(
              'lib.simple', ['src/simple.c'],
              extra_compile_args=["-Ofast", "-march=native"]),
          Extension(
              'lib.simd1', ['src/simd1.c'],
              extra_compile_args=["-Ofast", "-march=native"]),
          Extension(
              'lib.omp1', ['src/omp1.c'],
              extra_compile_args=["-Ofast", "-march=native", "-fopenmp"],
              libraries=["gomp"]),
          Extension(
              'lib.simdomp1', ['src/simdomp1.c'],
              extra_compile_args=["-Ofast", "-march=native", "-fopenmp"],
              libraries=["gomp"]),
      ], 
      
)
