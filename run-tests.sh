#!/bin/bash

rm lib/*.so
python ./setup.py build_ext --inplace &&
python ./run-tests.py
