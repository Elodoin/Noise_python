# About NoisePy
NoisePy is a Python package designed for fast and easy ambient noise cross-correlation.

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://github.come/mdenolle/NoisPy/latest) [![Build Status](https://travis-ci.org/mdenolle/Noise.jl.svg?branch=master)](https://travis-ci.org/mdenolle/NoisePy) [![Coverage Status](https://coveralls.io/repos/github/mdenolle/Noise.jl/badge.svg?branch=master)](https://coveralls.io/github/mdenolle/NoisePy?branch=master)

 
# Installation
This package contains 3 main scripts with 1 module named as noise_module. To install
it, simple go to src directory and run install.py. this will check whether the required
modules are installed, and do so if not. You can also find all dependencies in the 
requirement.txt file. 

# Functionality
* download continous noise data using Obspy modules in ASDF formate
* perform fast and easy cross-correlation for data downloaded using this package as 
well as those stored on local machine in SAC/miniSEED format
* do stacking (sub-stacking) of the cross-correlation functions for monitoring purpose

# Short tutorial

