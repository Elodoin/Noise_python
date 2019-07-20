# About NoisePy
NoisePy is a Python package designed for fast and easy ambient noise cross-correlation.

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://github.come/mdenolle/NoisPy/latest) [![Build Status](https://travis-ci.org/mdenolle/Noise.jl.svg?branch=master)](https://travis-ci.org/mdenolle/NoisePy) [![Coverage Status](https://coveralls.io/repos/github/mdenolle/Noise.jl/badge.svg?branch=master)](https://coveralls.io/github/mdenolle/NoisePy?branch=master)
------------------------------------------------------
 
# Installation
This package contains 3 main python scripts with 1 dependent module named as noise_module. To install
it, go to src directory and run install.py, which is simply checking whether the requried library are installed in the local machine or not. Due to the availablility of multiple version of dependent library,
we provide a list of library below that works well on macOS Mojave (10.14.5) for your reference. 

  **library**    **version**\
  [python](https://www.python.org/)  -> 3.7.3\
  [numpy](https://numpy.org/)  -> 1.16.3\
  [scipy](https://www.scipy.org/)    -> 1.3.0\
  [pandas](https://pandas.pydata.org/)  -> 0.24.2\
  [obspy](https://github.com/obspy/obspy/wiki)   -> 1.1.1\
  [pyasdf](http://seismicdata.github.io/pyasdf/)  -> 0.4.0\
  [numba](https://devblogs.nvidia.com/numba-python-cuda-acceleration/)  -> 0.44.1\
  [mpi4py](https://mpi4py.readthedocs.io/en/stable/)  -> 3.0.1


# Functionality
* download continous noise data using Obspy modules and save data in [ASDF](https://asdf-definition.readthedocs.io/en/latest/) format
* perform fast and easy cross-correlation for downloaded seismic data using this package as 
well as those stored on local machine in SAC/miniSEED format
* do stacking (sub-stacking) of the cross-correlation functions for monitoring purpose

# Short tutorial
1. Downloading seismic noise data
We have two ways. 

1a. aim at noise data in a region without prior station info
In this example, we try to download all the broadband CI stations around LA operated at 1/Jan/2008, and store the data as two chuncks, each with 12 h long continous recordings (inc_hours=12). In the script, we set the `down_list` to be `False` and fill the region info at Lxx. Set `flag` to be `True` if you want to see intermediate outputs during download. To run the code on a single core, go to your terminal setup with a python environment with required library as suggested above and run `python S0_download_ASDF_MPI.py`

If you want to use multiple cores (e.g, 3), run the script with `mpirun -n 3 python S0_download_ASDF_MPI.py`
(more details on reading the download seismic data in ASDF file can be found in xxx)

![downloaded data](/docs/src/downloaded.png)

1b. aim at noise data listed in a station list

2. Perform cross correlations
Several options for the cross correlation methods. We choose 'decon' as an example here.

3. Do stacking


