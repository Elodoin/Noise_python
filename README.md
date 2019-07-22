# About NoisePy
NoisePy is a Python package designed for fast and easy ambient noise cross-correlation.

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://github.come/mdenolle/NoisPy/latest) [![Build Status](https://travis-ci.org/mdenolle/Noise.jl.svg?branch=master)](https://travis-ci.org/mdenolle/NoisePy) [![Coverage Status](https://coveralls.io/repos/github/mdenolle/Noise.jl/badge.svg?branch=master)](https://coveralls.io/github/mdenolle/NoisePy?branch=master)

![noisepy logo](/docs/src/logo.png | width=200)

---------
 
# Installation
This package contains 3 main python scripts with 1 dependent module named as `noise_module`. To install
it, go to `src` directory and run `install.py`, which is simply checking whether the requried library are installed in the local machine or not. Due to the availablility of multiple version of dependent library,
we provide a list of library below that works well on `macOS Mojave (10.14.5)` for your reference. 

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
well as those stored on local machine as SAC/miniSEED format
* do stacking (sub-stacking) of the cross-correlation functions 
* provide several options to measure dv/v on the resulted cross-correlation functions

# Short tutorial
1. Downloading seismic noise data (`S0_download_MPI.py`)

    1a. interested in noise data from a region without prior station info\
    In this example, we aim to download all broadband CI stations around LA operated during 1/Jan/2008, and store the data as two chuncks, each with 12 h long continous recordings (set `inc_hours=12`). In the script, we also set `down_list` to be `False` since no station info is provided, and fill the region info at Lxx. The option of `flag` should be set to `True` if intermediate outputs/operational time is needed during the download process. To run the code on a single core, go to your terminal setup with a python environment with required library as suggested above and run `python S0_download_ASDF_MPI.py`  

    If you want to use multiple cores (e.g, 3), run the script with `mpirun -n 3 python S0_download_ASDF_MPI.py`\

![downloaded data](/docs/src/downloaded.png)

    Two files with 12 hour long continous recordings. The names are pretty straightforward to understand. (more details on reading the ASDF files with downloaded data can be found in docs/src/ASDF.md)\

    1b. download noise data for stations in a list\
    This time we try to download the data with a station list. For example, we use the station list outputed from example 1a to be here. To run this example, change the `down_list` to be `True`. This time, the region information will be useless.  

    (plotting script is provide to show the waveforms)

2. Perform cross correlations (`S1_fft_cc_MPI.py`)\
    This is the core script of NoisePy, which performs fft to the noise data for all of the data first before they are cross-correlated in frequency domain. Several options are provided for the cross correlation, including `raw`, `coherency` and `deconv`. We choose 'decon' as an example here.

    (show the cross-correlation functions from one single station) 

3. Do stacking (`S2_stacking.py`)\
    This script 


