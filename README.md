# About NoisePy
NoisePy is a Python package designed for fast and easy ambient noise cross-correlation. In particular, this package xx for noise monitoring application by providing a series of functions to measure dv/v. 

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://github.come/mdenolle/NoisPy/latest) [![Build Status](https://travis-ci.org/mdenolle/Noise.jl.svg?branch=master)](https://travis-ci.org/mdenolle/NoisePy) [![Coverage Status](https://coveralls.io/repos/github/mdenolle/Noise.jl/badge.svg?branch=master)](https://coveralls.io/github/mdenolle/NoisePy?branch=master)

<img src="/docs/src/logo.png" width="800" height="400">
 
# Installation
This package contains 3 main python scripts with 1 dependent module named as `noise_module`. All the depended library are listed below, and we recommend to install them using [conda](https://docs.conda.io/en/latest/) or [pip](https://pypi.org/project/pip/). Due to the availablility of multiple version of dependent library, we did not exclusively tested their performance. But the version information provided below works well on `macOS Mojave (10.14.5)`. 

    |  **library**  |  **version**  |\
:------:|:----------:|:----------:|\
[numpy](https://numpy.org/)|  1.16.3|\
[scipy](https://www.scipy.org/) | 1.3.0|\
[numba](https://devblogs.nvidia.com/numba-python-cuda-acceleration/) | 0.44.1|\
[obspy](https://github.com/obspy/obspy/wiki) |1.1.1|\
[pandas](https://pandas.pydata.org/) | 0.24.2|\
[pyasdf](http://seismicdata.github.io/pyasdf/) |0.4.0|\
[python](https://www.python.org/) |3.7.3|\
[mpi4py](https://mpi4py.readthedocs.io/en/stable/) | 3.0.1|


# Functionality
* download continous noise data based on obspy's [mass download](https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.mass_downloader.html) module and save data in [ASDF](https://asdf-definition.readthedocs.io/en/latest/) format, which convinently assemble meta, wavefrom and auxililary data into one single file
* perform fast and easy cross-correlation for downloaded seismic data in ASDF format as 
well as those stored on local machine as SAC/miniSEED format
* do linear or phase weighted stacking (substacking) to the cross-correlation functions 
* all scripts are coded with MPI functionality to run in parallel
* several functions are provided to measure dv/v on the resulted cross-correlation functions

# Short tutorial
**1. Downloading seismic noise data (`S0_download_MPI.py`)**
    
    In this example, we aim to download all broadband CI stations around LA operated during 1/Jan/2008, and store the data as two chuncks, each with 12 h long continous recordings.  
    To do this, we set `inc_hours=12` in the script. Also, `down_list` is set to be `False` since no station info is provided, and the info on the targeted region is given at Lxx. `flag` should be `True` if intermediate outputs/operational time is needed during downloading process. To run the code on a single core, go to your terminal setup with a python environment with required library as suggested above and run `python S0_download_ASDF_MPI.py`  

    If you want to use multiple cores (e.g, 3), run the script with `mpirun -n 3 python S0_download_ASDF_MPI.py`\

![downloaded data](/docs/src/downloaded.png)

    Two files with 12 hour long continous recordings. The names are pretty straightforward to understand. (more details on reading the ASDF files with downloaded data can be found in docs/src/ASDF.md)\

    1b. download noise data for stations in a list\
    This time we try to download the data with a station list. For example, we use the station list outputed from example 1a to be here. To run this example, change the `down_list` to be `True`. This time, the region information will be useless.  

    (plotting script is provide to show the waveforms)

**2. Perform cross correlations (`S1_fft_cc_MPI.py`)**\
    This is the core script of NoisePy, which performs fft to the noise data for all of the data first before they are cross-correlated in frequency domain. Several options are provided for the cross correlation, including `raw`, `coherency` and `deconv`. We choose 'decon' as an example here.

    (show the cross-correlation functions from one single station) 

**3. Do stacking (`S2_stacking.py`)**\
    This script assembles all computed cross-correlation functions from S1, and performs final stacking (including substacking) of them. in particular, two options of linear and pws stacking methods are provided. 

    Below is an example to plot the move-out of the final stacked cross-correlation.
    ```python
    import plot_modules,glob
    sfiles = glob.glob('/Users/chengxin/Documents/NoisePy_example/SCAL/STACK/*/linear*.h5')
    plot_modules.plot_all_moveout1(sfiles,0.1,0.2,'ZZ',1,200,True,'/Users/chengxin/Documents/NoisePy_example/SCAL/STACK')
    ```
<img src="/docs/src/linear_stack.png" width="400" height="250"><img src="/docs/src/pws_stack.png" width="400" height="250">


