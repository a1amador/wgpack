# Description:

This repository contains data querying, processing, and analysis tools for [CORDC](https://cordc.ucsd.edu/)'s Wave Glider fleet. 

**directory: wgpack/wgpack**
Contains data querying, processing, and analysis modules

**directory: wgpack/tests**
Contains test and example scripts

**directory: wgpack/data**
Contains data files for example scripts

**directory: wgpack/auto_plots**
Contains python scripts for automated plotting 

**directory: wgpack/mfiles**
Contains MATLAB code to enable alternative processing of the ADCP binaries 

For more information and documentation see:
Amador, A., Merrifield, S.T., Terrill, E.J. (2022). “Assessment of atmospheric and oceanographic measurements from an autonomous surface vehicle”. Journal of Atmospheric and Oceanic Technology. [https://doi.org/10.1175/JTECH-D- 22-0060.1](https://doi.org/10.1175/JTECH-D-22-0060.1)


---
# Getting Started:

## 1) Create a copy of the remote repository files on your computer:
`git clone https://seachest.ucsd.edu/waveglider/shoreside/toolbox/wgpack.git`

or 

`git clone https://github.com/a1amador/wgpack.git`

## 2) Create the environment from the wgpack_environment.yml file:
`conda create --name wgpack --file requirements.txt`

## 3) Create config.py and creds.py 
- wgpack/wgpack/config.py (copy from config_example.py) and define variables within it.
- wgpack/wgpack/creds.py (copy from creds_example.py) and define variables within it.

