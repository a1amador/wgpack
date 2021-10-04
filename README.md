# Description:

This repository contains data querying, processing, and analysis tools for CORDC's Wave Glider fleet. 

**directory: wgpack/wgpack**
Contains data querying, processing, and analysis modules

**directory: wgpack/mfiles**
Contains MATLAB code to enable the processing of ADCP binaries 

**directory: wgpack/tests**
Contains test and example scripts

**directory: wgpack/data**
Contains data files for example scripts

---
# Installation:

## 1) Create a copy of the remote repository files on your computer in ~/src/:
`git clone seachest-git:a1amador/wgpack.git`

## 2) Create the environment from the wgpack_environment.yml file:
`conda env create -f wgpack_environment.yml`

## 3) Create config.py and creds.py 
- wgpack/wgpack/config.py (copy from config_example.py) and define variables within it.
- wgpack/wgpack/creds.py (copy from creds_example.py) and define variables within it.

