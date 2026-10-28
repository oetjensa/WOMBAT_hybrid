# The hybridisation of WOMBATs  *(In Prep)*

Development and testing of hybrid WOMBAT on GADI  

**Experiment overview with more details below:**

|  Phase | model         | forcing         | mode    | do_shadow | do_emulator | Notes                                                                                           |
|---|---------------|-----------------|---------|-----------|-------------|-------------------------------------------------------------------------------------------------|
| 1 | WOMBATmid*    | iaf 1958-1990   | offline training | .false.   | .false.     | train MLP on poc1remi data, skip spin-up (3years)                                               |
| 2 | WOMBATmid_hy  | iaf 1958-       | online shadow  | .true.    | .false.     | use 'shadow' mode to test MLP behaviour online,  but still use original poc1remi to drive model |
| 3 | WOMBATmid_hy  | iaf 1958        | online emulator  | .false.   | .true.      | use MLP online to drive model                                                                   |
| 4 | WOMBATlite    | iaf 1958-2030** | offline testing | .false.   | .false.     | test offline performance                                                                        |
| 5 | WOMBATlite_hy | iaf 1958-2030** | online shadow  | .true.    | .false.     | test MLP in shadow mode online                                                                  |
| 6 | WOMBATlite_hy | iaf 1958-2030   | online emulator | .false.   | .true.      | use MLP online to drive model                                                                   |


## Introducing the WOMBATs  

Source code located at:  
*https://github.com/oetjensa/GFDL-generic-tracers/blob/wombatmid-hy/generic_tracers/generic_WOMBATmid.F90*  

### WOMBATmid*  (!this changed to WOMBAT-full now!)

WOMBATmid* is a dev model spin-off to include:  
- particle attached bacteria  
- prescribed particle sinking speeds  

### WOMBATmid_hy  

WOMBATmid_hy is the hybrid WOMBATmid*, with two logicals:  

- *do_shadow*: enables to call the trained MLP at runtime and store output in separate variable while the model is still driven with the original bacteria formulation  
- *do_emulator*: switch off bacteria and emulate remineralisation rate by calling the torchscript  

### WOMBATlite  

WOMBATlite is the off-the-shelf version used for the control run. The goal is to replace the temperature-dependent remineralisation rate with the emulator for a more realistic detritus loss.

### WOMBATlite_hy

WOMBATlite_hy is the hybrid WOMBATlite that can shadow and emulate the remineralisation rate.  

The same logicals as in WOMBATmid_hy are incorporated to allow testing offline and at run-time.  



### Model configuration and forcing  

- WOMBATmid was optimised to the *Southern Ocean Timeseries (SOTS)*   
- All experiments are run at SOTS in the [single water-column](https://github.com/ACCESS-Community-Hub/om3-mom6-single-column/tree/jra_iaf%2Bwombatlite) configuration  
- JRA55-do IAF 1958-2024 run on *cycle* to test for long-term stability (see config/forcing for details)  


## Phase 0: Setup PyTorch and FTorch on GADI  

### PyTorch

- create folder  
> cd ftorch_model                       
> module purge  
- load new python version  
> module load python3/3.11.7
- create virtual environment   
> python3 -m venv torchvenv            
> source torchvenv/bin/activate
-  update pip  
> pip install --upgrade pip            

**install torch and torchvision, check for version:**  
*https://pytorch.org/get-started/locally/*  

> pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu  


### FTorch  

*https://cambridge-iccs.github.io/FTorch/page/usage/generic_example.html*  

- load same compilers used for WOMBAT (ftx)  
> module load intel-compiler-llvm/2025.2.0 gcc/12.2.0 cmake  
> git clone https://github.com/Cambridge-ICCS/FTorch.git FTorch  
> mkdir build  
> cd build  

- cmake build  
*https://cambridge-iccs.github.io/FTorch/page/installation/general.html*  
> cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_Fortran_COMPILER=ifx -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)") -DCMAKE_INSTALL_PREFIX=$(cd ../.. && pwd)/FTorch_install -Wno-dev  

> cmake --build .  
> cmake --install .  

- location of config file:  
> find ../../FTorch_install -name FTorchConfig.cmake  

- copy (fresh console) all files to test directory  
 > export DEPS=/g/data/jk72/ao2582/ACCESS/wombat_dev/ACCESS-OM3/ftorch_model  

> cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_Fortran_COMPILER=ifx -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_PREFIX_PATH="$DEPS/FTorch_install;$(python -c 'import torch;
>  print(torch.utils.cmake_prefix_path)')" -DFTorch_DIR=$DEPS/FTorch_install/lib64/cmake/FTorch -Wno-dev  

> make VERBOSE=1 2>&1 | tee build_log.txt  

- test configuration:  
> ./build/standin_wombat  

- TODO: Adjust CMakefile in gtracers  
- TODO: hard code paths to torch, ftorch  



- Changes for Runtime  
 Input: set logicals, path to .pt  
 config file: Update EXE  

## Phase 1: Train model(s) on WOMBATmid output data (offline)  

> Notebooks to perform analysis and intercomparison of different MLPs are: emulator_training  

## Phase 2 \& 3: Testing of MLP within training model (online shadow and emulator mode)  

> Notebooks to perform analysis are: emulator_testing  

## Phase  4 - 6: Run WOMBATlite_hy and evaluate performance



