# The hybridisation of WOMBATs  *(In Prep)*

Development and testing of hybrid WOMBAT on GADI  


## Introducing the WOMBATs  

Source code located at:  
*https://github.com/oetjensa/GFDL-generic-tracers/blob/wombatmid-hy/generic_tracers/generic_WOMBATlite_hy.F90*  

### WOMBATfull* 

WOMBATfull* is a dev model spin-off to include:  
- particle attached bacteria  
- prescribed particle sinking speeds  

### WOMBATfull*_hy  

WOMBATfull*_hy is the hybrid WOMBATfull*, with two logicals:  

- *do_shadow*: enables to call the trained MLP at runtime and store output in separate variable while the model is still driven with the original bacteria formulation  
- *do_emulator*: switch off bacteria and emulate remineralisation rate by calling the torchscript  

### WOMBATlite  

WOMBATlite is the off-the-shelf version used for the control run. The goal is to replace the temperature-dependent remineralisation rate with the emulator for a more realistic detritus loss.

### WOMBATlite_hy

WOMBATlite_hy is the hybrid WOMBATlite that can shadow and emulate the remineralisation rate.  

**Implement emulator subroutines**  

```Fortran
  subroutine emulator_init()
    ! Initialise FTorch model
    logical :: file_exists

    if (remin_ready) return

    inquire(file=trim(remin_emulator_path), exist=file_exists)
    if (.not. file_exists) call mpp_error(FATAL, &
         'generic_WOMBAT_lite: emulator file not found: '//trim(remin_emulator_path))

    call torch_model_load(remin_model, trim(remin_emulator_path), torch_kCPU)
    remin_ready = .true.

  end subroutine emulator_init

  subroutine emulator_prepare(nmax)
    integer, intent(in) :: nmax

    if (remin_buffers_ready) then
      if (remin_batch_size /= nmax) call mpp_error(FATAL, &
          'generic_WOMBAT_lite: valid-cell count changed after emulator initialization')
      return
    endif

    if (.not. remin_ready) call mpp_error(FATAL, &
        'generic_WOMBAT_lite: emulator model is not initialized')

    allocate(remin_x1(nmax, 4), remin_y1(nmax))
    remin_x1 = 1.0d0
    remin_y1 = 0.0d0
    call torch_tensor_from_array(remin_in(1), remin_x1, torch_kCPU)
    call torch_tensor_from_array(remin_out(1), remin_y1, torch_kCPU)
    remin_batch_size = nmax
    remin_buffers_ready = .true.

  end subroutine emulator_prepare

  subroutine emulator_remin_batch()
    if (remin_batch_size <= 0) return

    if (.not. remin_buffers_ready) call mpp_error(FATAL, &
        'generic_WOMBAT_lite: emulator buffers are not initialized')

    call torch_model_forward(remin_model, remin_in, remin_out)

  end subroutine emulator_remin_batch

  subroutine emulator_disable_traps()
    if (remin_traps_disabled) return

    call ieee_get_halting_mode(remin_trap_flags, remin_halting)
    remin_traps_saved = .true.
    call ieee_set_halting_mode(remin_trap_flags, .false.)
    remin_traps_disabled = .true.
  end subroutine emulator_disable_traps

  subroutine emulator_restore_traps()
    if (.not. remin_traps_disabled) return

    call ieee_set_flag(remin_trap_flags, .false.)
    if (remin_traps_saved) call ieee_set_halting_mode(remin_trap_flags, remin_halting)
    remin_traps_saved = .false.
    remin_traps_disabled = .false.
  end subroutine emulator_restore_traps

  subroutine emulator_remin(temp, o2, detr, zoos, rate)
    real(kind=8), intent(in)  :: temp, o2, detr, zoos
    real(kind=8), intent(out) :: rate
    real(kind=8) :: tempv(1), o2v(1), detrv(1), zoosv(1)

    tempv(1) = temp
    o2v(1)   = o2
    detrv(1) = detr
    zoosv(1) = zoos

    if (.not. remin_buffers_ready .or. remin_batch_size /= 1) call mpp_error(FATAL, &
        'generic_WOMBAT_lite: scalar emulator call requires a batch size of one')
    remin_x1(1,1) = tempv(1)
    remin_x1(1,2) = o2v(1)
    remin_x1(1,3) = detrv(1)
    remin_x1(1,4) = zoosv(1)
    call emulator_remin_batch()

    rate = remin_y1(1)

  end subroutine emulator_remin
```

**Emulate remineralisation rate**   

```Fortran
    if (do_emulator_shadow .or. do_emulator_run) then
      nvalid = 0
      do kk = 1, nk
        do jj = jsc, jec
          do ii = isc, iec
            if (kk <= grid_kmt(ii,jj) .and. grid_tmask(ii,jj,kk) > 0.0) then
              nvalid = nvalid + 1
            endif
          enddo
        enddo
      enddo
      if (nvalid > 0) then
        call emulator_prepare(nvalid)
        allocate(i_b(nvalid), j_b(nvalid), k_b(nvalid))
        ncell = 0
        do kk = 1, nk
          do jj = jsc, jec
            do ii = isc, iec
              if (kk <= grid_kmt(ii,jj) .and. grid_tmask(ii,jj,kk) > 0.0) then
                ncell = ncell + 1
                i_b(ncell) = ii
                j_b(ncell) = jj
                k_b(ncell) = kk
              endif
            enddo
          enddo
        enddo
      endif
    else
      nvalid = 0
    endif
```

```Fortran
    if ((do_emulator_shadow .or. do_emulator_run) .and. nvalid > 0) &
        call emulator_disable_traps()

    do tn = 1,ts_npzd  !{

      n_pools(:,:,:,1) = n_pools(:,:,:,2)
      c_pools(:,:,:,1) = c_pools(:,:,:,2)
      fe_pools(:,:,:,1) = fe_pools(:,:,:,2)

      if (do_emulator_shadow .or. do_emulator_run) then
        do ncell = 1, nvalid
          ii = i_b(ncell)
          jj = j_b(ncell)
          kk = k_b(ncell)
          remin_x1(ncell,1) = Temp(ii,jj,kk)
          remin_x1(ncell,2) = max(0.0, wombat%p_o2(ii,jj,kk,tau))  !/ mmol_m3_to_mol_kg
          remin_x1(ncell,3) = max(0.0, wombat%p_det(ii,jj,kk,tau)) !/ mmol_m3_to_mol_kg
          remin_x1(ncell,4) = max(0.0, wombat%p_zoo(ii,jj,kk,tau)) !/ mmol_m3_to_mol_kg
        enddo

        if (nvalid > 0) then
          call emulator_remin_batch()

          do ncell = 1, nvalid
            ii = i_b(ncell)
            jj = j_b(ncell)
            kk = k_b(ncell)

            if (do_emulator_shadow) then
              det_kg   = max(0.0, wombat%p_det(ii,jj,kk,tau))
              det_mmol = det_kg / mmol_m3_to_mol_kg
              rate     = min(remin_y1(ncell), det_kg/dtsb)
              wombat%reminr_nn(ii,jj,kk) = rate / (det_mmol**2 * mmol_m3_to_mol_kg + epsi)
              !wombat%reminr_nn(ii,jj,kk) = min(remin_y1(ncell), &
              !    max(0.0, wombat%p_det(ii,jj,kk,tau)) / mmol_m3_to_mol_kg / dtsb) / &
              !    (max(0.0, wombat%p_det(ii,jj,kk,tau)) / mmol_m3_to_mol_kg + epsi)
            endif

            if (do_emulator_run) then
              det_kg   = max(0.0, wombat%p_det(ii,jj,kk,tau))
              det_mmol = det_kg / mmol_m3_to_mol_kg
              rate     = min(remin_y1(ncell), det_kg/dtsb)
              wombat%reminr(ii,jj,kk) = rate / (det_mmol**2 * mmol_m3_to_mol_kg + epsi)
               ! wombat%reminr(ii,jj,kk) = min(remin_y1(ncell), &
               !     max(0.0, wombat%p_det(ii,jj,kk,tau)) / mmol_m3_to_mol_kg / dtsb) / &
               !    (max(0.0, wombat%p_det(ii,jj,kk,tau)) / mmol_m3_to_mol_kg + epsi)
            endif
          enddo
        endif
      endif

```

```Fortran

      ! Temperature dependance of heterotrophy (applies to bact and zoo)
      fbc = wombat%bbioh ** (Temp(i,j,k))

      ! Variable rates of remineralisation
      o2lim = max(0.0, min(1.0, 1.0 - exp(-o2_mmolm3)))
      
      if (.not. do_emulator_run) then
        wombat%reminr(i,j,k) = wombat%detlrem * fbc * o2lim
      endif
```



### Model configuration and forcing  

- WOMBATfull* was optimised to the *Southern Ocean Timeseries (SOTS)*   
- All experiments are run at SOTS in the [single water-column](https://github.com/ACCESS-Community-Hub/om3-mom6-single-column/tree/jra_iaf%2Bwombatlite) configuration  
- JRA55-do IAF 1958-2024 run on *cycle* to test for long-term stability (see config/forcing for details)  


**Experiment overview with more details below:**

|  Phase | model         | forcing        | mode    | do_shadow | do_emulator | Notebook | Notes                                                              |
|---|---------------|-----------------|---------|-----------|-------------|------------------|-------------------------------------------------------------------------------|
| 0 | PyTorch, FTorch |              |            |        |                                  |      | Install and configure PyTorch/FTorch on GADI/HPC          |
| 1 | WOMBATfull*    | iaf 1958-1990   | offline training | .false.   | .false. | Training-MLP | train MLP on poc1remi data, skip spin-up (3years)                                               |
| 2 | WOMBATfull*_hy  | iaf 1958-       | online shadow  | .true.    | .false. | Shadow-WOMBATfull    | use 'shadow' mode to test MLP behaviour online,  but still use original poc1remi to drive model |
| 3 | WOMBATfull*_hy  | iaf 1958-        | online emulator  | .false.   | .true. | Emulator-WOMBATfull     | use MLP online to drive model                                                                   |
| 4 | WOMBATlite    | iaf 1958-2030** | offline testing | .false.   | .false. | Control-WOMBATlite    | test offline performance                                                                        |
| 5 | WOMBATlite_hy | iaf 1958-2030** | online shadow  | .true.    | .false. |Shadow-WOMBATlite    | test MLP in shadow mode online                                                                  |
| 6 | WOMBATlite_hy | iaf 1958-2030   | online emulator | .false.   | .true. | Emulator-WOMBATlite     | use MLP online to drive model                                                                   |


## Phase 0: Setup PyTorch and FTorch on GADI  

### PyTorch

```bash
cd ftorch_model                      # create folder
module purge
module load python3/3.11.7           # load new python version
python3 -m venv torchvenv            # create virtual environment
source torchvenv/bin/activate
pip install --upgrade pip            # update pip

# install torch and torchvision, check for version:
# https://pytorch.org/get-started/locally/

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### FTorch  

*https://cambridge-iccs.github.io/FTorch/page/usage/generic_example.html*  

```bash
# load same compilers used for WOMBAT (ftx)
module load intel-compiler-llvm/2025.2.0 gcc/12.2.0 cmake
 
git clone https://github.com/Cambridge-ICCS/FTorch.git FTorch
mkdir build
cd build

# cmake build :
# https://cambridge-iccs.github.io/FTorch/page/installation/general.html
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_Fortran_COMPILER=ifx -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)") -DCMAKE_INSTALL_PREFIX=$(cd ../.. && pwd)/FTorch_install -Wno-dev

cmake --build .
cmake --install .

# location of config file:
find ../../FTorch_install -name FTorchConfig.cmake

# fresh console
# copy all files to test directory
 export DEPS=/g/data/jk72/ao2582/ACCESS/wombat_dev/ACCESS-OM3/ftorch_model

cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_Fortran_COMPILER=ifx -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_PREFIX_PATH="$DEPS/FTorch_install;$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" -DFTorch_DIR=$DEPS/FTorch_install/lib64/cmake/FTorch -Wno-dev

make VERBOSE=1 2>&1 | tee build_log.txt

./build/standin_wombat
```

- Changes in model code, see *https://github.com/oetjensa/GFDL-generic-tracers/blob/wombatmid-hy/README.md* for more details 

## Phase 1: Train model(s) on WOMBATfull* output data (offline)  

> Notebooks to perform analysis and intercomparison of different MLPs are: emulator_training

<img width="336" height="369" alt="image" src="https://github.com/user-attachments/assets/7cef5ccc-7656-4825-979f-b6d7834a0ab8" />

<img width="339" height="309" alt="image" src="https://github.com/user-attachments/assets/33b41f80-993e-4b98-8ec7-640fcd302e38" />


## Phase 2 \& 3: Testing of MLP within training model (online shadow and emulator mode)  

> Notebooks to perform analysis are: emulator_testing

<img width="1120" height="472" alt="image" src="https://github.com/user-attachments/assets/2f620589-eb0d-4dcc-b956-58bb620acd02" />

<img width="952" height="839" alt="image" src="https://github.com/user-attachments/assets/6a4578fc-6123-41b1-8ac9-b01ccd9422bc" />


## Phase  4 - 6: Run WOMBATlite_hy and evaluate performance

<img width="1051" height="675" alt="image" src="https://github.com/user-attachments/assets/07a91b6d-461b-4b60-a411-8907d664a926" />



