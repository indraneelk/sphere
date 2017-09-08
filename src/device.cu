// device.cu -- GPU specific operations utilizing the CUDA API.
#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <cuda.h>
#include <helper_math.h>
#include <iomanip>
#include <time.h>

#include "vector_arithmetic.h"  // for arbitrary prec. vectors
//#include <vector_functions.h> // for single prec. vectors
#include "thrust/device_ptr.h"
#include "thrust/sort.h"

#include "sphere.h"
#include "datatypes.h"
#include "utility.h"
#include "constants.cuh"
#include "debug.h"
#include "version.h"

#include "sorting.cuh"
#include "contactmodels.cuh"
#include "cohesion.cuh"
#include "contactsearch.cuh"
#include "integration.cuh"
#include "raytracer.cuh"
#include "navierstokes.cuh"
#include "darcy.cuh"

// Returns the number of cores per streaming multiprocessor, which is
// a function of the device compute capability
int cudaCoresPerSM(int major, int minor)
{
    if (major == 1)
        return 8;
    else if (major == 2 && minor == 0)
        return 32;
    else if (major == 2 && minor == 1)
        return 48;
    else if (major == 3)
        return 192;
    else if (major == 5)
        return 128;
    else if (major == 6 && minor == 0)
        return 64;
    else if (major == 6 && minor == 1)
        return 128;
    else
        printf("Error in cudaCoresPerSM Device compute capability value "
                "(%d.%d) not recognized.", major, minor);
    return -1;
}

// Wrapper function for initializing the CUDA components.
// Called from main.cpp
void DEM::initializeGPU(void)
{
    using std::cout; // stdout

    // Specify target device
    int cudadevice = 0;

    // Variables containing device properties
    cudaDeviceProp prop;
    int deviceCount;
    int cudaDriverVersion;
    int cudaRuntimeVersion;

    checkForCudaErrors("Before initializing CUDA device");

    // Register number of devices
    cudaGetDeviceCount(&deviceCount);
    ndevices = deviceCount; // store in DEM class

    if (deviceCount == 0) {
        std::cerr << "\nERROR: No CUDA-enabled devices availible. Bye."
            << std::endl;
        exit(EXIT_FAILURE);
    } else if (deviceCount == 1) {
        if (verbose == 1)
            cout << "  System contains 1 CUDA compatible device.\n";
    } else {
        if (verbose == 1)
            cout << "  System contains " << deviceCount
                << " CUDA compatible devices.\n";
    }

    // Loop through GPU's and choose the one with the most CUDA cores
    if (device == -1) {
        int ncudacores;
        int max_ncudacores = 0;
        for (int d=0; d<ndevices; d++) {
            cudaGetDeviceProperties(&prop, d);
            cudaDriverGetVersion(&cudaDriverVersion);
            cudaRuntimeGetVersion(&cudaRuntimeVersion);

            ncudacores = prop.multiProcessorCount
                *cudaCoresPerSM(prop.major, prop.minor);
            if (ncudacores > max_ncudacores) {
                max_ncudacores = ncudacores;
                cudadevice = d;
            }

            if (verbose == 1) {
                cout << "  CUDA device ID: " << d << "\n";
                cout << "  - Name: " <<  prop.name << ", compute capability: " 
                     << prop.major << "." << prop.minor << ".\n";
                cout << "  - CUDA Driver version: " << cudaDriverVersion/1000 
                     << "." <<  cudaDriverVersion%100 
                     << ", runtime version " << cudaRuntimeVersion/1000 << "." 
                     << cudaRuntimeVersion%100 << std::endl;
            }
        }

        device = cudadevice; // store in DEM class
        if (verbose == 1) {
            cout << "  Using CUDA device ID " << device << " with "
                 << max_ncudacores << " cores." << std::endl;
        }

    } else {

        cudaGetDeviceProperties(&prop, device);
        cudaDriverGetVersion(&cudaDriverVersion);
        cudaRuntimeGetVersion(&cudaRuntimeVersion);

        int ncudacores = prop.multiProcessorCount
            *cudaCoresPerSM(prop.major, prop.minor);

        if (verbose == 1) {
            cout << "  CUDA device ID: " << device << "\n";
            cout << "  - Name: " <<  prop.name << ", compute capability: " 
                 << prop.major << "." << prop.minor << ".\n";
            cout << "  - CUDA Driver version: " << cudaDriverVersion/1000 
                 << "." <<  cudaDriverVersion%100 
                 << ", runtime version " << cudaRuntimeVersion/1000 << "." 
                 << cudaRuntimeVersion%100
                 << "\n  - " << ncudacores << " CUDA cores" << std::endl;
        }
    }

    // The value of device is now 0 or larger
    cudaSetDevice(device);

    checkForCudaErrors("While initializing CUDA device");
}

// Start timer for kernel profiling
void startTimer(cudaEvent_t* kernel_tic)
{
    cudaEventRecord(*kernel_tic);
}

// Stop timer for kernel profiling and time to function sum
void stopTimer(cudaEvent_t *kernel_tic,
        cudaEvent_t *kernel_toc,
        float *kernel_elapsed,
        double* sum)
{
    cudaEventRecord(*kernel_toc, 0);
    cudaEventSynchronize(*kernel_toc);
    cudaEventElapsedTime(kernel_elapsed, *kernel_tic, *kernel_toc);
    *sum += *kernel_elapsed;
}

// Check values of parameters in constant memory
__global__ void checkConstantValues(int* dev_equal,
        Grid* dev_grid,
        Params* dev_params)
{
    // Values ok (0)
    *dev_equal = 0;

    // Compare values between global- and constant
    // memory structures
    if (dev_grid->origo[0] != devC_grid.origo[0])
        *dev_equal = 1;
    if (dev_grid->origo[1] != devC_grid.origo[1])
        *dev_equal = 2; // Not ok
    if (dev_grid->origo[2] != devC_grid.origo[2])
        *dev_equal = 3; // Not ok
    if (dev_grid->L[0] != devC_grid.L[0])
        *dev_equal = 4; // Not ok
    if (dev_grid->L[1] != devC_grid.L[1])
        *dev_equal = 5; // Not ok
    if (dev_grid->L[2] != devC_grid.L[2])
        *dev_equal = 6; // Not ok
    if (dev_grid->num[0] != devC_grid.num[0])
        *dev_equal = 7; // Not ok
    if (dev_grid->num[1] != devC_grid.num[1])
        *dev_equal = 8; // Not ok
    if (dev_grid->num[2] != devC_grid.num[2])
        *dev_equal = 9; // Not ok
    if (dev_grid->periodic != devC_grid.periodic)
        *dev_equal = 10; // Not ok

    if (dev_params->g[0] != devC_params.g[0])
        *dev_equal = 11; // Not ok
    if (dev_params->g[1] != devC_params.g[1])
        *dev_equal = 12; // Not ok
    if (dev_params->g[2] != devC_params.g[2])
        *dev_equal = 13; // Not ok
    if (dev_params->k_n != devC_params.k_n)
        *dev_equal = 14; // Not ok
    if (dev_params->k_t != devC_params.k_t)
        *dev_equal = 15; // Not ok
    if (dev_params->k_r != devC_params.k_r)
        *dev_equal = 16; // Not ok
    if (dev_params->gamma_n != devC_params.gamma_n)
        *dev_equal = 17; // Not ok
    if (dev_params->gamma_t != devC_params.gamma_t)
        *dev_equal = 18; // Not ok
    if (dev_params->gamma_r != devC_params.gamma_r)
        *dev_equal = 19; // Not ok
    if (dev_params->mu_s != devC_params.mu_s)
        *dev_equal = 20; // Not ok
    if (dev_params->mu_d != devC_params.mu_d)
        *dev_equal = 21; // Not ok
    if (dev_params->mu_r != devC_params.mu_r)
        *dev_equal = 22; // Not ok
    if (dev_params->rho != devC_params.rho)
        *dev_equal = 23; // Not ok
    if (dev_params->contactmodel != devC_params.contactmodel)
        *dev_equal = 24; // Not ok
    if (dev_params->kappa != devC_params.kappa)
        *dev_equal = 25; // Not ok
    if (dev_params->db != devC_params.db)
        *dev_equal = 26; // Not ok
    if (dev_params->V_b != devC_params.V_b)
        *dev_equal = 27; // Not ok
    if (dev_params->lambda_bar != devC_params.lambda_bar)
        *dev_equal = 28; // Not ok
    if (dev_params->nb0 != devC_params.nb0)
        *dev_equal = 29; // Not ok
    if (dev_params->E != devC_params.E)
        *dev_equal = 30; // Not ok
}

__global__ void checkParticlePositions(
    const Float4* __restrict__ dev_x)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; // Thread id

    if (idx < devC_np) { // Condition prevents block size error
        Float4 x = dev_x[idx];

        // make sure grain doesn't have NaN or Inf position
        if (!isfinite(x.x) || !isfinite(x.y) || !isfinite(x.z)) {
            __syncthreads();
            printf("\nParticle %d has non-finite position: x = %f %f %f",
                    idx, x.x, x.y, x.z);
       }

        /*__syncthreads();
        printf("\nParticle %d: x = %f %f %f",
                idx, x.x, x.y, x.z);*/

        // check that the particle is inside of the simulation domain
        if (x.x < devC_grid.origo[0] ||
                x.y < devC_grid.origo[1] ||
                x.z < devC_grid.origo[2] ||
                x.x > devC_grid.L[0] ||
                x.y > devC_grid.L[1] ||
                x.z > devC_grid.L[2]) {
            __syncthreads();
            printf("\nParticle %d is outside the computational domain "
                    "(%f %f %f to %f %f %f): x = %f %f %f",
                    idx,
                    devC_grid.origo[0], devC_grid.origo[1], devC_grid.origo[2],
                    devC_grid.L[0], devC_grid.L[1], devC_grid.L[2],
                    x.x, x.y, x.z);
        }
    }
}


// Copy the constant data components to device memory,
// and check whether the values correspond to the 
// values in constant memory.
void DEM::checkConstantMemory()
{
    // Allocate space in global device memory
    Grid* dev_grid;
    Params* dev_params;
    cudaMalloc((void**)&dev_grid, sizeof(Grid));
    cudaMalloc((void**)&dev_params, sizeof(Params));

    // Copy structure data from host to global device memory
    cudaMemcpy(dev_grid, &grid, sizeof(Grid), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_params, &params, sizeof(Params), cudaMemcpyHostToDevice);

    // Compare values between global and constant memory
    // structures on the device.
    int* equal = new int;  // The values are equal = 0, if not = 1
    *equal = 0;
    int* dev_equal;
    cudaMalloc((void**)&dev_equal, sizeof(int));
    checkConstantValues<<<1,1>>>(dev_equal, dev_grid, dev_params);
    checkForCudaErrors("After constant memory check");

    // Copy result to host
    cudaMemcpy(equal, dev_equal, sizeof(int), cudaMemcpyDeviceToHost);

    // Free global device memory
    cudaFree(dev_grid);
    cudaFree(dev_params);
    cudaFree(dev_equal);

    // Are the values equal?
    if (*equal != 0) {
        std::cerr << "Error! The values in constant memory do not "
            << "seem to be correct (" << *equal << ")." << std::endl;
        exit(1);
    } else {
        if (verbose == 1)
            std::cout << "  Constant values ok (" << *equal << ")."
                << std::endl;
    }
}

// Copy selected constant components to constant device memory.
void DEM::transferToConstantDeviceMemory()
{
    using std::cout;

    if (verbose == 1)
        cout << "  Transfering data to constant device memory:     ";

    /*for (int d=0; d<ndevices; d++) {
      cudaSetDevice(d);*/
        cudaMemcpyToSymbol(devC_nd, &nd, sizeof(nd));
        cudaMemcpyToSymbol(devC_np, &np, sizeof(np));
        cudaMemcpyToSymbol(devC_nw, &walls.nw, sizeof(unsigned int));
        cudaMemcpyToSymbol(devC_nc, &NC, sizeof(int));
        cudaMemcpyToSymbol(devC_dt, &time.dt, sizeof(Float));
        cudaMemcpyToSymbol(devC_grid, &grid, sizeof(Grid));
        cudaMemcpyToSymbol(devC_params, &params, sizeof(Params));
        /*}
          cudaSetDevice(device);*/

    checkForCudaErrors("After transferring to device constant memory");

    if (verbose == 1)
        cout << "Done\n";

    // only for device with most CUDA cores
    checkConstantMemory();
}

__global__ void printWorldSize(Float4* dev_walls_nx)
{
    printf("\nL = %f, %f, %f\n",
            devC_grid.L[0], devC_grid.L[1], devC_grid.L[2]);
    printf("\ndev_walls_nx[0] = %f, %f, %f, %f\n",
            dev_walls_nx[0].x,
            dev_walls_nx[0].y,
            dev_walls_nx[0].z,
            dev_walls_nx[0].w);
}

void DEM::updateGridSize()
{
    //printf("\nDEM::updateGridSize() start\n");
    Float* Lz = new Float;

    // Get top wall position from dev_walls_nx[0].z
    cudaMemcpy(Lz, &dev_walls_nx[0].w, sizeof(Float), cudaMemcpyDeviceToHost);
    checkForCudaErrors("DEM::updateGridSize(): copying wall position");

    //printWorldSize<<<1,1>>>(dev_walls_nx);
    //cudaThreadSynchronize();
    //checkForCudaErrors("DEM::updateGridSize(): first printWorldSize");

    //printf("\nLz = %f\n", *Lz);

    // Write value to grid.L[2]
    grid.L[2] = *Lz;

    // Write value to devC_grid.L[2]
    //cudaMemcpyToSymbol(devC_grid.L[2], &Lz, sizeof(Float)); 
    cudaMemcpyToSymbol(devC_grid, &grid, sizeof(Grid));

    checkForCudaErrors("DEM::updateGridSize(): write to devC_grid.L[2]");

    //printWorldSize<<<1,1>>>(dev_walls_nx);
    //cudaThreadSynchronize();
    //checkForCudaErrors("DEM::updateGridSize(): second printWorldSize");

    // check value only during debugging
    //checkConstantMemory();
}


// Allocate device memory for particle variables,
// tied to previously declared pointers in structures
void DEM::allocateGlobalDeviceMemory(void)
{
    // Particle memory size
    unsigned int memSizeF  = sizeof(Float) * np;
    unsigned int memSizeF4 = sizeof(Float4) * np;

    if (verbose == 1)
        std::cout << "  Allocating global device memory:                ";

    k.acc = new Float4[np];
    k.angacc = new Float4[np];
#pragma omp parallel for if(np>100)
    for (unsigned int i = 0; i<np; ++i) {
        k.acc[i] = MAKE_FLOAT4(0.0, 0.0, 0.0, 0.0);
        k.angacc[i] = MAKE_FLOAT4(0.0, 0.0, 0.0, 0.0);
    }

    // Kinematics arrays
    cudaMalloc((void**)&dev_x, memSizeF4);
    cudaMalloc((void**)&dev_xyzsum, memSizeF4);
    cudaMalloc((void**)&dev_vel, memSizeF4);
    cudaMalloc((void**)&dev_vel0, memSizeF4);
    cudaMalloc((void**)&dev_acc, memSizeF4);
    cudaMalloc((void**)&dev_force, memSizeF4);
    cudaMalloc((void**)&dev_angpos, memSizeF4);
    cudaMalloc((void**)&dev_angvel, memSizeF4);
    cudaMalloc((void**)&dev_angvel0, memSizeF4);
    cudaMalloc((void**)&dev_angacc, memSizeF4);
    cudaMalloc((void**)&dev_torque, memSizeF4);

    // Particle contact bookkeeping arrays
    cudaMalloc((void**)&dev_contacts,
               sizeof(unsigned int)*np*NC);
    cudaMalloc((void**)&dev_distmod, memSizeF4*NC);
    cudaMalloc((void**)&dev_delta_t, memSizeF4*NC);
    cudaMalloc((void**)&dev_bonds, sizeof(uint2)*params.nb0);
    cudaMalloc((void**)&dev_bonds_delta, sizeof(Float4)*params.nb0);
    cudaMalloc((void**)&dev_bonds_omega, sizeof(Float4)*params.nb0);

    // Sorted arrays
    cudaMalloc((void**)&dev_x_sorted, memSizeF4);
    cudaMalloc((void**)&dev_vel_sorted, memSizeF4);
    cudaMalloc((void**)&dev_angvel_sorted, memSizeF4);

    // Energy arrays
    cudaMalloc((void**)&dev_es_dot, memSizeF);
    cudaMalloc((void**)&dev_ev_dot, memSizeF);
    cudaMalloc((void**)&dev_es, memSizeF);
    cudaMalloc((void**)&dev_ev, memSizeF);
    cudaMalloc((void**)&dev_p, memSizeF);

    // Cell-related arrays
    cudaMalloc((void**)&dev_gridParticleCellID, sizeof(unsigned int)*np);
    cudaMalloc((void**)&dev_gridParticleIndex, sizeof(unsigned int)*np);
    cudaMalloc((void**)&dev_cellStart, sizeof(unsigned int)
               *grid.num[0]*grid.num[1]*grid.num[2]);
    cudaMalloc((void**)&dev_cellEnd, sizeof(unsigned int)
               *grid.num[0]*grid.num[1]*grid.num[2]);

    // Host contact bookkeeping arrays
    k.contacts = new unsigned int[np*NC];
    // Initialize contacts lists to np
#pragma omp parallel for if(np>100)
    for (unsigned int i=0; i<(np*NC); ++i)
        k.contacts[i] = np;
    k.distmod = new Float4[np*NC];
    k.delta_t = new Float4[np*NC];

    // Wall arrays
    cudaMalloc((void**)&dev_walls_wmode, sizeof(int)*walls.nw);
    cudaMalloc((void**)&dev_walls_nx, sizeof(Float4)*walls.nw);
    cudaMalloc((void**)&dev_walls_mvfd, sizeof(Float4)*walls.nw);
    cudaMalloc((void**)&dev_walls_tau_x, sizeof(Float)*walls.nw);
    cudaMalloc((void**)&dev_walls_tau_eff_x_pp, sizeof(Float)*walls.nw*np);
    cudaMalloc((void**)&dev_walls_force_pp, sizeof(Float)*walls.nw*np);
    cudaMalloc((void**)&dev_walls_acc, sizeof(Float)*walls.nw);
    // dev_walls_force_partial allocated later
    // dev_walls_tau_eff_x_partial allocated later

    checkForCudaErrors("End of allocateGlobalDeviceMemory");
    if (verbose == 1)
        std::cout << "Done" << std::endl;
}

// Allocate global memory on other devices required for "interact" function.
// The values of domain_size[ndevices] must be set beforehand.
void DEM::allocateHelperDeviceMemory(void)
{
    // Particle memory size
    unsigned int memSizeF4 = sizeof(Float4) * np;

    // Initialize pointers to per-GPU arrays
    hdev_gridParticleIndex = (unsigned**)malloc(ndevices*sizeof(unsigned*));
    hdev_gridCellStart     = (unsigned**)malloc(ndevices*sizeof(unsigned*));
    hdev_gridCellEnd       = (unsigned**)malloc(ndevices*sizeof(unsigned*));
    hdev_x                 = (Float4**)malloc(ndevices*sizeof(Float4*));
    hdev_x_sorted          = (Float4**)malloc(ndevices*sizeof(Float4*));
    hdev_vel               = (Float4**)malloc(ndevices*sizeof(Float4*));
    hdev_vel_sorted        = (Float4**)malloc(ndevices*sizeof(Float4*));
    hdev_angvel            = (Float4**)malloc(ndevices*sizeof(Float4*));
    hdev_angvel_sorted     = (Float4**)malloc(ndevices*sizeof(Float4*));
    hdev_walls_nx          = (Float4**)malloc(ndevices*sizeof(Float4*));
    hdev_walls_mvfd        = (Float4**)malloc(ndevices*sizeof(Float4*));
    hdev_distmod           = (Float4**)malloc(ndevices*sizeof(Float4*));

    hdev_force             = (Float4**)malloc(ndevices*sizeof(Float4*));
    hdev_torque            = (Float4**)malloc(ndevices*sizeof(Float4*));
    hdev_delta_t           = (Float4**)malloc(ndevices*sizeof(Float4*));
    hdev_es_dot            = (Float**)malloc(ndevices*sizeof(Float*));
    hdev_es                = (Float**)malloc(ndevices*sizeof(Float*));
    hdev_ev_dot            = (Float**)malloc(ndevices*sizeof(Float*));
    hdev_ev                = (Float**)malloc(ndevices*sizeof(Float*));
    hdev_p                 = (Float**)malloc(ndevices*sizeof(Float*));
    hdev_walls_force_pp    = (Float**)malloc(ndevices*sizeof(Float*));
    hdev_contacts          = (unsigned**)malloc(ndevices*sizeof(unsigned*));

    for (int d=0; d<ndevices; d++) {

        // do not allocate memory on primary GPU
        if (d == device)
            continue;

        cudaSetDevice(d);

        // allocate space for full input arrays for interact()
        cudaMalloc((void**)&hdev_gridParticleIndex[d], sizeof(unsigned int)*np);
        cudaMalloc((void**)&hdev_gridCellStart[d], sizeof(unsigned int)
                   *grid.num[0]*grid.num[1]*grid.num[2]);
        cudaMalloc((void**)&hdev_gridCellEnd[d], sizeof(unsigned int)
                   *grid.num[0]*grid.num[1]*grid.num[2]);
        cudaMalloc((void**)&hdev_x[d], memSizeF4);
        cudaMalloc((void**)&hdev_x_sorted[d], memSizeF4);
        cudaMalloc((void**)&hdev_vel[d], memSizeF4);
        cudaMalloc((void**)&hdev_vel_sorted[d], memSizeF4);
        cudaMalloc((void**)&hdev_angvel[d], memSizeF4);
        cudaMalloc((void**)&hdev_angvel_sorted[d], memSizeF4);
        cudaMalloc((void**)&hdev_walls_nx[d], sizeof(Float4)*walls.nw);
        cudaMalloc((void**)&hdev_walls_mvfd[d], sizeof(Float4)*walls.nw);
        cudaMalloc((void**)&hdev_distmod[d], memSizeF4*NC);

        // allocate space for partial output arrays for interact()
        cudaMalloc((void**)&hdev_force[d], sizeof(Float4)*domain_size[d]);
        cudaMalloc((void**)&hdev_torque[d], sizeof(Float4)*domain_size[d]);
        cudaMalloc((void**)&hdev_es_dot[d], sizeof(Float)*domain_size[d]);
        cudaMalloc((void**)&hdev_ev_dot[d], sizeof(Float)*domain_size[d]);
        cudaMalloc((void**)&hdev_es[d], sizeof(Float)*domain_size[d]);
        cudaMalloc((void**)&hdev_ev[d], sizeof(Float)*domain_size[d]);
        cudaMalloc((void**)&hdev_p[d], sizeof(Float)*domain_size[d]);
        cudaMalloc((void**)&hdev_walls_force_pp[d],
                   sizeof(Float)*domain_size[d]*walls.nw);
        cudaMalloc((void**)&hdev_contacts[d],
                   sizeof(unsigned)*domain_size[d]*NC);
        cudaMalloc((void**)&hdev_delta_t[d], sizeof(Float4)*domain_size[d]*NC);

        checkForCudaErrors("During allocateGlobalDeviceMemoryOtherDevices");
    }
    cudaSetDevice(device); // select main GPU
}

void DEM::freeHelperDeviceMemory()
{
    for (int d=0; d<ndevices; d++) {

        // do not allocate memory on primary GPU
        if (d == device)
            continue;

        cudaSetDevice(d);

        cudaFree(hdev_gridParticleIndex[d]);
        cudaFree(hdev_gridCellStart[d]);
        cudaFree(hdev_gridCellEnd[d]);
        cudaFree(hdev_x[d]);
        cudaFree(hdev_vel[d]);
        cudaFree(hdev_vel_sorted[d]);
        cudaFree(hdev_angvel[d]);
        cudaFree(hdev_angvel_sorted[d]);
        cudaFree(hdev_walls_nx[d]);
        cudaFree(hdev_walls_mvfd[d]);
        cudaFree(hdev_distmod[d]);

        cudaFree(hdev_force[d]);
        cudaFree(hdev_torque[d]);
        cudaFree(hdev_es_dot[d]);
        cudaFree(hdev_ev_dot[d]);
        cudaFree(hdev_es[d]);
        cudaFree(hdev_ev[d]);
        cudaFree(hdev_p[d]);
        cudaFree(hdev_walls_force_pp[d]);
        cudaFree(hdev_contacts[d]);
        cudaFree(hdev_delta_t[d]);

        checkForCudaErrors("During helper device cudaFree calls");
    }
    cudaSetDevice(device); // select primary GPU
}

void DEM::freeGlobalDeviceMemory()
{
    if (verbose == 1)
        printf("\nFreeing device memory:                           ");

    // Particle arrays
    cudaFree(dev_x);
    cudaFree(dev_xyzsum);
    cudaFree(dev_vel);
    cudaFree(dev_vel0);
    cudaFree(dev_acc);
    cudaFree(dev_force);
    cudaFree(dev_angpos);
    cudaFree(dev_angvel);
    cudaFree(dev_angvel0);
    cudaFree(dev_angacc);
    cudaFree(dev_torque);

    cudaFree(dev_contacts);
    cudaFree(dev_distmod);
    cudaFree(dev_delta_t);
    cudaFree(dev_bonds);
    cudaFree(dev_bonds_delta);
    cudaFree(dev_bonds_omega);

    cudaFree(dev_es_dot);
    cudaFree(dev_es);
    cudaFree(dev_ev_dot);
    cudaFree(dev_ev);
    cudaFree(dev_p);

    cudaFree(dev_x_sorted);
    cudaFree(dev_vel_sorted);
    cudaFree(dev_angvel_sorted);

    // Cell-related arrays
    cudaFree(dev_gridParticleIndex);
    cudaFree(dev_cellStart);
    cudaFree(dev_cellEnd);

    // Wall arrays
    cudaFree(dev_walls_nx);
    cudaFree(dev_walls_mvfd);
    cudaFree(dev_walls_tau_x);
    cudaFree(dev_walls_force_partial);
    cudaFree(dev_walls_force_pp);
    cudaFree(dev_walls_acc);
    cudaFree(dev_walls_tau_eff_x_pp);
    cudaFree(dev_walls_tau_eff_x_partial);

    // Fluid arrays
    if (fluid == 1 && cfd_solver == 0) {
        freeNSmemDev();
    }
    if (fluid == 1 && cfd_solver == 1) {
        freeDarcyMemDev();
    }

    //checkForCudaErrors("During cudaFree calls");

    if (verbose == 1)
        std::cout << "Done" << std::endl;
}


void DEM::transferToGlobalDeviceMemory(int statusmsg)
{
    if (verbose == 1 && statusmsg == 1)
        std::cout << "  Transfering data to the device:                 ";

    // Commonly-used memory sizes
    unsigned int memSizeF  = sizeof(Float) * np;
    unsigned int memSizeF4 = sizeof(Float4) * np;

    // Copy static-size structure data from host to global device memory
    //cudaMemcpy(dev_time, &time, sizeof(Time), cudaMemcpyHostToDevice);

    // Kinematic particle values
    cudaMemcpy( dev_x,        k.x,
                memSizeF4, cudaMemcpyHostToDevice);
    cudaMemcpy( dev_xyzsum,   k.xyzsum,
                memSizeF4, cudaMemcpyHostToDevice);
    cudaMemcpy( dev_vel,      k.vel,
                memSizeF4, cudaMemcpyHostToDevice);
    cudaMemcpy( dev_vel0,     k.vel,
                memSizeF4, cudaMemcpyHostToDevice);
    cudaMemcpy( dev_acc,      k.acc,
                memSizeF4, cudaMemcpyHostToDevice);
    cudaMemcpy( dev_force,    k.force,
                memSizeF4, cudaMemcpyHostToDevice);
    cudaMemcpy( dev_angpos,   k.angpos,
                memSizeF4, cudaMemcpyHostToDevice);
    cudaMemcpy( dev_angvel,   k.angvel,
                memSizeF4, cudaMemcpyHostToDevice);
    cudaMemcpy( dev_angvel0,  k.angvel,
                memSizeF4, cudaMemcpyHostToDevice);
    cudaMemcpy( dev_angacc,   k.angacc,
                memSizeF4, cudaMemcpyHostToDevice);
    cudaMemcpy( dev_torque,   k.torque,
                memSizeF4, cudaMemcpyHostToDevice);
    cudaMemcpy( dev_contacts, k.contacts,
                sizeof(unsigned int)*np*NC, cudaMemcpyHostToDevice);
    cudaMemcpy( dev_distmod, k.distmod,
                memSizeF4*NC, cudaMemcpyHostToDevice);
    cudaMemcpy( dev_delta_t, k.delta_t,
                memSizeF4*NC, cudaMemcpyHostToDevice);
    cudaMemcpy( dev_bonds, k.bonds,
                sizeof(uint2)*params.nb0, cudaMemcpyHostToDevice);
    cudaMemcpy( dev_bonds_delta, k.bonds_delta,
                sizeof(Float4)*params.nb0, cudaMemcpyHostToDevice);
    cudaMemcpy( dev_bonds_omega, k.bonds_omega,
                sizeof(Float4)*params.nb0, cudaMemcpyHostToDevice);

    // Individual particle energy values
    cudaMemcpy( dev_es_dot, e.es_dot,
                memSizeF, cudaMemcpyHostToDevice);
    cudaMemcpy( dev_es,     e.es,
                memSizeF, cudaMemcpyHostToDevice);
    cudaMemcpy( dev_ev_dot, e.ev_dot,
                memSizeF, cudaMemcpyHostToDevice);
    cudaMemcpy( dev_ev,     e.ev,
                memSizeF, cudaMemcpyHostToDevice);
    cudaMemcpy( dev_p, e.p,
                memSizeF, cudaMemcpyHostToDevice);

    // Wall parameters
    cudaMemcpy( dev_walls_wmode, walls.wmode,
                sizeof(int)*walls.nw, cudaMemcpyHostToDevice);
    cudaMemcpy( dev_walls_nx,    walls.nx,
                sizeof(Float4)*walls.nw, cudaMemcpyHostToDevice);
    cudaMemcpy( dev_walls_mvfd,  walls.mvfd,
                sizeof(Float4)*walls.nw, cudaMemcpyHostToDevice);
    cudaMemcpy( dev_walls_tau_x,  walls.tau_x,
                sizeof(Float)*walls.nw, cudaMemcpyHostToDevice);

    // Fluid arrays
    if (fluid == 1) {
        if (cfd_solver == 0) {
            transferNStoGlobalDeviceMemory(1);
        } else if (cfd_solver == 1) {
            transferDarcyToGlobalDeviceMemory(1);
        } else {
            std::cerr << "Error: cfd_solver value not understood ("
                << cfd_solver << ")" << std::endl;
        }
    }

    checkForCudaErrors("End of transferToGlobalDeviceMemory");
    if (verbose == 1 && statusmsg == 1)
        std::cout << "Done" << std::endl;
}

void DEM::transferFromGlobalDeviceMemory()
{
    //std::cout << "  Transfering data from the device:               ";

    // Commonly-used memory sizes
    unsigned int memSizeF  = sizeof(Float) * np;
    unsigned int memSizeF4 = sizeof(Float4) * np;

    // Copy static-size structure data from host to global device memory
    //cudaMemcpy(&time, dev_time, sizeof(Time), cudaMemcpyDeviceToHost);

    // Kinematic particle values
    cudaMemcpy( k.x, dev_x,
            memSizeF4, cudaMemcpyDeviceToHost);
    cudaMemcpy( k.xyzsum, dev_xyzsum,
            memSizeF4, cudaMemcpyDeviceToHost);
    cudaMemcpy( k.vel, dev_vel,
            memSizeF4, cudaMemcpyDeviceToHost);
    cudaMemcpy( k.acc, dev_acc,
            memSizeF4, cudaMemcpyDeviceToHost);
    cudaMemcpy( k.force, dev_force,
            memSizeF4, cudaMemcpyDeviceToHost);
    cudaMemcpy( k.angpos, dev_angpos,
            memSizeF4, cudaMemcpyDeviceToHost);
    cudaMemcpy( k.angvel, dev_angvel,
            memSizeF4, cudaMemcpyDeviceToHost);
    cudaMemcpy( k.angacc, dev_angacc,
            memSizeF4, cudaMemcpyDeviceToHost);
    cudaMemcpy( k.torque, dev_torque,
            memSizeF4, cudaMemcpyDeviceToHost);
    cudaMemcpy( k.contacts, dev_contacts,
            sizeof(unsigned int)*np*NC, cudaMemcpyDeviceToHost);
    cudaMemcpy( k.distmod, dev_distmod,
            memSizeF4*NC, cudaMemcpyDeviceToHost);
    cudaMemcpy( k.delta_t, dev_delta_t,
            memSizeF4*NC, cudaMemcpyDeviceToHost);
    cudaMemcpy( k.bonds, dev_bonds,
            sizeof(uint2)*params.nb0, cudaMemcpyDeviceToHost);
    cudaMemcpy( k.bonds_delta, dev_bonds_delta,
            sizeof(Float4)*params.nb0, cudaMemcpyDeviceToHost);
    cudaMemcpy( k.bonds_omega, dev_bonds_omega,
            sizeof(Float4)*params.nb0, cudaMemcpyDeviceToHost);

    // Individual particle energy values
    cudaMemcpy( e.es_dot, dev_es_dot,
            memSizeF, cudaMemcpyDeviceToHost);
    cudaMemcpy( e.es, dev_es,
            memSizeF, cudaMemcpyDeviceToHost);
    cudaMemcpy( e.ev_dot, dev_ev_dot,
            memSizeF, cudaMemcpyDeviceToHost);
    cudaMemcpy( e.ev, dev_ev,
            memSizeF, cudaMemcpyDeviceToHost);
    cudaMemcpy( e.p, dev_p,
            memSizeF, cudaMemcpyDeviceToHost);

    // Wall parameters
    cudaMemcpy( walls.wmode, dev_walls_wmode,
            sizeof(int)*walls.nw, cudaMemcpyDeviceToHost);
    cudaMemcpy( walls.nx, dev_walls_nx,
            sizeof(Float4)*walls.nw, cudaMemcpyDeviceToHost);
    cudaMemcpy( walls.mvfd, dev_walls_mvfd,
            sizeof(Float4)*walls.nw, cudaMemcpyDeviceToHost);
    cudaMemcpy( walls.tau_x, dev_walls_tau_x,
            sizeof(Float)*walls.nw, cudaMemcpyDeviceToHost);

    // Fluid arrays
    if (fluid == 1 && cfd_solver == 0) {
        transferNSfromGlobalDeviceMemory(0);
    }
    else if (fluid == 1 && cfd_solver == 1) {
        transferDarcyFromGlobalDeviceMemory(0);
        checkDarcyStability();
    }

    //checkForCudaErrors("End of transferFromGlobalDeviceMemory");
}


// Iterate through time by explicit time integration
void DEM::startTime()
{
    using std::cout;
    using std::cerr;
    using std::endl;

    std::string outfile;
    char file[200];
    FILE *fp;

    // Synchronization point
    cudaThreadSynchronize();
    checkForCudaErrors("Start of startTime()");

    // Write initial data to output/<sid>.output00000.bin
    if (time.step_count == 0)
        writebin(("output/" + sid + ".output00000.bin").c_str());

    // Time variables
    clock_t tic, toc;
    double filetimeclock, time_spent;
    float dev_time_spent;

    // Start CPU clock
    tic = clock();

    //// GPU workload configuration
    unsigned int threadsPerBlock = 256; 
    //unsigned int threadsPerBlock = 512; 

    // Create enough blocks to accomodate the particles
    unsigned int blocksPerGrid = iDivUp(np, threadsPerBlock); 
    dim3 dimGrid(blocksPerGrid, 1, 1); // Blocks arranged in 1D grid
    dim3 dimBlock(threadsPerBlock, 1, 1); // Threads arranged in 1D block

    unsigned int blocksPerGridBonds = iDivUp(params.nb0, threadsPerBlock); 
    dim3 dimGridBonds(blocksPerGridBonds, 1, 1); // Blocks arranged in 1D grid

    // Use 3D block and grid layout for cell-centered fluid calculations
    dim3 dimBlockFluid(8, 8, 8);    // 512 threads per block
    dim3 dimGridFluid(
            iDivUp(grid.num[0], dimBlockFluid.x),
            iDivUp(grid.num[1], dimBlockFluid.y),
            iDivUp(grid.num[2], dimBlockFluid.z));
    if (dimGridFluid.z > 64 && fluid == 1) {
        cerr << "Error: dimGridFluid.z > 64" << endl;
        exit(1);
    }

    // Use 3D block and grid layout for cell-face fluid calculations
    dim3 dimBlockFluidFace(8, 8, 8);    // 512 threads per block
    dim3 dimGridFluidFace(
            iDivUp(grid.num[0]+1, dimBlockFluidFace.x),
            iDivUp(grid.num[1]+1, dimBlockFluidFace.y),
            iDivUp(grid.num[2]+1, dimBlockFluidFace.z));
    if (dimGridFluidFace.z > 64 && fluid == 1) {
        cerr << "Error: dimGridFluidFace.z > 64" << endl;
        exit(1);
    }


    // Shared memory per block
    unsigned int smemSize = sizeof(unsigned int)*(threadsPerBlock+1);

    // Pre-sum of force per wall
    cudaMalloc((void**)&dev_walls_force_partial,
            sizeof(Float)*dimGrid.x*walls.nw);

    // Pre-sum of shear stress per wall
    cudaMalloc((void**)&dev_walls_tau_eff_x_partial,
            sizeof(Float)*dimGrid.x*walls.nw);

    // Report to stdout
    if (verbose == 1) {
        cout << "\n  Device memory allocation and transfer complete.\n"
            << "  - Blocks per grid: "
            << dimGrid.x << "*" << dimGrid.y << "*" << dimGrid.z << "\n"
            << "  - Threads per block: "
            << dimBlock.x << "*" << dimBlock.y << "*" << dimBlock.z << "\n"
            << "  - Shared memory required per block: " << smemSize << " bytes"
            << endl;
        if (fluid == 1) {
            cout << "  - Blocks per fluid grid: "
                << dimGridFluid.x << "*" << dimGridFluid.y << "*" <<
                dimGridFluid.z << "\n"
                << "  - Threads per fluid block: "
                << dimBlockFluid.x << "*" << dimBlockFluid.y << "*" <<
                dimBlockFluid.z << endl;
        }
    }

    // Initialize counter variable values
    filetimeclock = 0.0;
    long iter = 0;
    const int stdout_report = 10; // no of steps between reporting to stdout

    // Create first status.dat
    //sprintf(file,"output/%s.status.dat", sid);
    outfile = "output/" + sid + ".status.dat";
    fp = fopen(outfile.c_str(), "w");
    fprintf(fp,"%2.4e %2.4e %d\n", 
            time.current, 
            100.0*time.current/time.total, 
            time.step_count);
    fclose(fp);

    if (verbose == 1) {
        cout << "\n  Entering the main calculation time loop...\n\n"
            << "  IMPORTANT: Do not close this terminal, doing so will \n"
            << "             terminate this SPHERE process. Follow the \n"
            << "             progress by executing:\n"
            << "                $ ./sphere_status " << sid << endl << endl;
    }


    // Start GPU clock
    cudaEvent_t dev_tic, dev_toc;
    cudaEventCreate(&dev_tic);
    cudaEventCreate(&dev_toc);
    cudaEventRecord(dev_tic, 0);

    // If profiling is enabled, initialize timers for each kernel
    cudaEvent_t kernel_tic, kernel_toc;
    float kernel_elapsed;
    double t_calcParticleCellID = 0.0;
    double t_thrustsort = 0.0;
    double t_reorderArrays = 0.0;
    double t_topology = 0.0;
    double t_interact = 0.0;
    double t_bondsLinear = 0.0;
    double t_latticeBoltzmannD3Q19 = 0.0;
    double t_integrate = 0.0;
    double t_summation = 0.0;
    double t_integrateWalls = 0.0;

    double t_findPorositiesDev = 0.0;
    double t_findNSstressTensor = 0.0;
    double t_findNSdivphiviv = 0.0;
    double t_findNSdivtau = 0.0;
    double t_findPredNSvelocities = 0.0;
    double t_setNSepsilon = 0.0;
    double t_setNSdirichlet = 0.0;
    double t_setNSghostNodesDev = 0.0;
    double t_findNSforcing = 0.0;
    double t_jacobiIterationNS = 0.0;
    double t_updateNSvelocityPressure = 0.0;

    double t_findDarcyPorosities = 0.0;
    double t_setDarcyGhostNodes = 0.0;
    double t_findDarcyPressureForce = 0.0;
    double t_setDarcyTopPressure = 0.0;
    double t_findDarcyPermeabilities = 0.0;
    double t_findDarcyPermeabilityGradients = 0.0;
    //double t_findDarcyPressureChange = 0.0;
    double t_updateDarcySolution = 0.0;
    double t_copyValues = 0.0;
    double t_findDarcyVelocities = 0.0;

    
    if (PROFILING == 1) {
        cudaEventCreate(&kernel_tic);
        cudaEventCreate(&kernel_toc);
    }

    // The model start time is saved for profiling performance
    double t_start = time.current;
    double t_ratio;     // ration between time flow in model vs. reality

    // Hard-coded parameters for stepwise velocity change (rate-state exp)
    int velocity_state = 1;  // 1: v1, 2: v2
    int change_velocity_state = 0;  // 1: increase velocity, 2: decrease vel.
    const Float velocity_factor = 10.0;  // v2 = v1*velocity_factor
    const Float v2_start = 10.0; // seconds
    const Float v2_end = 15.0;  // seconds

    // Index of dynamic top wall (if it exists)
    unsigned int wall0_iz = 10000000;
    // weight of fluid between two cells in z direction
    Float dp_dz;
    if (fluid == 1) {
        if (cfd_solver == 0)
            dp_dz = fabs(ns.rho_f*params.g[2]*grid.L[2]/grid.num[2]);
        else if (cfd_solver == 1) {
            dp_dz = fabs(darcy.rho_f*params.g[2]*grid.L[2]/grid.num[2]);

            // determine pressure at top wall at t=0
            darcy.p_top_orig = darcy.p[d_idx(0,0,darcy.nz-1)]
                                - darcy.p_mod_A
                                *sin(2.0*M_PI*darcy.p_mod_f*time.current
                                        + darcy.p_mod_phi);
        }
    }
    //std::cout << "dp_dz = " << dp_dz << std::endl;

    // Write a log file of the number of iterations it took before
    // convergence in the fluid solver
    std::ofstream convlog;
    if (write_conv_log == 1) {
        std::string f = "output/" + sid + "-conv.log";
        convlog.open(f.c_str());
    }

    if (verbose == 1)
        cout << "  Current simulation time: " << time.current << " s.";

    // MAIN CALCULATION TIME LOOP
    while (time.current <= time.total) {

        // Print current step number to terminal
        //printf("\n\n@@@ DEM time step: %ld\n", iter);

        // Routine check for errors
        checkForCudaErrors("Start of main while loop");

        if (np > 0) {

            // check if particle positions have finite values
#ifdef CHECK_PARTICLES_FINITE
            checkParticlePositions<<<dimGrid, dimBlock>>>(dev_x);
            cudaThreadSynchronize();
            checkForCudaErrorsIter("Post checkParticlePositions", iter);
#endif

            // If the grid is adaptive, readjust the grid height to equal the 
            // positions of the dynamic walls
            if (grid.adaptive == 1 && walls.nw > 0) {
                updateGridSize();
            }

            // For each particle: 
            // Compute hash key (cell index) from position 
            // in the fine, uniform and homogenous grid.
            if (PROFILING == 1)
                startTimer(&kernel_tic);
            calcParticleCellID<<<dimGrid, dimBlock>>>(dev_gridParticleCellID,
                    dev_gridParticleIndex, 
                    dev_x);

            // Synchronization point
            cudaThreadSynchronize();
            if (PROFILING == 1)
                stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed,
                        &t_calcParticleCellID);
            checkForCudaErrorsIter("Post calcParticleCellID", iter);


            // Sort particle (key, particle ID) pairs by hash key with Thrust
            // radix sort
            if (PROFILING == 1)
                startTimer(&kernel_tic);
            thrust::sort_by_key(
                    thrust::device_ptr<uint>(dev_gridParticleCellID),
                    thrust::device_ptr<uint>(dev_gridParticleCellID + np),
                    thrust::device_ptr<uint>(dev_gridParticleIndex));
            cudaThreadSynchronize(); // Maybe Thrust synchronizes implicitly?
            if (PROFILING == 1)
                stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed,
                        &t_thrustsort);
            checkForCudaErrorsIter("Post thrust::sort_by_key", iter);


            // Zero cell array values by setting cellStart to its highest
            // possible value, specified with pointer value 0xffffffff, which
            // for a 32 bit unsigned int is 4294967295.
            cudaMemset(dev_cellStart, 0xffffffff, 
                    grid.num[0]*grid.num[1]*grid.num[2]*sizeof(unsigned int));
            cudaThreadSynchronize();
            checkForCudaErrorsIter("Post cudaMemset", iter);

            // Use sorted order to reorder particle arrays (position,
            // velocities, radii) to ensure coherent memory access. Save ordered
            // configurations in new arrays (*_sorted).
            if (PROFILING == 1)
                startTimer(&kernel_tic);
            reorderArrays<<<dimGrid, dimBlock, smemSize>>>(dev_cellStart, 
                    dev_cellEnd,
                    dev_gridParticleCellID, 
                    dev_gridParticleIndex,
                    dev_x, dev_vel, 
                    dev_angvel,
                    dev_x_sorted, 
                    dev_vel_sorted, 
                    dev_angvel_sorted);

            // Synchronization point
            cudaThreadSynchronize();
            if (PROFILING == 1)
                stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed,
                        &t_reorderArrays);
            checkForCudaErrorsIter("Post reorderArrays", iter);

            // The contact search in topology() is only necessary for
            // determining the accumulated shear distance needed in the linear
            // elastic and nonlinear contact force model
            if (params.contactmodel == 2 || params.contactmodel == 3) {
                // For each particle: Search contacts in neighbor cells
                if (PROFILING == 1)
                    startTimer(&kernel_tic);
                topology<<<dimGrid, dimBlock>>>(dev_cellStart, 
                        dev_cellEnd,
                        dev_gridParticleIndex,
                        dev_x_sorted, 
                        dev_contacts,
                        dev_distmod);

                // Synchronization point
                cudaThreadSynchronize();
                if (PROFILING == 1)
                    stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed,
                            &t_topology);
                checkForCudaErrorsIter(
                        "Post topology: One or more particles moved "
                        "outside the grid.\nThis could possibly be caused by a "
                        "numerical instability.\nIs the computational time step"
                        " too large?", iter);
            }

            // For each particle process collisions and compute resulting forces
            //cudaPrintfInit();
            if (PROFILING == 1)
                startTimer(&kernel_tic);
            interact<<<dimGrid, dimBlock>>>(dev_gridParticleIndex,
                    dev_cellStart,
                    dev_cellEnd,
                    dev_x,
                    dev_x_sorted,
                    dev_vel_sorted,
                    dev_angvel_sorted,
                    dev_vel,
                    dev_angvel,
                    dev_force, 
                    dev_torque, 
                    dev_es_dot,
                    dev_ev_dot, 
                    dev_es,
                    dev_ev,
                    dev_p,
                    dev_walls_nx,
                    dev_walls_mvfd,
                    dev_walls_force_pp,
                    dev_contacts,
                    dev_distmod,
                    dev_delta_t);

            // Synchronization point
            cudaThreadSynchronize();
            //cudaPrintfDisplay(stdout, true);
            if (PROFILING == 1)
                stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed,
                        &t_interact);
            checkForCudaErrorsIter(
                    "Post interact - often caused if particles move "
                    "outside the grid", iter);

            // Process particle pairs
            if (params.nb0 > 0) {
                if (PROFILING == 1)
                    startTimer(&kernel_tic);
                bondsLinear<<<dimGridBonds, dimBlock>>>(
                        dev_bonds,
                        dev_bonds_delta,
                        dev_bonds_omega,
                        dev_x,
                        dev_vel,
                        dev_angvel,
                        dev_force,
                        dev_torque);
                // Synchronization point
                cudaThreadSynchronize();
                //cudaPrintfDisplay(stdout, true);
                if (PROFILING == 1)
                    stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed,
                            &t_bondsLinear);
                checkForCudaErrorsIter("Post bondsLinear", iter);
            }
        }

        // Solve fluid flow through the grid
        if (fluid == 1) {

            // Navier-Stokes solution
            if (cfd_solver == 0) {

                checkForCudaErrorsIter("Before findPorositiesDev", iter);
                // Find cell porosities, average particle velocities, and
                // average particle diameters. These are needed for predicting
                // the fluid velocities
                if (PROFILING == 1)
                    startTimer(&kernel_tic);
                findPorositiesVelocitiesDiametersSpherical
                //findPorositiesVelocitiesDiametersSphericalGradient
                    <<<dimGridFluid, dimBlockFluid>>>(
                            dev_cellStart,
                            dev_cellEnd,
                            dev_x_sorted,
                            dev_vel_sorted,
                            dev_ns_phi,
                            dev_ns_dphi,
                            dev_ns_vp_avg,
                            dev_ns_d_avg,
                            iter,
                            np,
                            ns.c_phi);
                cudaThreadSynchronize();
                if (PROFILING == 1)
                    stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed,
                            &t_findPorositiesDev);
                checkForCudaErrorsIter("Post findPorositiesDev", iter);

#ifdef CFD_DEM_COUPLING
                /*if (params.nu <= 0.0) {
                  std::cerr << "Error! The fluid needs a positive viscosity "
                  "value in order to simulate particle-fluid interaction."
                  << std::endl;
                  exit(1);
                  }*/
                if (iter == 0) {
                    // set cell center ghost nodes
                    setNSghostNodes<Float3><<<dimGridFluid, dimBlockFluid>>>(
                            dev_ns_v, ns.bc_bot, ns.bc_top);

                    // find cell face velocities
                    interpolateCenterToFace
                        <<<dimGridFluidFace, dimBlockFluidFace>>>(
                                dev_ns_v,
                                dev_ns_v_x,
                                dev_ns_v_y,
                                dev_ns_v_z);
                    cudaThreadSynchronize();
                    checkForCudaErrors("Post interpolateCenterToFace");
                }

                setNSghostNodesFace<Float>
                    <<<dimGridFluidFace, dimBlockFluidFace>>>(
                            dev_ns_v_x,
                            dev_ns_v_y,
                            dev_ns_v_z,
                            ns.bc_bot,
                            ns.bc_top);
                cudaThreadSynchronize();
                checkForCudaErrorsIter("Post setNSghostNodesFace", iter);

                findFaceDivTau<<<dimGridFluidFace, dimBlockFluidFace>>>(
                        dev_ns_v_x,
                        dev_ns_v_y,
                        dev_ns_v_z,
                        ns.mu,
                        dev_ns_div_tau_x,
                        dev_ns_div_tau_y,
                        dev_ns_div_tau_z);
                cudaThreadSynchronize();
                checkForCudaErrorsIter("Post findFaceDivTau", iter);

                setNSghostNodesFace<Float>
                    <<<dimGridFluidFace, dimBlockFluid>>>(
                            dev_ns_div_tau_x,
                            dev_ns_div_tau_y,
                            dev_ns_div_tau_z,
                            ns.bc_bot,
                            ns.bc_top);
                cudaThreadSynchronize();
                checkForCudaErrorsIter("Post setNSghostNodes(dev_ns_div_tau)",
                        iter);

                setNSghostNodes<Float><<<dimGridFluid, dimBlockFluid>>>(
                        dev_ns_p, ns.bc_bot, ns.bc_top);
                cudaThreadSynchronize();
                checkForCudaErrorsIter("Post setNSghostNodes(dev_ns_p)", iter);

                setNSghostNodes<Float><<<dimGridFluid, dimBlockFluid>>>(
                        dev_ns_phi, ns.bc_bot, ns.bc_top);
                cudaThreadSynchronize();
                checkForCudaErrorsIter("Post setNSghostNodes(dev_ns_p)", iter);


                if (np > 0) {

                    // Per particle, find the fluid-particle interaction force
                    // f_pf and apply it to the particle
                    findInteractionForce<<<dimGrid, dimBlock>>>(
                            dev_x,
                            dev_vel,
                            dev_ns_phi,
                            dev_ns_p,
                            dev_ns_v_x,
                            dev_ns_v_y,
                            dev_ns_v_z,
                            dev_ns_div_tau_x,
                            dev_ns_div_tau_y,
                            dev_ns_div_tau_z,
                            //ns.c_v,
                            ns.mu,
                            ns.rho_f,
                            dev_ns_f_pf,
                            dev_force,
                            dev_ns_f_d,
                            dev_ns_f_p,
                            dev_ns_f_v,
                            dev_ns_f_sum);
                    cudaThreadSynchronize();
                    checkForCudaErrorsIter("Post findInteractionForce", iter);

                    setNSghostNodes<Float><<<dimGridFluid, dimBlockFluid>>>(
                            dev_ns_p, ns.bc_bot, ns.bc_top);
                    cudaThreadSynchronize();
                    checkForCudaErrorsIter("Post setNSghostNodes(dev_ns_p)",
                            iter);

                    // Apply fluid-particle interaction force to the fluid
                    applyInteractionForceToFluid
                        <<<dimGridFluid, dimBlockFluid>>>(
                            dev_gridParticleIndex,
                            dev_cellStart,
                            dev_cellEnd,
                            dev_ns_f_pf,
                            dev_ns_F_pf);
                    //dev_ns_F_pf_x,
                    //dev_ns_F_pf_y,
                    //dev_ns_F_pf_z);
                    cudaThreadSynchronize();
                    checkForCudaErrorsIter("Post applyInteractionForceToFluid",
                            iter);

                    setNSghostNodes<Float3><<<dimGridFluid, dimBlockFluid>>>(
                            dev_ns_F_pf, ns.bc_bot, ns.bc_top);
                    cudaThreadSynchronize();
                    checkForCudaErrorsIter("Post setNSghostNodes(F_pf)", iter);
                }
#endif

                if ((iter % ns.ndem) == 0) {
                    // Initial guess for the top epsilon values. These may be
                    // changed in setUpperPressureNS
                    // TODO: Check if this should only be set when top bc=Dirichlet
                    Float pressure = ns.p[idx(0,0,ns.nz-1)];
                    Float pressure_new = pressure; // Dirichlet
                    Float epsilon_value = pressure_new - ns.beta*pressure;
                    setNSepsilonTop<<<dimGridFluid, dimBlockFluid>>>(
                            dev_ns_epsilon,
                            dev_ns_epsilon_new,
                            epsilon_value);
                    cudaThreadSynchronize();
                    checkForCudaErrorsIter("Post setNSepsilonTop", iter);

#if defined(REPORT_EPSILON) || defined(REPORT_V_P_COMPONENTS) || defined(REPORT_V_C_COMPONENTS)
                    std::cout
                        << "\n\n@@@@@@ TIME STEP " << iter << " @@@"
                        << std::endl;
#endif

                    // find cell containing top wall
                    if (walls.nw > 0 &&
                            (walls.wmode[0] == 1 || walls.wmode[0] == 3)) {
                        wall0_iz = walls.nx->w/(grid.L[2]/grid.num[2]);
                        setNSepsilonAtTopWall<<<dimGridFluid, dimBlockFluid>>>(
                                dev_ns_epsilon,
                                dev_ns_epsilon_new,
                                wall0_iz,
                                epsilon_value,
                                dp_dz);
                        cudaThreadSynchronize();
                        checkForCudaErrorsIter("Post setNSepsilonAtTopWall",
                                iter);

#ifdef REPORT_EPSILON
                        std::cout
                            << "\n###### EPSILON setNSepsilonAtTopWall "
                            << "######" << std::endl;
                        transferNSepsilonFromGlobalDeviceMemory();
                        printNSarray(stdout, ns.epsilon, "epsilon");
#endif
                    }

                    // Modulate the pressures at the upper boundary cells
                    if ((ns.p_mod_A > 1.0e-5 || ns.p_mod_A < -1.0e-5) &&
                            ns.p_mod_f > 1.0e-7) {
                        // original pressure
                        Float new_pressure = ns.p[idx(0,0,ns.nz-1)]
                            + ns.p_mod_A*sin(2.0*M_PI*ns.p_mod_f*time.current
                                    + ns.p_mod_phi);
                        setUpperPressureNS<<<dimGridFluid, dimBlockFluid>>>(
                                dev_ns_p,
                                dev_ns_epsilon,
                                dev_ns_epsilon_new,
                                ns.beta,
                                new_pressure);
                        cudaThreadSynchronize();
                        checkForCudaErrorsIter("Post setUpperPressureNS", iter);

#ifdef REPORT_MORE_EPSILON
                        std::cout
                            << "\n@@@@@@ TIME STEP " << iter << " @@@@@@"
                            << "\n###### EPSILON AFTER setUpperPressureNS "
                            << "######" << std::endl;
                        transferNSepsilonFromGlobalDeviceMemory();
                        printNSarray(stdout, ns.epsilon, "epsilon");
#endif
                    }

                    // Set the values of the ghost nodes in the grid
                    if (PROFILING == 1)
                        startTimer(&kernel_tic);

                    setNSghostNodes<Float><<<dimGridFluid, dimBlockFluid>>>(
                            dev_ns_p, ns.bc_bot, ns.bc_top);

                    //setNSghostNodes<Float3><<<dimGridFluid, dimBlockFluid>>>(
                    //dev_ns_v, ns.bc_bot, ns.bc_top);

                    setNSghostNodesFace<Float>
                        <<<dimGridFluidFace, dimBlockFluidFace>>>(
                                dev_ns_v_p_x,
                                dev_ns_v_p_y,
                                dev_ns_v_p_z,
                                ns.bc_bot, ns.bc_top);

                    setNSghostNodes<Float><<<dimGridFluid, dimBlockFluid>>>(
                            dev_ns_phi, ns.bc_bot, ns.bc_top);

                    setNSghostNodes<Float><<<dimGridFluid, dimBlockFluid>>>(
                            dev_ns_dphi, ns.bc_bot, ns.bc_top);

                    cudaThreadSynchronize();
                    if (PROFILING == 1)
                        stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed,
                                &t_setNSghostNodesDev);
                    checkForCudaErrorsIter("Post setNSghostNodesDev", iter);
                    /*std::cout
                      << "\n###### EPSILON AFTER setNSghostNodesDev #####"
                      << std::endl;
                      transferNSepsilonFromGlobalDeviceMemory();
                      printNSarray(stdout, ns.epsilon, "epsilon");*/

                    // interpolate velocities to cell centers which makes
                    // velocity prediction easier
                    interpolateFaceToCenter<<<dimGridFluid, dimBlockFluid>>>(
                            dev_ns_v_x,
                            dev_ns_v_y,
                            dev_ns_v_z,
                            dev_ns_v);
                    cudaThreadSynchronize();
                    checkForCudaErrorsIter(
                            "Post interpolateFaceToCenter", iter);

                    // Set cell-center velocity ghost nodes
                    setNSghostNodes<Float3><<<dimGridFluid, dimBlockFluid>>>(
                            dev_ns_v, ns.bc_bot, ns.bc_top);
                    cudaThreadSynchronize();
                    checkForCudaErrorsIter("Post setNSghostNodes(v)", iter);

                    // Find the divergence of phi*vi*v, needed for predicting
                    // the fluid velocities
                    if (PROFILING == 1)
                        startTimer(&kernel_tic);
                    findNSdivphiviv<<<dimGridFluid, dimBlockFluid>>>(
                            dev_ns_phi,
                            dev_ns_v,
                            dev_ns_div_phi_vi_v);
                    cudaThreadSynchronize();
                    if (PROFILING == 1)
                        stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed,
                                &t_findNSdivphiviv);
                    checkForCudaErrorsIter("Post findNSdivphiviv", iter);

                    // Set cell-center ghost nodes
                    setNSghostNodes<Float3><<<dimGridFluid, dimBlockFluid>>>(
                            dev_ns_div_phi_vi_v, ns.bc_bot, ns.bc_top);
                    cudaThreadSynchronize();
                    checkForCudaErrorsIter("Post setNSghostNodes(div_phi_vi_v)",
                            iter);

                    // Predict the fluid velocities on the base of the old
                    // pressure field and ignoring the incompressibility
                    // constraint
                    if (PROFILING == 1)
                        startTimer(&kernel_tic);
                    findPredNSvelocities<<<dimGridFluidFace, dimBlockFluidFace>>>(
                            dev_ns_p,
                            dev_ns_v_x,
                            dev_ns_v_y,
                            dev_ns_v_z,
                            dev_ns_phi,
                            dev_ns_dphi,
                            dev_ns_div_tau_x,
                            dev_ns_div_tau_y,
                            dev_ns_div_tau_z,
                            dev_ns_div_phi_vi_v,
                            ns.bc_bot,
                            ns.bc_top,
                            ns.beta,
                            dev_ns_F_pf,
                            ns.ndem,
                            wall0_iz,
                            ns.c_v,
                            ns.mu,
                            ns.rho_f,
                            dev_ns_v_p_x,
                            dev_ns_v_p_y,
                            dev_ns_v_p_z);
                    cudaThreadSynchronize();
                    if (PROFILING == 1)
                        stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed,
                                &t_findPredNSvelocities);
                    checkForCudaErrorsIter("Post findPredNSvelocities", iter);

                    setNSghostNodesFace<Float>
                        <<<dimGridFluidFace, dimBlockFluidFace>>>(
                                dev_ns_v_p_x,
                                dev_ns_v_p_y,
                                dev_ns_v_p_z,
                                ns.bc_bot, ns.bc_top);
                    cudaThreadSynchronize();
                    checkForCudaErrorsIter(
                            "Post setNSghostNodesFace(dev_ns_v_p)", iter);

                    interpolateFaceToCenter<<<dimGridFluid, dimBlockFluid>>>(
                            dev_ns_v_p_x,
                            dev_ns_v_p_y,
                            dev_ns_v_p_z,
                            dev_ns_v_p);
                    cudaThreadSynchronize();
                    checkForCudaErrorsIter(
                            "Post interpolateFaceToCenter", iter);


                    // In the first iteration of the sphere program, we'll need
                    // to manually estimate the values of epsilon. In the
                    // subsequent iterations, the previous values are  used.
                    if (iter == 0) {

                        // Define the first estimate of the values of epsilon.
                        // The initial guess depends on the value of ns.beta.
                        Float pressure = ns.p[idx(2,2,2)];
                        Float pressure_new = pressure; // Guess p_curr = p_new
                        Float epsilon_value = pressure_new - ns.beta*pressure;
                        if (PROFILING == 1)
                            startTimer(&kernel_tic);
                        setNSepsilonInterior<<<dimGridFluid, dimBlockFluid>>>(
                                dev_ns_epsilon, epsilon_value);
                        cudaThreadSynchronize();

                        setNSnormZero<<<dimGridFluid, dimBlockFluid>>>
                            (dev_ns_norm);
                        cudaThreadSynchronize();

                        if (PROFILING == 1)
                            stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed,
                                    &t_setNSepsilon);
                        checkForCudaErrorsIter("Post setNSepsilonInterior",
                                iter);

#ifdef REPORT_MORE_EPSILON
                        std::cout
                            << "\n###### EPSILON AFTER setNSepsilonInterior "
                            << "######" << std::endl;
                        transferNSepsilonFromGlobalDeviceMemory();
                        printNSarray(stdout, ns.epsilon, "epsilon");
#endif

                        // Set the epsilon values at the lower boundary
                        pressure = ns.p[idx(0,0,0)];
                        pressure_new = pressure; // Guess p_current = p_new
                        epsilon_value = pressure_new - ns.beta*pressure;
                        if (PROFILING == 1)
                            startTimer(&kernel_tic);
                        setNSepsilonBottom<<<dimGridFluid, dimBlockFluid>>>(
                                dev_ns_epsilon,
                                dev_ns_epsilon_new,
                                epsilon_value);
                        cudaThreadSynchronize();
                        if (PROFILING == 1)
                            stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed,
                                    &t_setNSdirichlet);
                        checkForCudaErrorsIter("Post setNSepsilonBottom", iter);

#ifdef REPORT_MORE_EPSILON
                        std::cout
                            << "\n###### EPSILON AFTER setNSepsilonBottom "
                            << "######" << std::endl;
                        transferNSepsilonFromGlobalDeviceMemory();
                        printNSarray(stdout, ns.epsilon, "epsilon");
#endif

                        /*setNSghostNodes<Float>
                          <<<dimGridFluid, dimBlockFluid>>>(
                          dev_ns_epsilon);
                          cudaThreadSynchronize();
                          checkForCudaErrors(
                          "Post setNSghostNodesFloat(dev_ns_epsilon)",
                          iter);*/
                        setNSghostNodes<Float><<<dimGridFluid, dimBlockFluid>>>(
                                dev_ns_epsilon,
                                ns.bc_bot, ns.bc_top);
                        cudaThreadSynchronize();
                        checkForCudaErrorsIter("Post setNSghostNodesEpsilon(1)",
                                iter);

#ifdef REPORT_MORE_EPSILON
                        std::cout <<
                            "\n###### EPSILON AFTER setNSghostNodes(epsilon) "
                            << "######" << std::endl;
                        transferNSepsilonFromGlobalDeviceMemory();
                        printNSarray(stdout, ns.epsilon, "epsilon");
#endif
                    }

                    // Solve the system of epsilon using a Jacobi iterative
                    // solver.  The average normalized residual is initialized
                    // to a large value.
                    //double avg_norm_res;
                    double max_norm_res;

                    // Write a log file of the normalized residuals during the
                    // Jacobi iterations
                    std::ofstream reslog;
                    if (write_res_log == 1)
                        reslog.open("max_res_norm.dat");

                    // transfer normalized residuals from GPU to CPU
#ifdef REPORT_MORE_EPSILON
                    std::cout << "\n###### BEFORE FIRST JACOBI ITERATION ######"
                        << "\n@@@@@@ TIME STEP " << iter << " @@@@@@"
                        << std::endl;
                    transferNSepsilonFromGlobalDeviceMemory();
                    printNSarray(stdout, ns.epsilon, "epsilon");
#endif

                    for (unsigned int nijac = 0; nijac<ns.maxiter; ++nijac) {

                        // Only grad(epsilon) changes during the Jacobi
                        // iterations.  The remaining terms of the forcing
                        // function are only calculated during the first
                        // iteration.
                        if (PROFILING == 1)
                            startTimer(&kernel_tic);
                        findNSforcing<<<dimGridFluid, dimBlockFluid>>>(
                                dev_ns_epsilon,
                                dev_ns_phi,
                                dev_ns_dphi,
                                dev_ns_v_p,
                                dev_ns_v_p_x,
                                dev_ns_v_p_y,
                                dev_ns_v_p_z,
                                nijac,
                                ns.ndem,
                                ns.c_v,
                                ns.rho_f,
                                dev_ns_f1,
                                dev_ns_f2,
                                dev_ns_f);
                        cudaThreadSynchronize();
                        if (PROFILING == 1)
                            stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed,
                                    &t_findNSforcing);
                        checkForCudaErrorsIter("Post findNSforcing", iter);
                        /*setNSghostNodesForcing
                          <<dimGridFluid, dimBlockFluid>>>(
                          dev_ns_f1,
                          dev_ns_f2,
                          dev_ns_f,
                          nijac);
                          cudaThreadSynchronize();
                          checkForCudaErrors("Post setNSghostNodesForcing",
                          iter);*/

                        setNSghostNodes<Float><<<dimGridFluid, dimBlockFluid>>>(
                                dev_ns_epsilon,
                                ns.bc_bot, ns.bc_top);
                        cudaThreadSynchronize();
                        checkForCudaErrorsIter("Post setNSghostNodesEpsilon(2)",
                                iter);

#ifdef REPORT_EPSILON
                        std::cout << "\n###### JACOBI ITERATION "
                            << nijac
                            << " after setNSghostNodes(epsilon,2) ######"
                            << std::endl;
                        transferNSepsilonFromGlobalDeviceMemory();
                        printNSarray(stdout, ns.epsilon, "epsilon");
#endif

                        // Perform a single Jacobi iteration
                        if (PROFILING == 1)
                            startTimer(&kernel_tic);
                        jacobiIterationNS<<<dimGridFluid, dimBlockFluid>>>(
                                dev_ns_epsilon,
                                dev_ns_epsilon_new,
                                dev_ns_norm,
                                dev_ns_f,
                                ns.bc_bot,
                                ns.bc_top,
                                ns.theta,
                                wall0_iz,
                                dp_dz);
                        cudaThreadSynchronize();
                        if (PROFILING == 1)
                            stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed,
                                    &t_jacobiIterationNS);
                        checkForCudaErrorsIter("Post jacobiIterationNS", iter);

                        // set Dirichlet and Neumann BC at cells containing top
                        // wall
                        /*if (walls.nw > 0 && walls.wmode[0] == 1) {
                          setNSepsilonAtTopWall
                          <<<dimGridFluid, dimBlockFluid>>>(
                          dev_ns_epsilon,
                          dev_ns_epsilon_new,
                          wall0_iz,
                          epsilon_value,
                          dp_dz);
                          cudaThreadSynchronize();
                          checkForCudaErrorsIter("Post setNSepsilonAtTopWall",
                          iter);
                          }*/

                        // Copy new values to current values
                        copyValues<Float><<<dimGridFluid, dimBlockFluid>>>(
                                dev_ns_epsilon_new,
                                dev_ns_epsilon);
                        cudaThreadSynchronize();
                        checkForCudaErrorsIter
                            ("Post copyValues (epsilon_new->epsilon)", iter);

#ifdef REPORT_EPSILON
                        std::cout << "\n###### JACOBI ITERATION "
                            << nijac << " after jacobiIterationNS ######"
                            << std::endl;
                        transferNSepsilonFromGlobalDeviceMemory();
                        printNSarray(stdout, ns.epsilon, "epsilon");
#endif

                        if (nijac % nijacnorm == 0) {

                            // Read the normalized residuals from the device
                            transferNSnormFromGlobalDeviceMemory();

                            // Write the normalized residuals to the terminal
                            //printNSarray(stdout, ns.norm, "norm");

                            // Find the maximum value of the normalized
                            // residuals
                            max_norm_res = maxNormResNS();

                            // Write the Jacobi iteration number and maximum
                            // value of the normalized residual to the log file
                            if (write_res_log == 1)
                                reslog << nijac << '\t' << max_norm_res
                                    << std::endl;
                        }

                        if (max_norm_res < ns.tolerance) {

                            if (write_conv_log == 1
                                    && iter % conv_log_interval == 0)
                                convlog << iter+1 << '\t' << nijac << std::endl;

                            setNSghostNodes<Float>
                                <<<dimGridFluid, dimBlockFluid>>>(
                                        dev_ns_epsilon,
                                        ns.bc_bot, ns.bc_top);
                            cudaThreadSynchronize();
                            checkForCudaErrorsIter
                                ("Post setNSghostNodesEpsilon(4)", iter);

                            // Apply smoothing if requested
                            if (ns.gamma > 0.0) {

                                smoothing<<<dimGridFluid, dimBlockFluid>>>(
                                        dev_ns_epsilon,
                                        ns.gamma,
                                        ns.bc_bot, ns.bc_top);
                                cudaThreadSynchronize();
                                checkForCudaErrorsIter("Post smoothing", iter);

                                setNSghostNodes<Float>
                                    <<<dimGridFluid, dimBlockFluid>>>(
                                            dev_ns_epsilon,
                                            ns.bc_bot, ns.bc_top);
                                cudaThreadSynchronize();
                                checkForCudaErrorsIter
                                    ("Post setNSghostNodesEpsilon(4)", iter);
                            }

#ifdef REPORT_EPSILON
                            std::cout << "\n###### JACOBI ITERATION "
                                << nijac << " after smoothing ######"
                                << std::endl;
                            transferNSepsilonFromGlobalDeviceMemory();
                            printNSarray(stdout, ns.epsilon, "epsilon");
#endif

                            break;  // solution has converged, exit Jacobi loop
                        }

                        if (nijac >= ns.maxiter-1) {

                            if (write_conv_log == 1)
                                convlog << iter+1 << '\t' << nijac << std::endl;

                            std::cerr << "\nIteration " << iter << ", time " 
                                << iter*time.dt << " s: "
                                "Error, the epsilon solution in the fluid "
                                "calculations did not converge. Try increasing "
                                "the value of 'ns.maxiter' (" << ns.maxiter
                                << ") or increase 'ns.tolerance' ("
                                << ns.tolerance << ")." << std::endl;
                        }
                        //break; // end after Jacobi first iteration
                    } // end Jacobi iteration loop

                    if (write_res_log == 1)
                        reslog.close();

                    // Find the new pressures and velocities
                    if (PROFILING == 1)
                        startTimer(&kernel_tic);

                    updateNSpressure<<<dimGridFluid, dimBlockFluid>>>(
                            dev_ns_epsilon,
                            ns.beta,
                            dev_ns_p);
                    cudaThreadSynchronize();
                    checkForCudaErrorsIter("Post updateNSpressure", iter);

                    updateNSvelocity<<<dimGridFluidFace, dimBlockFluidFace>>>(
                            dev_ns_v_p_x,
                            dev_ns_v_p_y,
                            dev_ns_v_p_z,
                            dev_ns_phi,
                            dev_ns_epsilon,
                            ns.beta,
                            ns.bc_bot,
                            ns.bc_top,
                            ns.ndem,
                            ns.c_v,
                            ns.rho_f,
                            wall0_iz,
                            iter,
                            dev_ns_v_x,
                            dev_ns_v_y,
                            dev_ns_v_z);
                    cudaThreadSynchronize();
                    if (PROFILING == 1)
                        stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed,
                                &t_updateNSvelocityPressure);
                    checkForCudaErrorsIter("Post updateNSvelocity", iter);

                    setNSghostNodesFace<Float>
                        <<<dimGridFluidFace, dimBlockFluidFace>>>(
                                dev_ns_v_p_x,
                                dev_ns_v_p_y,
                                dev_ns_v_p_z,
                                ns.bc_bot, ns.bc_top);
                    cudaThreadSynchronize();
                    checkForCudaErrorsIter(
                            "Post setNSghostNodesFace(dev_ns_v)", iter);

                    interpolateFaceToCenter<<<dimGridFluid, dimBlockFluid>>>(
                            dev_ns_v_x,
                            dev_ns_v_y,
                            dev_ns_v_z,
                            dev_ns_v);
                    cudaThreadSynchronize();
                    checkForCudaErrorsIter("Post interpolateFaceToCenter",
                            iter);
                } // end iter % ns.dem == 0
            } // end cfd_solver == 0

            // Darcy solution
            else if (cfd_solver == 1) { 

#if defined(REPORT_EPSILON) || defined(REPORT_FORCING_TERMS)
                std::cout << "\n\n@@@@@@ TIME STEP " << iter << " @@@"
                        << std::endl;
#endif

                if (walls.nw > 0 &&
                        (walls.wmode[0] == 1 || walls.wmode[0] == 3)) {
                    wall0_iz = walls.nx->w/(grid.L[2]/grid.num[2]);
                }

                if (np > 0) {

                    if (PROFILING == 1)
                        startTimer(&kernel_tic);
                    setDarcyGhostNodes<Float><<<dimGridFluid, dimBlockFluid>>>(
                            dev_darcy_p,
                            darcy.bc_xn, darcy.bc_xp,
                            darcy.bc_yn, darcy.bc_yp,
                            darcy.bc_bot, darcy.bc_top);
                    cudaThreadSynchronize();
                    if (PROFILING == 1)
                        stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed,
                                &t_setDarcyGhostNodes);
                    checkForCudaErrorsIter("Post setDarcyGhostNodes("
                            "dev_darcy_p) before findDarcyPressureForce", iter);

                    if (PROFILING == 1)
                        startTimer(&kernel_tic);
                    findDarcyPressureGradient<<<dimGridFluid, dimBlockFluid>>>(
                            dev_darcy_p,
                            dev_darcy_grad_p);
                    cudaThreadSynchronize();
                    checkForCudaErrorsIter("After findDarcyPressureGradient",
                            iter);

                    setDarcyGhostNodes<Float3><<<dimGridFluid, dimBlockFluid>>>(
                            dev_darcy_grad_p,
                            darcy.bc_xn, darcy.bc_xp,
                            darcy.bc_yn, darcy.bc_yp,
                            darcy.bc_bot, darcy.bc_top);
                    cudaThreadSynchronize();
                    if (PROFILING == 1)
                        stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed,
                                &t_setDarcyGhostNodes);
                    checkForCudaErrorsIter("Post setDarcyGhostNodes("
                            "dev_darcy_grad_p)", iter);

                    /*if (PROFILING == 1)
                        startTimer(&kernel_tic);
                    findDarcyPorositiesLinear<<<dimGridFluid, dimBlockFluid>>>(
                            dev_cellStart,
                            dev_cellEnd,
                            dev_x_sorted,
                            dev_vel_sorted,
                            iter,
                            darcy.ndem,
                            np,
                            darcy.c_phi,
                            dev_darcy_phi,
                            dev_darcy_dphi,
                            dev_darcy_div_v_p);
                    cudaThreadSynchronize();
                    if (PROFILING == 1)
                        stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed,
                                &t_findDarcyPorosities);
                    checkForCudaErrorsIter("Post findDarcyPorosities", iter);*/

                    /*findDarcyPressureForce<<<dimGrid, dimBlock>>>(
                            dev_x,
                            dev_darcy_p,
                            wall0_iz,
                            darcy.rho_f,
                            dev_force,
                            dev_darcy_f_p);*/
                    findDarcyPressureForceLinear<<<dimGrid, dimBlock>>>(
                            dev_x,
                            dev_darcy_grad_p,
                            dev_darcy_phi,
                            wall0_iz,
                            darcy.rho_f,
                            darcy.bc_top,
                            dev_force,
                            dev_darcy_f_p);
                    cudaThreadSynchronize();
                    if (PROFILING == 1)
                        stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed,
                                &t_findDarcyPressureForce);
                    checkForCudaErrorsIter("Post findDarcyPressureForce",
                            iter);
                }

                if ((iter % darcy.ndem) == 0) {

                    if (PROFILING == 1)
                        startTimer(&kernel_tic);
                    /*findDarcyPorosities<<<dimGridFluid, dimBlockFluid>>>(
                            dev_cellStart,
                            dev_cellEnd,
                            dev_x_sorted,
                            dev_vel_sorted,
                            iter,
                            darcy.ndem,
                            np,
                            darcy.c_phi,
                            dev_darcy_phi,
                            dev_darcy_dphi);*/
                    findDarcyPorositiesLinear<<<dimGridFluid, dimBlockFluid>>>(
                            dev_cellStart,
                            dev_cellEnd,
                            dev_x_sorted,
                            dev_vel_sorted,
                            iter,
                            darcy.ndem,
                            np,
                            darcy.c_phi,
                            dev_darcy_phi,
                            dev_darcy_dphi,
                            dev_darcy_div_v_p,
                            dev_darcy_vp_avg);
                    cudaThreadSynchronize();
                    if (PROFILING == 1)
                        stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed,
                                &t_findDarcyPorosities);
                    checkForCudaErrorsIter("Post findDarcyPorosities", iter);

                    // copy porosities to the frictionless Y boundaries
                    if (grid.periodic == 2) {
                        copyDarcyPorositiesToEdges<<<dimGridFluid, 
                            dimBlockFluid>>>(
                                dev_darcy_phi,
                                dev_darcy_dphi,
                                dev_darcy_div_v_p,
                                dev_darcy_vp_avg);
                        cudaThreadSynchronize();
                    }

                    // copy porosities to the frictionless lower Z boundary
                    if (grid.periodic == 2) {
                        copyDarcyPorositiesToBottom<<<dimGridFluid, 
                                dimBlockFluid>>>(
                                dev_darcy_phi,
                                dev_darcy_dphi,
                                dev_darcy_div_v_p,
                                dev_darcy_vp_avg);
                        cudaThreadSynchronize();
                    }

                    // Modulate the pressures at the upper boundary cells
                    if ((darcy.p_mod_A > 1.0e-5 || darcy.p_mod_A < -1.0e-5) &&
                            darcy.p_mod_f > 1.0e-7) {
                        // original pressure
                        Float new_pressure =
                            darcy.p_top_orig + darcy.p_mod_A
                            *sin(2.0*M_PI*darcy.p_mod_f*time.current
                                    + darcy.p_mod_phi);
                        if (PROFILING == 1)
                            startTimer(&kernel_tic);
                        setDarcyTopPressure<<<dimGridFluid, dimBlockFluid>>>(
                                new_pressure,
                                dev_darcy_p,
                                wall0_iz);
                        cudaThreadSynchronize();
                        checkForCudaErrorsIter("Post setUpperPressureNS", iter);

                        // Modulate the pressures at the top wall
                        setDarcyTopWallPressure
                            <<<dimGridFluid, dimBlockFluid>>>(
                                    new_pressure,
                                    wall0_iz,
                                    dev_darcy_p);
                        cudaThreadSynchronize();
                        checkForCudaErrorsIter("Post setDarcyTopWallPressure",
                                iter);

                        if (PROFILING == 1)
                            stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed,
                                    &t_setDarcyTopPressure);
                    }

                    if (PROFILING == 1)
                        startTimer(&kernel_tic);
                    findDarcyPermeabilities<<<dimGridFluid, dimBlockFluid>>>(
                            darcy.k_c, dev_darcy_phi, dev_darcy_k);
                    cudaThreadSynchronize();
                    if (PROFILING == 1)
                        stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed,
                                &t_findDarcyPermeabilities);
                    checkForCudaErrorsIter("Post findDarcyPermeabilities",
                            iter);

                    if (PROFILING == 1)
                        startTimer(&kernel_tic);
                    setDarcyGhostNodes<Float><<<dimGridFluid, dimBlockFluid>>>(
                            dev_darcy_phi,
                            darcy.bc_xn, darcy.bc_xp,
                            darcy.bc_yn, darcy.bc_yp,
                            darcy.bc_bot, darcy.bc_top);
                    cudaThreadSynchronize();
                    if (PROFILING == 1)
                        stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed,
                                &t_setDarcyGhostNodes);
                    checkForCudaErrorsIter(
                            "Post setDarcyGhostNodes(dev_darcy_phi)", iter);

                    if (PROFILING == 1)
                        startTimer(&kernel_tic);
                    setDarcyGhostNodes<Float><<<dimGridFluid, dimBlockFluid>>>(
                            dev_darcy_k,
                            darcy.bc_xn, darcy.bc_xp,
                            darcy.bc_yn, darcy.bc_yp,
                            darcy.bc_bot, darcy.bc_top);
                    cudaThreadSynchronize();
                    if (PROFILING == 1)
                        stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed,
                                &t_setDarcyGhostNodes);
                    checkForCudaErrorsIter(
                            "Post setDarcyGhostNodes(dev_darcy_k)", iter);

                    if (PROFILING == 1)
                        startTimer(&kernel_tic);
                    findDarcyPermeabilityGradients
                        <<<dimGridFluid, dimBlockFluid>>>
                        (dev_darcy_k, dev_darcy_grad_k);
                    cudaThreadSynchronize();
                    if (PROFILING == 1)
                        stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed,
                                &t_findDarcyPermeabilityGradients);
                    checkForCudaErrorsIter(
                            "Post findDarcyPermeabilityGradients", iter);

                    if (iter == 0) {
                        setDarcyNormZero<<<dimGridFluid, dimBlockFluid>>>(
                                dev_darcy_norm);
                        cudaThreadSynchronize();
                        checkForCudaErrorsIter("Post setDarcyNormZero", iter);

                        if (PROFILING == 1)
                            startTimer(&kernel_tic);
                        copyValues<Float><<<dimGridFluid, dimBlockFluid>>>(
                                dev_darcy_p,
                                dev_darcy_p_old);
                        cudaThreadSynchronize();
                        if (PROFILING == 1)
                            stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed,
                                    &t_copyValues);
                        checkForCudaErrorsIter("Post copyValues(p -> p_old)",
                                iter);
                    }

                    if (darcy.bc_bot == 4 || darcy.bc_top == 4) {
                        if (PROFILING == 1)
                            startTimer(&kernel_tic);
                        setDarcyGhostNodesFlux<Float>
                            <<<dimGridFluid, dimBlockFluid>>>(
                                dev_darcy_p,
                                darcy.bc_bot,
                                darcy.bc_top,
                                darcy.bc_bot_flux,
                                darcy.bc_top_flux,
                                dev_darcy_k,
                                darcy.mu);
                        cudaThreadSynchronize();
                        if (PROFILING == 1)
                            stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed,
                                    &t_setDarcyGhostNodes);
                        checkForCudaErrorsIter(
                                "Post setDarcyGhostNodesFlux", iter);
                    }

                    // Solve the system of epsilon using a Jacobi iterative
                    // solver.  The average normalized residual is initialized
                    // to a large value.
                    //double avg_norm_res;
                    double max_norm_res;

                    // Write a log file of the normalized residuals during the
                    // Jacobi iterations
                    std::ofstream reslog;
                    if (write_res_log == 1)
                        reslog.open("max_res_norm.dat");

                    for (unsigned int nijac = 0; nijac<darcy.maxiter; ++nijac) {

#if defined(REPORT_EPSILON) || defined(REPORT_FORCING_TERMS)
                std::cout << "\n\n### Jacobi iteration " << nijac << std::endl;
#endif

                        if (nijac == 0) {
                            if (PROFILING == 1)
                                startTimer(&kernel_tic);
                            copyValues<Float><<<dimGridFluid, dimBlockFluid>>>(
                                    dev_darcy_p,
                                    dev_darcy_p_old);
                            cudaThreadSynchronize();
                            if (PROFILING == 1)
                                stopTimer(&kernel_tic, &kernel_toc,
                                        &kernel_elapsed, &t_copyValues);
                            checkForCudaErrorsIter(
                                    "Post copyValues(p -> p_old)", iter);
                        }

                        if (PROFILING == 1)
                            startTimer(&kernel_tic);
                        setDarcyGhostNodes<Float>
                            <<<dimGridFluid, dimBlockFluid>>>(
                                dev_darcy_p,
                                darcy.bc_xn, darcy.bc_xp,
                                darcy.bc_yn, darcy.bc_yp,
                                darcy.bc_bot, darcy.bc_top);
                        cudaThreadSynchronize();
                        if (PROFILING == 1)
                            stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed,
                                    &t_setDarcyGhostNodes);
                        checkForCudaErrorsIter("Post setDarcyGhostNodes("
                                "dev_darcy_p) in Jacobi loop", iter);

                        if (nijac == 0) {
                            if (PROFILING == 1)
                                startTimer(&kernel_tic);
                            firstDarcySolution<<<dimGridFluid, dimBlockFluid>>>(
                                    dev_darcy_p,
                                    dev_darcy_k,
                                    dev_darcy_phi,
                                    dev_darcy_dphi,
                                    dev_darcy_div_v_p,
                                    dev_darcy_vp_avg,
                                    dev_darcy_grad_k,
                                    darcy.beta_f,
                                    darcy.mu,
                                    darcy.bc_xn,
                                    darcy.bc_xp,
                                    darcy.bc_yn,
                                    darcy.bc_yp,
                                    darcy.bc_bot,
                                    darcy.bc_top,
                                    darcy.ndem,
                                    wall0_iz,
                                    dev_darcy_p_constant,
                                    dev_darcy_dp_expl);
                            cudaThreadSynchronize();
                            if (PROFILING == 1)
                                stopTimer(&kernel_tic, &kernel_toc,
                                        &kernel_elapsed,
                                        &t_updateDarcySolution);
                            checkForCudaErrorsIter("Post updateDarcySolution",
                                    iter);
                        }

                        if (PROFILING == 1)
                            startTimer(&kernel_tic);
                        updateDarcySolution<<<dimGridFluid, dimBlockFluid>>>(
                                dev_darcy_p_old,
                                //dev_darcy_dpdt,
                                dev_darcy_dp_expl,
                                dev_darcy_p,
                                dev_darcy_k,
                                dev_darcy_phi,
                                dev_darcy_dphi,
                                dev_darcy_div_v_p,
                                dev_darcy_vp_avg,
                                dev_darcy_grad_k,
                                darcy.beta_f,
                                darcy.mu,
                                darcy.bc_xn,
                                darcy.bc_xp,
                                darcy.bc_yn,
                                darcy.bc_yp,
                                darcy.bc_bot,
                                darcy.bc_top,
                                darcy.ndem,
                                wall0_iz,
                                dev_darcy_p_constant,
                                dev_darcy_p_new,
                                dev_darcy_norm);
                        cudaThreadSynchronize();
                        if (PROFILING == 1)
                            stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed,
                                    &t_updateDarcySolution);
                        checkForCudaErrorsIter("Post updateDarcySolution",
                                iter);

                        if (darcy.bc_top == 1) {
                            if (PROFILING == 1)
                                startTimer(&kernel_tic);
                            setDarcyTopWallFixedFlow
                                <<<dimGridFluid, dimBlockFluid>>>
                                (wall0_iz, dev_darcy_p);
                            cudaThreadSynchronize();
                            if (PROFILING == 1)
                                stopTimer(&kernel_tic, &kernel_toc,
                                        &kernel_elapsed,
                                        &t_updateDarcySolution);
                            checkForCudaErrorsIter(
                                    "Post setDarcyTopWallFixedFlow", iter);
                        }

                        if (darcy.bc_bot == 4 || darcy.bc_top == 4) {
                            if (PROFILING == 1)
                                startTimer(&kernel_tic);
                            setDarcyGhostNodesFlux<Float>
                                <<<dimGridFluid, dimBlockFluid>>>(
                                        dev_darcy_p,
                                        darcy.bc_bot,
                                        darcy.bc_top,
                                        darcy.bc_bot_flux,
                                        darcy.bc_top_flux,
                                        dev_darcy_k,
                                        darcy.mu);
                            cudaThreadSynchronize();
                            if (PROFILING == 1)
                                stopTimer(&kernel_tic, &kernel_toc,
                                        &kernel_elapsed,
                                        &t_setDarcyGhostNodes);
                            checkForCudaErrorsIter(
                                    "Post setDarcyGhostNodesFlux", iter);
                        }

                        // Copy new values to current values
                        if (PROFILING == 1)
                            startTimer(&kernel_tic);
                        copyValues<Float><<<dimGridFluid, dimBlockFluid>>>(
                                dev_darcy_p_new,
                                dev_darcy_p);
                        cudaThreadSynchronize();
                        if (PROFILING == 1)
                            stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed,
                                    &t_copyValues);
                        checkForCudaErrorsIter("Post copyValues(p_new -> p)",
                                iter);

#ifdef REPORT_EPSILON
                        std::cout << "\n###### JACOBI ITERATION "
                            << nijac << " after copyValues ######" << std::endl;
                        transferDarcyPressuresFromGlobalDeviceMemory();
                        printDarcyArray(stdout, darcy.p, "p");
#endif

                        if (nijac % nijacnorm == 0) {
                            // Read the normalized residuals from the device
                            transferDarcyNormFromGlobalDeviceMemory();

                            // Write the normalized residuals to the terminal
                            //printDarcyArray(stdout, darcy.norm, "norm");

                            // Find the maximum value of the normalized
                            // residuals
                            max_norm_res = maxNormResDarcy();

                            // Write the Jacobi iteration number and maximum
                            // value of the normalized residual to the log file
                            if (write_res_log == 1)
                                reslog << nijac << '\t' << max_norm_res
                                    << std::endl;

                            if (max_norm_res <= darcy.tolerance) {
                                if (write_conv_log == 1
                                        && iter % conv_log_interval == 0)
                                    convlog << iter+1 << '\t' << nijac
                                        << std::endl;

                                break;  // solution has converged
                            }
                        }

                        if (nijac == darcy.maxiter-1) {

                            if (write_conv_log == 1)
                                convlog << iter+1 << '\t' << nijac << std::endl;

                            std::cerr << "\nIteration " << iter << ", time " 
                                << iter*time.dt << " s: "
                                "Error, the pressure solution in the fluid "
                                "calculations did not converge. Try increasing "
                                "the value of 'darcy.maxiter' ("
                                << darcy.maxiter
                                << ") or increase 'darcy.tolerance' ("
                                << darcy.tolerance << ")." << std::endl;
                        }

                        if (write_res_log == 1)
                            reslog.close();

                        //break; // end after first iteration
                    }

                    // Zero all dphi values right after they are used in fluid
                    // solution, unless a file is written in this step.
                    if (filetimeclock + time.dt < time.file_dt) {
                        setDarcyZeros<Float> <<<dimGridFluid, dimBlockFluid>>>
                            (dev_darcy_dphi);
                        cudaThreadSynchronize();
                        checkForCudaErrorsIter(
                                "After setDarcyZeros(dev_darcy_dphi)", iter);
                    }

                    if (PROFILING == 1)
                        startTimer(&kernel_tic);
                    setDarcyGhostNodes<Float> <<<dimGridFluid, dimBlockFluid>>>
                        (dev_darcy_p,
                         darcy.bc_xn, darcy.bc_xp,
                         darcy.bc_yn, darcy.bc_yp,
                         darcy.bc_bot, darcy.bc_top);
                    cudaThreadSynchronize();
                    if (PROFILING == 1)
                        stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed,
                                &t_setDarcyGhostNodes);
                    checkForCudaErrorsIter("Post setDarcyGhostNodes("
                            "dev_darcy_p) after Jacobi loop", iter);

                    if (PROFILING == 1)
                        startTimer(&kernel_tic);
                    findDarcyVelocities<<<dimGridFluid, dimBlockFluid>>>(
                            dev_darcy_p,
                            dev_darcy_phi,
                            dev_darcy_k,
                            darcy.mu,
                            dev_darcy_v);
                    cudaThreadSynchronize();
                    if (PROFILING == 1)
                        stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed,
                                &t_findDarcyVelocities);
                    checkForCudaErrorsIter("Post findDarcyVelocities", iter);
                }
            }
        }
        //break; // end after first iteration

        if (np > 0) {

            // Find shear stresses on upper fixed particles if a shear stress BC
            // is specified (wmode[0] == 3)
            if (walls.nw > 0 && walls.wmode[0] == 3) {

                if (PROFILING == 1)
                    startTimer(&kernel_tic);
                findShearStressOnFixedMovingParticles<<<dimGrid, dimBlock>>>
                    (dev_x,
                     dev_vel,
                     dev_force,
                     dev_walls_tau_eff_x_pp);
                cudaThreadSynchronize();
                if (PROFILING == 1)
                    stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed,
                            &t_summation);
                checkForCudaErrorsIter(
                        "Post findShearStressOnFixedMovingParticles", iter);

                if (PROFILING == 1)
                    startTimer(&kernel_tic);
                summation<<<dimGrid, dimBlock>>>(dev_walls_tau_eff_x_pp,
                        dev_walls_tau_eff_x_partial);
                cudaThreadSynchronize();
                if (PROFILING == 1)
                    stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed,
                            &t_summation);
                checkForCudaErrorsIter("Post shear stress summation", iter);
            }

            // Determine whether it is time to step the velocity
            if (time.current >= v2_start && time.current < v2_end &&
                    velocity_state == 1) {
                change_velocity_state = 1.0;
                velocity_state = 2;
            } else if (time.current >= v2_end && velocity_state == 2) {
                change_velocity_state = -1.0;
                velocity_state = 1;
            }

            updateFixvel<<<dimGrid,dimBlock>>>(dev_x_sorted, dev_walls_nx, dev_vel_sorted);
	    cudaThreadSynchronize();

            // Update particle kinematics
            if (PROFILING == 1)
                startTimer(&kernel_tic);
            integrate<<<dimGrid, dimBlock>>>(dev_x_sorted, 
                    dev_vel_sorted, 
                    dev_angvel_sorted,
                    dev_x, 
                    dev_vel, 
                    dev_angvel,
                    dev_force,
                    dev_torque, 
                    dev_angpos,
                    dev_acc,
                    dev_angacc,
                    dev_vel0,
                    dev_angvel0,
                    dev_xyzsum,
                    dev_gridParticleIndex,
                    iter,
                    dev_walls_wmode,
                    dev_walls_mvfd,
                    dev_walls_tau_eff_x_partial,
                    dev_walls_tau_x,
                    walls.tau_x[0],
                    change_velocity_state,
                    velocity_factor,
                    blocksPerGrid);
            cudaThreadSynchronize();
            checkForCudaErrorsIter("Post integrate", iter);
            if (PROFILING == 1)
                stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed,
                        &t_integrate);

            if (change_velocity_state != 0)
                change_velocity_state = 0;

            // Summation of forces on wall
            if (PROFILING == 1)
                startTimer(&kernel_tic);
            if (walls.nw > 0) {
                summation<<<dimGrid, dimBlock>>>(dev_walls_force_pp,
                        dev_walls_force_partial);
            }
            cudaThreadSynchronize();
            if (PROFILING == 1)
                stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed,
                        &t_summation);
            checkForCudaErrorsIter("Post wall force summation", iter);

            // Update wall kinematics
            if (PROFILING == 1)
                startTimer(&kernel_tic);
            if (walls.nw > 0) {
                integrateWalls<<< 1, walls.nw>>>(
                        dev_walls_nx,
                        dev_walls_mvfd,
                        dev_walls_wmode,
                        dev_walls_force_partial,
                        dev_walls_acc,
                        blocksPerGrid,
                        time.current,
                        iter);
            }
            cudaThreadSynchronize();
            if (PROFILING == 1)
                stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed,
                        &t_integrateWalls);
            checkForCudaErrorsIter("Post integrateWalls", iter);
        }

        // Update timers and counters
        //time.current  = iter*time.dt;
        time.current  += time.dt;
        filetimeclock += time.dt;
        ++iter;

        // Make sure all preceding tasks are complete
        if (cudaDeviceSynchronize() != cudaSuccess) {
            cerr << "Error during cudaDeviceSynchronize()" << endl;
        }

        // Report time to console
        if (verbose == 1 && (iter % stdout_report == 0)) {

            toc = clock();
            time_spent = (toc - tic)/(CLOCKS_PER_SEC); // real time spent

            // Real time it takes to compute a second of model time
            t_ratio = time_spent/(time.current - t_start);
            time_t estimated_seconds_left(t_ratio*(time.total - time.current));
            tm *time_eta = gmtime(&estimated_seconds_left);

            cout << "\r  Current time: " << time.current << "/"
                << time.total << " s. ("
                << t_ratio << " s_real/s_sim, ETA: "
                << time_eta->tm_yday << "d "
                << std::setw(2) << std::setfill('0') << time_eta->tm_hour << ":"
                << std::setw(2) << std::setfill('0') << time_eta->tm_min << ":"
                << std::setw(2) << std::setfill('0') << time_eta->tm_sec
                << ")       "; // << std::flush;
        }


        // Produce output binary if the time interval 
        // between output files has been reached
        if (filetimeclock >= time.file_dt) {

            // Pause the CPU thread until all CUDA calls previously issued are
            // completed
            cudaThreadSynchronize();
            checkForCudaErrorsIter("Beginning of file output section", iter);

            // v_x, v_y, v_z -> v
            if (fluid == 1 && cfd_solver == 0) {
                interpolateFaceToCenter<<<dimGridFluid, dimBlockFluid>>>(
                        dev_ns_v_x,
                        dev_ns_v_y,
                        dev_ns_v_z,
                        dev_ns_v);
                cudaThreadSynchronize();
                checkForCudaErrorsIter("Post interpolateFaceToCenter", iter);
            }

            //// Copy device data to host memory
            transferFromGlobalDeviceMemory();
            checkForCudaErrorsIter("After transferFromGlobalDeviceMemory()",
                    iter);

            // Empty the dphi values after device to host transfer
            if (fluid == 1) {
                if (cfd_solver == 1) {
                    setDarcyZeros<Float> <<<dimGridFluid, dimBlockFluid>>>
                        (dev_darcy_dphi);
                    cudaThreadSynchronize();
                    checkForCudaErrorsIter(
                            "After setDarcyZeros(dev_darcy_dphi) after transfer",
                            iter);
                }
            }

            // Pause the CPU thread until all CUDA calls previously issued are
            // completed
            cudaThreadSynchronize();

            // Check the numerical stability of the NS solver
            if (fluid == 1)
                if (cfd_solver == 0)
                    checkNSstability();

            // Write binary output file
            time.step_count += 1;
            sprintf(file,"output/%s.output%05d.bin", sid.c_str(),
                    time.step_count);
            writebin(file);

            /*std::cout
              << "\n###### OUTPUT FILE " << time.step_count << " ######"
                << std::endl;
            transferNSepsilonFromGlobalDeviceMemory();
            printNSarray(stdout, ns.epsilon, "epsilon");*/

            // Write fluid arrays
            /*if (fluid == 1 && cfd_solver == 0) {
                sprintf(file,"output/%s.ns_phi.output%05d.bin", sid.c_str(),
                    time.step_count);
                writeNSarray(ns.phi, file);
            }*/

            if (CONTACTINFO == 1) {
                // Write contact information to stdout
                cout << "\n\n---------------------------\n"
                    << "t = " << time.current << " s.\n"
                    << "---------------------------\n";

                for (int n = 0; n < np; ++n) {
                    cout << "\n## Particle " << n << " ##\n";

                    cout  << "- contacts:\n";
                    for (int nc = 0; nc < NC; ++nc) 
                        cout << "[" << nc << "]=" << k.contacts[nc+NC*n] <<
                            '\n';

                    cout << "\n- delta_t:\n";
                    for (int nc = 0; nc < NC; ++nc) 
                        cout << k.delta_t[nc+NC*n].x << '\t'
                            << k.delta_t[nc+NC*n].y << '\t'
                            << k.delta_t[nc+NC*n].z << '\t'
                            << k.delta_t[nc+NC*n].w << '\n';

                    cout << "\n- distmod:\n";
                    for (int nc = 0; nc < NC; ++nc) 
                        cout << k.distmod[nc+NC*n].x << '\t'
                            << k.distmod[nc+NC*n].y << '\t'
                            << k.distmod[nc+NC*n].z << '\t'
                            << k.distmod[nc+NC*n].w << '\n';
                }
                cout << '\n';
            }

            // Update status.dat at the interval of filetime 
            outfile = "output/" + sid + ".status.dat";
            fp = fopen(outfile.c_str(), "w");
            fprintf(fp,"%2.4e %2.4e %d\n", 
                    time.current, 
                    100.0*time.current/time.total,
                    time.step_count);
            fclose(fp);

            filetimeclock = 0.0;
        }

        // Uncomment break command to stop after the first iteration
        //break;
    }

    if (write_conv_log == 1)
        convlog.close();


    // Stop clock and display calculation time spent
    toc = clock();
    cudaEventRecord(dev_toc, 0);
    cudaEventSynchronize(dev_toc);

    time_spent = (toc - tic)/(CLOCKS_PER_SEC);
    cudaEventElapsedTime(&dev_time_spent, dev_tic, dev_toc);

    if (verbose == 1) {
        cout << "\nSimulation ended. Statistics:\n"
            << "  - Last output file number: " 
            << time.step_count << "\n"
            << "  - GPU time spent: "
            << dev_time_spent/1000.0f << " s\n"
            << "  - CPU time spent: "
            << time_spent << " s\n"
            << "  - Mean duration of iteration:\n"
            << "      " << dev_time_spent/((double)iter*1000.0f) << " s"
            << std::endl; 
    }

    cudaEventDestroy(dev_tic);
    cudaEventDestroy(dev_toc);

    cudaEventDestroy(kernel_tic);
    cudaEventDestroy(kernel_toc);

    // Report time spent on each kernel
    if (PROFILING == 1 && verbose == 1) {
        double t_sum = t_calcParticleCellID + t_thrustsort + t_reorderArrays +
            t_topology + t_interact + t_bondsLinear + t_latticeBoltzmannD3Q19 +
            t_integrate + t_summation + t_integrateWalls + t_findPorositiesDev +
            t_findNSstressTensor +
            t_findNSdivphiviv + t_findNSdivtau + t_findPredNSvelocities +
            t_setNSepsilon + t_setNSdirichlet + t_setNSghostNodesDev +
            t_findNSforcing + t_jacobiIterationNS + t_updateNSvelocityPressure +
            t_findDarcyPorosities + t_setDarcyGhostNodes +
            t_findDarcyPressureForce + t_setDarcyTopPressure +
            t_findDarcyPermeabilities + t_findDarcyPermeabilityGradients +
            //t_findDarcyPressureChange +
            t_updateDarcySolution + t_copyValues + t_findDarcyVelocities;

        cout << "\nKernel profiling statistics:\n"
            << "  - calcParticleCellID:\t\t" << t_calcParticleCellID/1000.0
            << " s"
            << "\t(" << 100.0*t_calcParticleCellID/t_sum << " %)\n"
            << "  - thrustsort:\t\t\t" << t_thrustsort/1000.0 << " s"
            << "\t(" << 100.0*t_thrustsort/t_sum << " %)\n"
            << "  - reorderArrays:\t\t" << t_reorderArrays/1000.0 << " s"
            << "\t(" << 100.0*t_reorderArrays/t_sum << " %)\n";
        if (params.contactmodel == 2 || params.contactmodel == 3) {
            cout
            << "  - topology:\t\t\t" << t_topology/1000.0 << " s"
            << "\t(" << 100.0*t_topology/t_sum << " %)\n";
        }
        cout << "  - interact:\t\t\t" << t_interact/1000.0 << " s"
            << "\t(" << 100.0*t_interact/t_sum << " %)\n";
        if (params.nb0 > 0) {
            cout << "  - bondsLinear:\t\t" << t_bondsLinear/1000.0 << " s"
            << "\t(" << 100.0*t_bondsLinear/t_sum << " %)\n";
        }
        cout << "  - integrate:\t\t\t" << t_integrate/1000.0 << " s"
            << "\t(" << 100.0*t_integrate/t_sum << " %)\n"
            << "  - summation:\t\t\t" << t_summation/1000.0 << " s"
            << "\t(" << 100.0*t_summation/t_sum << " %)\n"
            << "  - integrateWalls:\t\t" << t_integrateWalls/1000.0 << " s"
            << "\t(" << 100.0*t_integrateWalls/t_sum << " %)\n";
        if (fluid == 1 && cfd_solver == 0) {
            cout << "  - findPorositiesDev:\t\t" << t_findPorositiesDev/1000.0
                << " s" << "\t(" << 100.0*t_findPorositiesDev/t_sum << " %)\n"
                << "  - findNSstressTensor:\t\t" << t_findNSstressTensor/1000.0
                << " s" << "\t(" << 100.0*t_findNSstressTensor/t_sum << " %)\n"
                << "  - findNSdivphiviv:\t\t" << t_findNSdivphiviv/1000.0
                << " s" << "\t(" << 100.0*t_findNSdivphiviv/t_sum << " %)\n"
                << "  - findNSdivtau:\t\t" << t_findNSdivtau/1000.0
                << " s" << "\t(" << 100.0*t_findNSdivtau/t_sum << " %)\n"
                << "  - findPredNSvelocities:\t" <<
                t_findPredNSvelocities/1000.0 << " s" << "\t(" <<
                100.0*t_findPredNSvelocities/t_sum << " %)\n"
                << "  - setNSepsilon:\t\t" << t_setNSepsilon/1000.0
                << " s" << "\t(" << 100.0*t_setNSepsilon/t_sum << " %)\n"
                << "  - setNSdirichlet:\t\t" << t_setNSdirichlet/1000.0
                << " s" << "\t(" << 100.0*t_setNSdirichlet/t_sum << " %)\n"
                << "  - setNSghostNodesDev:\t\t" << t_setNSghostNodesDev/1000.0
                << " s" << "\t(" << 100.0*t_setNSghostNodesDev/t_sum << " %)\n"
                << "  - findNSforcing:\t\t" << t_findNSforcing/1000.0 << " s"
                << "\t(" << 100.0*t_findNSforcing/t_sum << " %)\n"
                << "  - jacobiIterationNS:\t\t" << t_jacobiIterationNS/1000.0
                << " s"
                << "\t(" << 100.0*t_jacobiIterationNS/t_sum << " %)\n"
                << "  - updateNSvelocityPressure:\t"
                << t_updateNSvelocityPressure/1000.0 << " s"
                << "\t(" << 100.0*t_updateNSvelocityPressure/t_sum << " %)\n";
        } else if (fluid == 1 && cfd_solver == 1) {
            cout << "  - findDarcyPorosities:\t" <<
                t_findDarcyPorosities/1000.0 << " s" << "\t(" <<
                100.0*t_findDarcyPorosities/t_sum << " %)\n"
                << "  - setDarcyGhostNodes:\t\t" <<
                t_setDarcyGhostNodes/1000.0 << " s" << "\t(" <<
                100.0*t_setDarcyGhostNodes/t_sum << " %)\n"
                << "  - findDarcyPressureForce:\t" <<
                t_findDarcyPressureForce/1000.0 << " s" << "\t(" <<
                100.0*t_findDarcyPressureForce/t_sum << " %)\n"
                << "  - setDarcyTopPressure:\t" <<
                t_setDarcyTopPressure/1000.0 << " s" << "\t(" <<
                100.0*t_setDarcyTopPressure/t_sum << " %)\n"
                << "  - findDarcyPermeabilities:\t" <<
                t_findDarcyPermeabilities/1000.0 << " s" << "\t(" <<
                100.0*t_findDarcyPermeabilities/t_sum << " %)\n"
                << "  - findDarcyPermeabilityGrads:\t" <<
                t_findDarcyPermeabilityGradients/1000.0 << " s" << "\t(" <<
                100.0*t_findDarcyPermeabilityGradients/t_sum << " %)\n"
                //<< "  - findDarcyPressureChange:\t" <<
                //t_findDarcyPressureChange/1000.0 << " s" << "\t(" <<
                //100.0*t_findDarcyPressureChange/t_sum << " %)\n"
                << "  - updateDarcySolution:\t" <<
                t_updateDarcySolution/1000.0 << " s" << "\t(" <<
                100.0*t_updateDarcySolution/t_sum << " %)\n"
                << "  - copyValues:\t\t\t" <<
                t_copyValues/1000.0 << " s" << "\t(" <<
                100.0*t_copyValues/t_sum << " %)\n"
                << "  - findDarcyVelocities:\t" <<
                t_findDarcyVelocities/1000.0 << " s" << "\t(" <<
                100.0*t_findDarcyVelocities/t_sum << " %)" << std::endl;
        }
    }

    // Free GPU device memory
    freeGlobalDeviceMemory();
    checkForCudaErrorsIter("After freeGlobalDeviceMemory()", iter);

    // Free contact info arrays
    delete[] k.contacts;
    delete[] k.distmod;
    delete[] k.delta_t;

    if (fluid == 1) {
        if (cfd_solver == 0)
            endNS();
        else if (cfd_solver == 1)
            endDarcy();
    }

    cudaDeviceReset();
}
// vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
