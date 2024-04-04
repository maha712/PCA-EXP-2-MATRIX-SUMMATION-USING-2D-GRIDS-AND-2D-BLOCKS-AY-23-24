# PCA-EXP-2-Matrix-Summation-using-2D-Grids-and-2D-Blocks-AY-23-24


MATRIX SUMMATION WITH A 2D GRID AND 2D BLOCKS 
i.  Use the file sumMatrixOnGPU-2D-grid-2D-block.cu
ii. Matrix summation with a 2D grid and 2D blocks. Adapt it to integer matrix addition. Find the best execution configuration. </h3>

## AIM:
To perform  matrix summation with a 2D grid and 2D blocks and adapting it to integer matrix addition.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler




## PROCEDURE:

1.	Initialize the data: Generate random data for two input arrays using the initialData function.
2.	Perform the sum on the host: Use the sumMatrixOnHost function to calculate the sum of the two input arrays on the host (CPU) for later verification of the GPU results.
3.	Allocate memory on the device: Allocate memory on the GPU for the two input arrays and the output array using cudaMalloc.
4.	Transfer data from the host to the device: Copy the input arrays from the host to the device using cudaMemcpy.
5.	Set up the execution configuration: Define the size of the grid and blocks. Each block contains multiple threads, and the grid contains multiple blocks. The total number of threads is equal to the size of the grid times the size of the block.
6.	Perform the sum on the device: Launch the sumMatrixOnGPU2D kernel on the GPU. This kernel function calculates the sum of the two input arrays on the device (GPU).
7.	Synchronize the device: Use cudaDeviceSynchronize to ensure that the device has finished all tasks before proceeding.
8.	Transfer data from the device to the host: Copy the output array from the device back to the host using cudaMemcpy.
9.	Check the results: Use the checkResult function to verify that the output array calculated on the GPU matches the output array calculated on the host.
10.	Free the device memory: Deallocate the memory that was previously allocated on the GPU using cudaFree.
11.	Free the host memory: Deallocate the memory that was previously allocated on the host.
12.	Reset the device: Reset the device using cudaDeviceReset to ensure that all resources are cleaned up before the program exits.

## PROGRAM:

#include "common.h"

#include <cuda_runtime.h>

#include <stdio.h>

void initialData(int *ip, const int size)

{

    int i;

    for(i = 0; i < size; i++)
    
    {
    
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
        
    }

    return;
    
}

void sumMatrixOnHost(int*A, int*B, int*C, const int nx,

                     const int ny)
                     
{

    int*ia = A;
    
    int*ib = B;
    
    int*ic = C;

    for (int iy = 0; iy < ny; iy++)
    
    {
    
        for (int ix = 0; ix < nx; ix++)
        
        {
        
            ic[ix] = ia[ix] + ib[ix];

        }

        ia += nx;
        
        ib += nx;
        
        ic += nx;
    }

    return;
    
}


void checkResult(int*hostRef, int*gpuRef, const int N)

{

    double epsilon = 1.0E-8;
    
    bool match = 1;

    for (int i = 0; i < N; i++)
    
    {
    
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        
        {
        
            match = 0;
            
            printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
            
            break;
            
        }
        
    }

    if (match)
    
        printf("Arrays match.\n\n");
        
    else
    
        printf("Arrays do not match.\n\n");
        
}

// grid 2D block 2D

__global__ void sumMatrixOnGPU2D(int*MatA, int*MatB, int*MatC, int nx,int ny)

{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
    
        MatC[idx] = MatA[idx] + MatB[idx];
        
}

int main(int argc, char **argv)

{

    printf("%s Starting...\n", argv[0]);
    

    // set up device
    
    int dev = 0;
    
    cudaDeviceProp deviceProp;
    
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    
    CHECK(cudaSetDevice(dev));

    // set up data size of matrix
    
    int nx = 1 << 14;
    
    int ny = 1 << 14;

    int nxy = nx * ny;
    
    int nBytes = nxy * sizeof(float);
    
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    
    int*h_A, *h_B, *hostRef, *gpuRef;
    
    h_A = (int*)malloc(nBytes);
    
    h_B = (int*)malloc(nBytes);
    
    hostRef = (int*)malloc(nBytes);
    
    gpuRef = (int*)malloc(nBytes);

    // initialize data at host side
    
    double iStart = seconds();
    
    initialData(h_A, nxy);
    
    initialData(h_B, nxy);
    
    double iElaps = seconds() - iStart;
    
    printf("Matrix initialization elapsed %f sec\n", iElaps);

    memset(hostRef, 0, nBytes);
    
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result checks
    
    iStart = seconds();
    
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    
    iElaps = seconds() - iStart;
    
    printf("sumMatrixOnHost elapsed %f sec\n", iElaps);

    // malloc device global memory
    
    int*d_MatA, *d_MatB, *d_MatC;
    
    CHECK(cudaMalloc((void **)&d_MatA, nBytes));
    
    CHECK(cudaMalloc((void **)&d_MatB, nBytes));
    
    CHECK(cudaMalloc((void **)&d_MatC, nBytes));

    // transfer data from host to device
    
    CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
    
    CHECK(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

    // invoke kernel at host side
    
    int dimx = 32;
    
    int dimy = 32;
    
    dim3 block(dimx, dimy);
    
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    iStart = seconds();
    
    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    
    CHECK(cudaDeviceSynchronize());
    
    iElaps = seconds() - iStart;
    
    printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", grid.x,
    
           grid.y,
           
           block.x, block.y, iElaps);
           
    // check kernel error
    
    CHECK(cudaGetLastError());

    // copy kernel result back to host side
    
    CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));

    // check device results
    
    checkResult(hostRef, gpuRef, nxy);

    // free device global memory
    
    CHECK(cudaFree(d_MatA));
    
    CHECK(cudaFree(d_MatB));
    
    CHECK(cudaFree(d_MatC));

    // free host memory
    
    free(h_A);
    
    free(h_B);
    
    free(hostRef);
    
    free(gpuRef);

    // reset device
    
    CHECK(cudaDeviceReset());

    return (0);
}

## OUTPUT:

![PCA 21](https://github.com/maha712/PCA-EXP-2-MATRIX-SUMMATION-USING-2D-GRIDS-AND-2D-BLOCKS-AY-23-24/assets/121156360/a5f9a5b4-b5e7-4db7-952f-8814e159ca98)


## RESULT:
Thus, matrix summation using 2D grids and 2D blocks has been performed successfully.
