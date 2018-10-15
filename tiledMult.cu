#include "common.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

using namespace std;

#define TILEDIM 8


//Code used from examples and modified for activity
//Adrian Biller A01018940
//matrix multiplication with 2D 2D using tiled method





//inicialization of matrices
void initialData(int *ip, const int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        ip[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/10));
    }

    return;
}

//printing arrays
void printArray(int * arr, int size)
{
  int totalSize = size * size;
  int row = 1;
  for(int x = 0; x < totalSize; x++){
    printf("%d ", arr[x]);
    if((size * row)-1 == x){
      row++;
      printf("\n");
    }
  }
}


//multiplication of matrices using cpu
void multiplyMatrixOnHost(int *A, int *B, int *C, const int nx,
                     const int ny)
{
      for(int i = 0; i < nx; i++){
        for(int j = 0; j < nx ; j++){
          for(int k = 0; k < nx; k++){
            C[i*nx+j] += A[i*nx+k] * B[k*nx+j];
          }
        }
      }

    return;
}

//checking result of gpu and comparing them with cpu matrix
void checkResult(int *hostRef, int *gpuRef, const int N)
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




//matrix calculation using cpu
__global__ void multMatrixOnGPU2D(int *MatA, int *MatB, int *MatC, int nx,
    int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;


    if (ix < nx && iy < ny){
        for(int k = 0; k < nx; k++){
          MatC[ix * nx + iy] += MatA[ix * nx + k] * MatB[k * nx + iy];
        }
    }
}

//matrix calculation using tile method
__global__ void tiledMult(int *MatA, int *MatB, int *MatC, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    // unsigned int idx = iy * nx + ix;

    __shared__ float sharedMatA[TILEDIM][TILEDIM];
    __shared__ float sharedMatB[TILEDIM][TILEDIM];

    printf("%d\n", blockDim.x);
    printf("%d\n", TILEDIM);
    // for(int i = 0; i < DIM; i ++) {
    //   for(int j = 0; j < DIM; j++) {
    //     sharedMatA[i][j] = 0;
    //     sharedMatB[i][j] = 0;
    //   }
    // }
}



int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev), "Error device prop");
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    SAFE_CALL(cudaSetDevice(dev), "Error setting device");

    // set up data size of matrix
    // int nx = 1 << 12;
    // int ny = 1 << 12;
    int nx = 4000;
    int ny = 4000;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(int);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    int *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (int *)malloc(nBytes);
    h_B = (int *)malloc(nBytes);
    hostRef = (int *)malloc(nBytes);
    gpuRef = (int *)malloc(nBytes);

    // initialize data at host side

    initialData(h_A, nxy);
    initialData(h_B, nxy);
    // printArray(h_A, nx);
    // printf("\n");
    // printArray(h_B, nx);
    // printf("\n");

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result SAFE_CALLs
    auto start_cpu =  chrono::high_resolution_clock::now();
    multiplyMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    auto end_cpu =  chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;

    printf("multiplyMatrixOnHost elapsed %f ms\n", duration_ms.count());

    // malloc device global memory
    int *d_MatA, *d_MatB, *d_MatC;
    SAFE_CALL(cudaMalloc((void **)&d_MatA, nBytes), "Error allocating d_MatA");
    SAFE_CALL(cudaMalloc((void **)&d_MatB, nBytes), "Error allocating d_MatB");
    SAFE_CALL(cudaMalloc((void **)&d_MatC, nBytes), "Error allocating d_MatC");

    // transfer data from host to device
    SAFE_CALL(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatA");
    SAFE_CALL(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatB");
    SAFE_CALL(cudaMemset(d_MatC, 0, nBytes), "Error setting d_MatC to zeros");

    // invoke kernel at host side
    int dimx = TILEDIM;
    int dimy = TILEDIM;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // start_cpu =  chrono::high_resolution_clock::now();
    // multMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    // SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
    // end_cpu =  chrono::high_resolution_clock::now();
    //
    // duration_ms = end_cpu - start_cpu;
    //
    // printf("sumMatrixOnGPU1D <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n", grid.x,
    //        grid.y,
    //        block.x, block.y, duration_ms.count());
    //
    // // SAFE_CALL kernel error
    // SAFE_CALL(cudaGetLastError(), "Error with last error");
    //
    // // copy kernel result back to host side
    // SAFE_CALL(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC");

    // printArray(hostRef, nx);
    // printf("Host\n");
    // printArray(gpuRef, nx);
    // printf("GPU\n");
    // // check device results
    // checkResult(hostRef, gpuRef, nxy);


    // dimx = TILEDIM;
    // dimy = TILEDIM;
    // dim3 block(dimx, dimy);
    // dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    //calculating matrix multiplication using Tiling
    start_cpu =  chrono::high_resolution_clock::now();
    tiledMult<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
    end_cpu =  chrono::high_resolution_clock::now();

    duration_ms = end_cpu - start_cpu;

    printf("Matrix multiplication with tiling <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n", grid.x,
           grid.y,
           block.x, block.y, duration_ms.count());


    // SAFE_CALL kernel error
    SAFE_CALL(cudaGetLastError(), "Error with last error");

    // copy kernel result back to host side
    SAFE_CALL(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC");


    checkResult(hostRef, gpuRef, nxy);




    // free device global memory
    SAFE_CALL(cudaFree(d_MatA), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatB), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatC), "Error freeing memory");

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // reset device
    SAFE_CALL(cudaDeviceReset(), "Error reseting");

    return (0);
}
