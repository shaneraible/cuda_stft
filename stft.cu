#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <cufft.h>
#include <time.h>

#define MAX_SIGNAL_SIZE 1202500
#define PI 3.141592654
// one side of the signal
double signal[MAX_SIGNAL_SIZE];
int signal_len = 0;

__global__ void apply_window(double *signal, int signal_len, int window_len, int hop, cufftComplex *complexSignal){
    // window size is 1024
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;

    int iStart = block_id*hop;
    if(iStart+window_len>=signal_len || thread_id>=window_len || block_id>=4036) return;


    double window_multiplier = 0.5 * (1 - cos(2*PI*thread_id/(window_len-1)));

    // windowed[i] = window_multiplier * signal[iStart + thread_id];
    if(block_id>4034 ) signal[thread_id] = iStart;
    // else signal[thread_id] = 0

    complexSignal[i].x = window_multiplier * signal[iStart + thread_id];
    complexSignal[i].y = 0;

    //iterate over block
}

__global__ void apply_log(cufftComplex *complexSignal){
    // window size is 1024
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;

    int iStart = block_id*hop;
    if(iStart+window_len>=signal_len || thread_id>=window_len || block_id>=4036) return;


    double window_multiplier = 0.5 * (1 - cos(2*PI*thread_id/(window_len-1)));

    // windowed[i] = window_multiplier * signal[iStart + thread_id];
    if(block_id>4034 ) signal[thread_id] = iStart;
    // else signal[thread_id] = 0

    complexSignal[i].x = window_multiplier * signal[iStart + thread_id];
    complexSignal[i].y = 0;

    //iterate over block
}

// __global__ void calc_fft_window(){
//     // setup the fft plan
//     cufftPlan1d(&plan, signal_len, CUFFT_C2C, 1);

//     cufftExecC2C(plan, gpu_complex_signal, gpu_complex_signal, CUFFT_FORWARD);

// }

int main( int argc, char **argv )
{    
    std::ifstream ifile("gliss.ascii", std::ios::in);

    //check to see that the file was opened correctly:
    if (!ifile.is_open()) {
        std::cerr << "There was a problem opening the input file!\n";
        exit(1);//exit or do additional error checking
    }

    double num = 0.0;
    //keep storing values from the text file so long as data exists:
    int count_total = 0;
    while (ifile >> num) {
        if(count_total++%2 ==0) signal[signal_len++] = num;
    }

    std::cout<<"Signal Length: "<<signal_len<<std::endl;
    
    int nFFT = 1024;
    int hop = floor(nFFT/4);
    int nFrames = floor(signal_len/hop);

    std::cout<<nFrames<<" frames, "<<std::endl;
    std::cout<<hop<<" hop"<<std::endl;

    //configure cuFFT
    cufftHandle plan;

    // begin parallel execution section
    // This takes a few seconds to initialize the runtime
    cudaThreadSynchronize(); 
    clock_t start = clock();


    double *gpu_signal;
    cufftComplex *complex_signal_windowed = (cufftComplex *)malloc(nFrames*nFFT*sizeof(cufftComplex));

    cufftComplex *gpu_complex_signal_windowed;

    double *gpu_fft_result;
    double *gpu_db_result;

    double *gpu_window;

    // allocate gpu mem
    cudaMalloc((void**) &gpu_signal, signal_len*sizeof(double));
    // cudaMalloc((void**) &gpu_wind_signal, nFrames*nFFT*sizeof(double));
    cudaMalloc((void**) &gpu_fft_result, nFrames*nFFT*sizeof(double));
    cudaMalloc((void**) &gpu_db_result, (int)nFrames/2*nFFT*sizeof(double));
    cudaMalloc((void**) &gpu_window, nFFT*sizeof(double));

    cudaMalloc((void **)&gpu_complex_signal_windowed, nFrames*nFFT*sizeof(cufftComplex));

    cudaMemcpy(gpu_signal, signal, signal_len * sizeof(double), cudaMemcpyHostToDevice);

    // setup the fft plan
    cufftPlan1d(&plan, signal_len, CUFFT_C2C, 1);

    // appply hanning window
    apply_window <<< nFrames, nFFT >>> (gpu_signal, signal_len, nFFT, hop, gpu_complex_signal_windowed);
    
    
    // compute FFT in blocks
    int rank=1;
    int batch=nFrames; 
    int size_per=nFFT;

    // cufftExecC2C(plan, gpu_complex_signal, gpu_complex_signal, CUFFT_FORWARD);
    // cufftPlanMany(&plan, rank, size_per, NULL, hop, idist,
    //     NULL, hop, odist, CUFFT_C2C, batch);

    cufftPlan1d(&plan, size_per, CUFFT_C2C, batch);

    cufftExecC2C(plan, gpu_complex_signal_windowed, gpu_complex_signal_windowed, CUFFT_FORWARD);

    cudaThreadSynchronize();
    clock_t end = clock();
    double elapsed = double(end - start)/CLOCKS_PER_SEC;



    // copy the fft  result back to the host 
    cudaMemcpy(complex_signal_windowed, gpu_complex_signal_windowed, nFrames*nFFT*sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    cudaMemcpy(signal, gpu_signal, signal_len*sizeof(double), cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    std::cout<<"Time elapsed: "<<elapsed<<std::endl;

    for(int i=0; i<1024; i++){
        std::string mid = complex_signal_windowed[i].y>=0? "\t+ " :"\t- ";

        std::cout<<i+1<<":\t"<<complex_signal_windowed[i].x<< mid << std::fabs(complex_signal_windowed[i].y)<<"j"<<std::endl;
    }    
    // for(int i=0; i<1024; i++){
    //     std::cout<<signal[i]<<std::endl;
    // }

    return 0;
}