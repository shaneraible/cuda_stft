# Load CUDA using the following command
# module load cuda
#
CC = nvcc
CFLAGS = -O3 -arch=compute_35 -code=sm_35 -gencode arch=compute_35,code=[compute_35,sm_35] -gencode arch=compute_61,code=[compute_61,sm_61]
NVCCFLAGS = -O3 -arch=compute_35 -code=sm_35 -gencode arch=compute_35,code=[compute_35,sm_35] -gencode arch=compute_61,code=[compute_61,sm_61]
NVCCLIBS = -lcufft

TARGETS = stft

all:	$(TARGETS)

stft: stft.o
	$(CC) -o $@ $(NVCCLIBS) stft.o 

stft.o: stft.cu
	$(CC) -c $(NVCCFLAGS) stft.cu


clean:
	rm -f *.o $(TARGETS) *.stdout *.txt
