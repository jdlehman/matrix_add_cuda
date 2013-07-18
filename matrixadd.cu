/* matrixadd.cu
 * 
 *
 * Jonathan Lehman
 * February 12, 2012
 *
 * Homework Assignment 3
 *
 * This program uses a CUDA capable GPU to add two randomly generated matrices in parallel.  The matrix dimensions
 * are specified by the user as an argument, as are the grid and block dimensions to be used on the GPU.
 * This program outputs the time it takes to do the matrix addition (not including the time it takes to transfer data
 * from the host to the device or back.  It also checks arguments to ensure that it will give correct error messages for 
 * any invalid input or data sizes that the GPU cannot handle.
 * 
 */

#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

__global__
void add( float*, float*, float*, int, int );
void buildArrays( void );
void checkArgs(int, char**);
void checkGPUCapabilities(int, int, int, int, int);
double getTime();

//user input
int GRID_WIDTH;
int GRID_HEIGHT;
int BLOCK_WIDTH;
int BLOCK_HEIGHT;
int MATRIX_HEIGHT;
int MATRIX_WIDTH;

int NUM_VALUES;

// Keep track of the time.
double startTime, stopTime;


//arrays
float* a;
float* b;
float* c;

int main( int argc, char *argv[] ){
  	
	float *dev_a, *dev_b, *dev_c;
	
	//check validity of arguments
	checkArgs(argc, argv);
  
	//assign variables
	GRID_WIDTH = atoi(argv[1]);
	GRID_HEIGHT = atoi(argv[2]);
	BLOCK_WIDTH = atoi(argv[3]);
	BLOCK_HEIGHT = atoi(argv[4]);
	MATRIX_HEIGHT = atoi(argv[5]);
	MATRIX_WIDTH = atoi(argv[6]);
	
	NUM_VALUES = MATRIX_WIDTH * MATRIX_HEIGHT;
	
	//check that GPU can handle arguments
	checkGPUCapabilities(GRID_WIDTH, GRID_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT, NUM_VALUES);
  
	/* Initialize the source arrays here. */
  	a = new float[NUM_VALUES];
  	b = new float[NUM_VALUES];
  	c = new float[NUM_VALUES];
  
  	//fill array a and b with random doubles
  	buildArrays();
  	
  	//TEST print array values
  	/*printf( "The a array:\n" );
  	for( int i = 0; i < NUM_VALUES; i++ ) 
  		printf( "%f\n", a[i] );
    	printf( "\n" );
    	
    	printf( "The b array:\n" );
  	for( int i = 0; i < NUM_VALUES; i++ ) 
  		printf( "%f\n", b[i] );
    	printf( "\n" );*/
    	
    	//check if there will be enough blocks to handle matrix size (if not some threads will take on more than one addition)
    	int reps = ceil((double)(MATRIX_WIDTH * MATRIX_HEIGHT) / (BLOCK_WIDTH * BLOCK_HEIGHT));
   
  	
  	/* Allocate global device memory. */
  	cudaMalloc( (void **)&dev_a, sizeof(float) * NUM_VALUES );
  	cudaMalloc( (void **)&dev_b, sizeof(float) * NUM_VALUES );
  	cudaMalloc( (void **)&dev_c, sizeof(float) * NUM_VALUES );
  
  	/* Copy the host values to global device memory. */
  	cudaMemcpy( dev_a, a, sizeof(float) * NUM_VALUES, cudaMemcpyHostToDevice );
  	cudaMemcpy( dev_b, b, sizeof(float) * NUM_VALUES, cudaMemcpyHostToDevice);
  	
  	/* Start the timer. */
  	startTime = getTime();
  
  	/* Execute the kernel. */
  	dim3 grid(GRID_WIDTH, GRID_HEIGHT); //blocks w x h
  	dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT); //threads w x h
  	add<<<grid, block>>>(dev_a, dev_b, dev_c, NUM_VALUES, reps);

  	/* Wait for the kernel to complete. Needed for timing. */  
  	cudaThreadSynchronize();
  	
  	/* Stop the timer and print the resulting time. */
	  stopTime = getTime();
	  double totalTime = stopTime - startTime;
	  printf( "Time: %f secs\n", totalTime );
  
  	/* Get result from device. */
  	cudaMemcpy(c, dev_c, sizeof(float) * NUM_VALUES, cudaMemcpyDeviceToHost);
  	
  	/*printf( "The c array:\n" );
  	for( int i = 0; i < NUM_VALUES; i++ ) 
  		printf( "%f\n", c[i] );
    	printf( "\n" );*/
  
    	
  	/* Free the allocated device memory. */
  	cudaFree(dev_a);
  	cudaFree(dev_b);
  	cudaFree(dev_c);
  
  	//free allocated host memory
	free(a);
	free(b);
	free(c);
}

__global__
void add( float *a, float *b, float *c, int size , int reps)
{
	//grid dimensions (# blocks)
	int gridW = gridDim.x;   
	int gridH = gridDim.y;
	
	//block dimensions (# threads)
	int blockW = blockDim.x;
        int blockH = blockDim.y;
	
        //block id
	int blockX = blockIdx.x;
	int blockY = blockIdx.y;  
	
	//thread id
	int threadX = threadIdx.x;
	int threadY = threadIdx.y;
       
        
        //loop through number of times matrix elements fill more than an entire grid
	for(int i = 0; i < reps; i++){
		int colNum = threadX + blockW * blockX; 
		int rowNum = threadY + blockH * blockY;
	
		int index = rowNum * blockW + colNum + (gridW * gridH * blockW * blockH * i);
	
		//check that element index is within the arrays being added
		if(index < size){
		  c[index] = a[index]+b[index];
		}
        }        
}

void buildArrays( void ){
	/* Seed the random number generator. */
	srand( 200 );

	for(int i = 0; i < NUM_VALUES; i++){
		float val = rand() / (float(RAND_MAX));
		a[i] = val;
	}
  
	srand( 300 );
  
	for(int i = 0; i < NUM_VALUES; i++){ 
  		float val = rand() / (float(RAND_MAX));
  		b[i] = val;
  	}


}

void checkArgs(int argc, char *argv[]){
	
	//check number of arguments
	if(argc != 7){
		fprintf(stderr, "\nmatrixadd: Incorrect number of arguments. matrixadd requires 6 arguments not %d\nCorrect usage: \"matrixadd grid-width grid-height blck-width blck-height mat-height mat-width\"\n", argc - 1);
		exit(1);
	}
	
	
	char* invalChar;
	long arg;
	
	//check each argument
	for(int i = 1; i < 7; i++){
		//check for overflow of argument
		if((arg = strtol(argv[i], &invalChar, 10)) >= INT_MAX){
			fprintf(stderr, "\nmatrixadd: Overflow. Invalid argument %d for matrixadd, '%s'.\nThe argument must be a valid, positive, non-zero integer less than %d.\n", i, argv[i], INT_MAX);
			exit(1);
		}
	
		//check that argument is a valid positive integer and check underflow
		if(!(arg > 0) || (*invalChar)){
			fprintf(stderr, "\nmatrixadd: Invalid argument %d for matrixadd, '%s'.  The argument must be a valid, positive, non-zero integer.\n", i, argv[i]);
			exit(1);
		}
		
	}	
}

void checkGPUCapabilities(int gridW, int gridH, int blockW, int blockH, int size){
	//check what GPU is being used
	int devId;  
	cudaGetDevice( &devId );
	
	//get device properties for GPU being used
	cudaDeviceProp gpuProp;
	cudaGetDeviceProperties( &gpuProp, devId );
	
	//check if GPU has enough memory to handle the 3 arrays
	if(gpuProp.totalGlobalMem < (size * sizeof(float)) * 3){
		fprintf(stderr, "\nmatrixadd: Insufficient GPU. GPU does not have enough memory to handle the data size: %ld. It can only handle data sizes up to %ld.\n", (size * sizeof(float)) * 3, gpuProp.totalGlobalMem);
		exit(1);
	}
	
	//check if GPU can handle the number of threads per bloc
	if(gpuProp.maxThreadsPerBlock < (blockW * blockH)){
		fprintf(stderr, "\nmatrixadd: Insufficient GPU. GPU can only handle %d threads per block, not %d.\n", gpuProp.maxThreadsPerBlock, (blockW * blockH));
		exit(1);
	}
	
	//check that GPU can handle the number of threads in the block width
	if(gpuProp.maxThreadsDim[0] < blockW){
		fprintf(stderr, "\nmatrixadd: Insufficient GPU. GPU can only handle %d threads as the block width of each block, not %d.\n", gpuProp.maxThreadsDim[0], blockW );
		exit(1);
	}
	
	//check that GPU can handle the number of threads in the block height
	if(gpuProp.maxThreadsDim[1] < blockH){
		fprintf(stderr, "\nmatrixadd: Insufficient GPU. GPU can only handle %d threads as the block height of each block, not %d.\n", gpuProp.maxThreadsDim[1], blockH );
		exit(1);
	}
	
	//check that GPU can handle the number of blocks in the grid width
	if(gpuProp.maxGridSize[0] < gridW){
		fprintf(stderr, "\nmatrixadd: Insufficient GPU. GPU can only handle %d blocks as the grid width of each grid, not %d.\n", gpuProp.maxGridSize[0], gridW );
		exit(1);
	}
	
	//check that GPU can handle the number of blocks in the grid height
	if(gpuProp.maxGridSize[1] < gridH){
		fprintf(stderr, "\nmatrixadd: Insufficient GPU. GPU can only handle %d blocks as the grid height of each grid, not %d.\n", gpuProp.maxGridSize[1], gridH );
		exit(1);
	}
}

double getTime(){
  timeval thetime;
  gettimeofday( &thetime, 0 );
  return thetime.tv_sec + thetime.tv_usec / 1000000.0;
}
