
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <time.h>

#define COORD_MAX 100000
#define CLUSTER_NUM 20
#define POINT_NUM 1000000
#define POINT_FEATURES 3
#define CLUSTER_FEATURES 4
#define THREAD_PER_BLOCK 1024


#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

#else
static __inline__ __device__ double atomicAdd(double* address, double val) {
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	if (val == 0.0)
		return __longlong_as_double(old);
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}


#endif

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


double random_double() {
	double x;
	x = (double)rand() * (double)32767;
	x = fmod(x, COORD_MAX);
	return x;
}

void print_points(double* punti) {
	printf("----------- Punti -------------\n");
	for (int i = 0; i < POINT_NUM; i++) {
		double x = punti[i * POINT_FEATURES + 0];
		double y = punti[i * POINT_FEATURES + 1];
		double cluster =punti[i * POINT_FEATURES + 2];
		printf("punto %d, x:%f y:%f, c:%f\n", i, x, y, cluster);
	}
	printf("-------------------------------\n");
}
void print_clusters(double* clusters) {
	printf("----------- Cluster -------------\n");
	for (int i = 0; i < CLUSTER_NUM; i++) {
		double centro = clusters[i * CLUSTER_FEATURES + 0];
		double sizex = clusters[i * CLUSTER_FEATURES + 1];
		double sizey = clusters[i * CLUSTER_FEATURES + 2];
		double numPoints = clusters[i * CLUSTER_FEATURES + 3];
		printf("cluster %d, centro: %f, sizex; %f, sizey:%f, numpoints: %f\n", i, centro, sizex, sizey, numPoints);
	}
	printf("-------------------------------\n");
}
void init_all(double* punti, double* clusters) {
	for (int i = 0; i < POINT_NUM; i++) {			//punto: <x,y,cluster>
		punti[i*  POINT_FEATURES+0] = random_double();
		punti[i * POINT_FEATURES + 1] = random_double();
		punti[i * POINT_FEATURES + 2] = 0;
	}

	for (int i = 0; i < CLUSTER_NUM;i++) {			//cluster: <centro,size_x,size_y,punti>
		clusters[i * CLUSTER_FEATURES + 0] = rand() % POINT_NUM;
		clusters[i * CLUSTER_FEATURES + 1] = 0;
		clusters[i * CLUSTER_FEATURES + 2] = 0;
		clusters[i * CLUSTER_FEATURES + 3] = 0;
	}
}
void write_to_file(double *punti) {
	printf("\nStampo su file!\n");
	FILE* fPtr;
	char filePath[100] = { "G:\\file.dat" };
	char dataToAppend[1000];
	fPtr = fopen(filePath, "a");
	for (int i = 0; i < POINT_NUM; i++) {
		double x = punti[i * POINT_FEATURES + 0];
		double y = punti[i * POINT_FEATURES + 1];
		int cluster = punti[i * POINT_FEATURES + 2];
		fprintf(fPtr, "%f %f %d\n", x,y,cluster);
	}
}
__device__ double distance(double x1, double x2, double y1, double y2) {
	return sqrt(((x1 - x2) * (x1 - x2)) + ((y1 - y2) * (y1 - y2)));
}



__global__ void assign_clusters(double* punti, double* clusters) {

	long id_punto = threadIdx.x + blockIdx.x * blockDim.x;
	if (id_punto < POINT_NUM) {
		double x_punto, x_cluster, y_punto, y_cluster = 0;
		x_punto = punti[id_punto * POINT_FEATURES + 0];
		y_punto = punti[id_punto * POINT_FEATURES + 1];
		long best_fit = 0;
		long distMax = LONG_MAX;
		for (int i = 0; i < CLUSTER_NUM; i++) {
			int cluster_index_point = clusters[i * CLUSTER_FEATURES + 0];
			x_cluster = punti[cluster_index_point * POINT_FEATURES + 0];
			y_cluster = punti[cluster_index_point * POINT_FEATURES + 1];
			if (distance(x_punto, x_cluster, y_punto, y_cluster) < distMax) {
				best_fit = i;
				distMax = distance(x_punto, x_cluster, y_punto, y_cluster);
			}
		}

		punti[id_punto * POINT_FEATURES + 2] = best_fit;
		atomicAdd(&clusters[best_fit * CLUSTER_FEATURES + 1], x_punto);
		atomicAdd(&clusters[best_fit * CLUSTER_FEATURES + 2], y_punto);
		atomicAdd(&clusters[best_fit * CLUSTER_FEATURES + 3],1);

	    //printf("id %d: Assegno il punto %f,%f al cluster %f\n",id_punto, x_punto, y_punto, punti[id_punto * POINT_FEATURES + 2]);
	}
	
}


bool clusters_recomputeCenter(double *points, double *clusters) { //cluster <punto,sizeX,sizeY,n_points>
	int acc = 0;
	for (int i = 0; i < CLUSTER_NUM; i++) {
		double sizeX = clusters[i * CLUSTER_FEATURES + 1];
		double sizeY = clusters[i * CLUSTER_FEATURES + 2];
		double nPoints = clusters[i * CLUSTER_FEATURES + 3];
		double newX = sizeX / nPoints;
		double newY = sizeY / nPoints;
		long cluster_center_index = (long)clusters[i* CLUSTER_FEATURES+0];
		double x = points[cluster_center_index * POINT_FEATURES +0];
		double y = points[cluster_center_index* POINT_FEATURES +1];
		if (x == newX && y == newY) {
			acc = acc;
		}
		else {
			points[cluster_center_index *POINT_FEATURES + 0] = newX;
			points[cluster_center_index* POINT_FEATURES +1] = newY;
			acc++;
		}
	}
	if (acc == 0) {
		return false;
	}
	else {
		return true;
	}
}

int main()
{
	double* punti = (double*)malloc(POINT_NUM * POINT_FEATURES* sizeof(double));
	double* clusters = (double*)malloc(CLUSTER_NUM * CLUSTER_FEATURES * sizeof(double));

	init_all(punti, clusters);

	double* punti_d = 0;
	double* cluster_d = 0;

	int iterazioni = 10;
	clock_t begin = clock();

	cudaMalloc(&punti_d, POINT_NUM * POINT_FEATURES * sizeof(double));
	cudaMalloc(&cluster_d, CLUSTER_NUM * CLUSTER_FEATURES * sizeof(double));

	for (int i = 0; i < iterazioni; i++) {
		
		cudaMemcpy(punti_d, punti, POINT_NUM * POINT_FEATURES * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(cluster_d, clusters, CLUSTER_NUM * CLUSTER_FEATURES * sizeof(double), cudaMemcpyHostToDevice);

		assign_clusters << < (POINT_NUM + THREAD_PER_BLOCK -1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK >> > (punti_d, cluster_d);

		cudaMemcpy(punti, punti_d, POINT_NUM * POINT_FEATURES * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(clusters, cluster_d, CLUSTER_NUM * CLUSTER_FEATURES * sizeof(double), cudaMemcpyDeviceToHost);

		//clusters_recomputeCenter(punti, clusters);
	}
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("Tempo %f", time_spent);
	//print_points(punti);
	//print_clusters(clusters);
	write_to_file(punti);
	
}

