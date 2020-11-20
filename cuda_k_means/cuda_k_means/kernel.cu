
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <time.h>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>


#define COORD_MAX 100000
#define CLUSTER_NUM 20
#define POINT_NUM 10000000
#define POINT_FEATURES 3
#define CLUSTER_FEATURES 4
#define THREAD_PER_BLOCK 1024

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


float random_float() {
	float x;
	x = (float)rand() * (float)32767;
	x = fmod(x, COORD_MAX);
	return x;
}

void print_points(float* punti) {
	printf("----------- Punti -------------\n");
	for (int i = 0; i < POINT_NUM; i++) {
		float x = punti[i * POINT_FEATURES + 0];
		float y = punti[i * POINT_FEATURES + 1];
		float cluster = punti[i * POINT_FEATURES + 2];
		printf("punto %d, x:%f y:%f, c:%f\n", i, x, y, cluster);
	}
	printf("-------------------------------\n");
}
void print_clusters(float* clusters) {
	printf("----------- Cluster -------------\n");
	for (int i = 0; i < CLUSTER_NUM; i++) {
		float centro = clusters[i * CLUSTER_FEATURES + 0];
		float sizex = clusters[i * CLUSTER_FEATURES + 1];
		float sizey = clusters[i * CLUSTER_FEATURES + 2];
		float numPoints = clusters[i * CLUSTER_FEATURES + 3];
		printf("cluster %d, centro: %f, sizex; %f, sizey:%f, numpoints: %f\n", i, centro, sizex, sizey, numPoints);
	}
	printf("-------------------------------\n");
}
void init_all(float* punti, float* clusters) {
	for (int i = 0; i < POINT_NUM; i++) {			//punto: <x,y,cluster>
		punti[i * POINT_FEATURES + 0] = random_float();
		punti[i * POINT_FEATURES + 1] = random_float();
		punti[i * POINT_FEATURES + 2] = 0;
	}

	for (int i = 0; i < CLUSTER_NUM; i++) {			//cluster: <centro,size_x,size_y,punti>
		clusters[i * CLUSTER_FEATURES + 0] = rand() % POINT_NUM;
		clusters[i * CLUSTER_FEATURES + 1] = 0;
		clusters[i * CLUSTER_FEATURES + 2] = 0;
		clusters[i * CLUSTER_FEATURES + 3] = 0;
	}
}
void write_to_file(float* punti) {
	printf("\nStampo su file!\n");
	FILE* fPtr;
	char filePath[100] = { "G:\\file.dat" };
	char dataToAppend[1000];
	fPtr = fopen(filePath, "a");
	for (int i = 0; i < POINT_NUM; i++) {
		float x = punti[i * POINT_FEATURES + 0];
		float y = punti[i * POINT_FEATURES + 1];
		int cluster = punti[i * POINT_FEATURES + 2];
		fprintf(fPtr, "%f %f %d\n", x, y, cluster);
	}
}
__device__ float distance(float x1, float x2, float y1, float y2) {
	return sqrt(((x1 - x2) * (x1 - x2)) + ((y1 - y2) * (y1 - y2)));
}



__global__ void assign_clusters(float* punti, float* clusters) {

	long id_punto = threadIdx.x + blockIdx.x * blockDim.x;
	if (id_punto < POINT_NUM) {
		float x_punto, x_cluster, y_punto, y_cluster = 0;
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
		//clusters[best_fit * CLUSTER_FEATURES + 1] = clusters[best_fit * CLUSTER_FEATURES + 1] + x_punto;
		//clusters[best_fit * CLUSTER_FEATURES + 2] = clusters[best_fit * CLUSTER_FEATURES + 2] + y_punto;
		//clusters[best_fit * CLUSTER_FEATURES + 3] = clusters[best_fit * CLUSTER_FEATURES + 3] + 1;
		atomicAdd(&clusters[best_fit * CLUSTER_FEATURES + 3], 1);

		//printf("id %d: Assegno il punto %f,%f al cluster %f\n",id_punto, x_punto, y_punto, punti[id_punto * POINT_FEATURES + 2]);
	}

}

__global__ void cluster_recomputecenters_cuda(float* points, float* clusters) {

	bool flag;
	long id_cluster = threadIdx.x + blockIdx.x * blockDim.x;
	//printf("Ricalcolo il cluster %d\n", id_cluster);
	float sizeX = clusters[id_cluster * CLUSTER_FEATURES + 1];
	float sizeY = clusters[id_cluster * CLUSTER_FEATURES + 2];
	float nPoints = clusters[id_cluster * CLUSTER_FEATURES + 3];
	float newX = sizeX / nPoints;
	float newY = sizeY / nPoints;
	long cluster_center_index = (long)clusters[id_cluster * CLUSTER_FEATURES + 0];
	float x = points[cluster_center_index * POINT_FEATURES + 0];
	float y = points[cluster_center_index * POINT_FEATURES + 1];

	if (x == newX && y == newY) {
		flag = false;
	}
	else {
		points[cluster_center_index * POINT_FEATURES + 0] = newX;
		points[cluster_center_index * POINT_FEATURES + 1] = newY;
		flag = true;
	}




}

__global__ void cuda_remove_points_cluster(float* clusters) {
	//<centro, size_x, size_y, punti>
	long id_cluster = threadIdx.x + blockIdx.x * blockDim.x;
	clusters[id_cluster * CLUSTER_FEATURES + 1] = 0;
	clusters[id_cluster * CLUSTER_FEATURES + 2] = 0;
	clusters[id_cluster * CLUSTER_FEATURES + 3] = 0;
}

int main()
{
	float* punti = (float*)malloc(POINT_NUM * POINT_FEATURES * sizeof(float));
	float* clusters = (float*)malloc(CLUSTER_NUM * CLUSTER_FEATURES * sizeof(float));

	init_all(punti, clusters);

	float* punti_d = 0;
	float* cluster_d = 0;

	int iterazioni = 10;
	clock_t begin = clock();

	cudaMalloc(&punti_d, POINT_NUM * POINT_FEATURES * sizeof(float));
	cudaMalloc(&cluster_d, CLUSTER_NUM * CLUSTER_FEATURES * sizeof(float));

	cudaMemcpy(punti_d, punti, POINT_NUM * POINT_FEATURES * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cluster_d, clusters, CLUSTER_NUM * CLUSTER_FEATURES * sizeof(float), cudaMemcpyHostToDevice);

	for (int i = 0; i < iterazioni; i++) {


		assign_clusters << < (POINT_NUM + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK >> > (punti_d, cluster_d);
		cudaDeviceSynchronize();
		cluster_recomputecenters_cuda << <1, CLUSTER_NUM >> > (punti_d, cluster_d);
		cudaDeviceSynchronize();
		cuda_remove_points_cluster << <1, CLUSTER_NUM >> > (cluster_d);
		cudaDeviceSynchronize();
		//clusters_recomputeCenter(punti, clusters);
	}
	cudaMemcpy(punti, punti_d, POINT_NUM * POINT_FEATURES * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(clusters, cluster_d, CLUSTER_NUM * CLUSTER_FEATURES * sizeof(float), cudaMemcpyDeviceToHost);
	clock_t end = clock();
	float time_spent = (float)(end - begin) / CLOCKS_PER_SEC;
	printf("Tempo %f", time_spent);
	//print_points(punti);
	//print_clusters(clusters);
	write_to_file(punti);

}

