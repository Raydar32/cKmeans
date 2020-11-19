
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define COORD_MAX 10
#define CLUSTER_NUM 3
#define POINT_NUM 10
#define POINT_FEATURES 3
#define CLUSTER_FEATURES 4
#define NUM_THREAD 12

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
		int cluster =punti[i * POINT_FEATURES + 2];
		printf("punto %d, x:%f y:%f, c:%d\n", i, x, y, cluster);
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
double distance(double x1, double x2, double y1, double y2) {
	return sqrt(((x1 - x2) * (x1 - x2)) + ((y1 - y2) * (y1 - y2)));
}


void assign_clusters(int id,double* punti, double* clusters) {
	int id_punto = id;
	double x_punto, x_cluster, y_punto, y_cluster = 0;
	x_punto = punti[id_punto* POINT_FEATURES+0];
	y_punto = punti[id_punto* POINT_FEATURES+1];
	int best_fit = 0;
	long distMax = LONG_MAX;
	for (int i = 0; i < CLUSTER_NUM; i++) {
		int cluster_index_point = clusters[i*CLUSTER_FEATURES+0];
		x_cluster = punti[cluster_index_point*POINT_FEATURES+0];
		y_cluster = punti[cluster_index_point * POINT_FEATURES + 1];
		if (distance(x_punto, x_cluster, y_punto, y_cluster) < distMax) {
			best_fit = i;
			distMax = distance(x_punto, x_cluster, y_punto, y_cluster);
		}
	}
	punti[id_punto * POINT_FEATURES+2] = best_fit;
	clusters[best_fit * CLUSTER_FEATURES + 1] = clusters[best_fit * CLUSTER_FEATURES + 1] + x_punto;
	clusters[best_fit * CLUSTER_FEATURES + 2] = clusters[best_fit * CLUSTER_FEATURES + 2] + y_punto;
	clusters[best_fit * CLUSTER_FEATURES + 3] = clusters[best_fit * CLUSTER_FEATURES + 3] + 1;
	printf("Assegno il punto %f,%f al cluster %d\n", x_punto, y_punto, best_fit);
}

int main()
{
	double* punti = (double*)malloc(POINT_NUM * POINT_FEATURES* sizeof(double));
	double* clusters = (double*)malloc(CLUSTER_NUM * CLUSTER_FEATURES * sizeof(double));
	init_all(punti, clusters);

	for (int i = 0; i < POINT_NUM; i++) {
		assign_clusters(i, punti, clusters);
	}

	print_points(punti);
	print_clusters(clusters);
	write_to_file(punti);
	
}

