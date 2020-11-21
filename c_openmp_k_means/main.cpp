
//'---------------------------------------------------------------------------------------
//' File      : main.cpp
//' Author    : Alessandro Mini (mat. 7060381)
//' Date      : 8/11/2020
//' Purpose   : Main class for both sequential and OpenMP versions of K-means.
//'---------------------------------------------------------------------------------------

#include <stdio.h>
#include <time.h>
#include <random>
#include <limits.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <float.h>
//Here we have some defines:

#define COORD_MAX 100000        // <- coordinates range
#define CLUSTER_NUM 30          // <- number of clusters
#define POINT_NUM 1000000       // <- number of points
#define POINT_FEATURES 3		// <- features of a point (x,y,cluster)
#define CLUSTER_FEATURES 4		// <- feature of a cluster (center,sizex,sizey,npoints)
#define NUM_THREAD 12           // <- number of threads (1=sequential.) tested on Ryzen5 3600
#define IT_MAX 10               // <- number of iterations
#define EPSILON 0.001           // <- value that extabilish the tolerance from which 2 points
                                //    are "near enough" to be considered the same point.

using namespace std;

//'---------------------------------------------------------------------------------------
//' Method    : distance
//' Purpose   : With this method i calculate the (euclidean) distance between 2 points
//'---------------------------------------------------------------------------------------
float distance(float x1, float x2, float y1, float y2) {
    return sqrt(((x1 - x2) * (x1 - x2)) + ((y1 - y2) * (y1 - y2)));
}


//'---------------------------------------------------------------------------------------
//' Method    : clusters_recomputeCenter
//' Purpose   : With this method i re-calculate each cluster center using the k-means
//'             formula, there is a size sizeX and sizeY that accumulates during the
//'             execution.
//'---------------------------------------------------------------------------------------
void clusters_recomputeCenter(float points[POINT_NUM][POINT_FEATURES],float clusters[CLUSTER_NUM][CLUSTER_FEATURES]) { //cluster <punto,sizeX,sizeY,n_points>
    #pragma omp parallel for default(none) shared(clusters,points) num_threads(NUM_THREAD)
    for (int i = 0; i < CLUSTER_NUM; i++) {
        float newX = clusters[i][1] / clusters[i][3];
        float newY = clusters[i][2] / clusters[i][3];
        int cluster_center_index = (int) clusters[i][0];
        float x = points[cluster_center_index][0];
        float y = points[cluster_center_index][1];
        if(!(abs(x - newX) < EPSILON && abs(y - newY) < EPSILON)) {
            points[cluster_center_index][0] = newX;
            points[cluster_center_index][1] = newY;
        }
    }
}

//'---------------------------------------------------------------------------------------
//' Method    : remove_points_from_clusters
//' Purpose   : this method sets to "zero" each cluster in terms of sizex,sizey and
//'             number of points inside that cluster.
//'---------------------------------------------------------------------------------------
void remove_points_from_clusters(float clusters[CLUSTER_NUM][CLUSTER_FEATURES]) {
#pragma omp parallel for default(none) shared(clusters) num_threads(NUM_THREAD)     // <- parallelized
    for (int i = 0; i < CLUSTER_NUM; i++) {                                         // <- <punto,sizeX,sizeY,n_points>
        clusters[i][1] = 0;
        clusters[i][2] = 0;
        clusters[i][3] = 0;
    }
}


//'---------------------------------------------------------------------------------------
//' Method    : assignPoints
//' Purpose   : this method is the "core" of the k-means algorithm. it will assign each
//'             point to its best fitting cluster based on the distance.
//'             each point has this structure  : p: <x,y,cluster>
//'             each cluster has this structure: c: <center,sizeX,sizeY,n_points>
//'---------------------------------------------------------------------------------------
void assignPoints(int punto, float punti[POINT_NUM][POINT_FEATURES], float clusters[CLUSTER_NUM][CLUSTER_FEATURES]) {
    float x_punto, x_cluster, y_punto, y_cluster = 0;
    x_punto = punti[punto][0];
    y_punto = punti[punto][1];
    int best_fit = 0;
    float distMax = FLT_MAX;
    for (int i = 0; i < CLUSTER_NUM; i++) {
        int cluster_index_point = (int)clusters[i][0];
        x_cluster = punti[cluster_index_point][0];
        y_cluster = punti[cluster_index_point][1];
        if (distance(x_punto, x_cluster, y_punto, y_cluster) < distMax) {
            best_fit = i;
            distMax = distance(x_punto, x_cluster, y_punto, y_cluster);
        }
    }
    punti[punto][2] = (float)best_fit;
#pragma omp atomic                                              // <- here we have a critical section
    clusters[best_fit][1] = clusters[best_fit][1] + x_punto;    //    two points that are assigned
#pragma omp atomic                                              //    fast enough can start a race cond.
    clusters[best_fit][2] = clusters[best_fit][2] + y_punto;
#pragma omp atomic
    clusters[best_fit][3] = clusters[best_fit][3] + 1;

};


//'---------------------------------------------------------------------------------------
//' Method    : random_float
//' Purpose   : This method generates a random-ish float number in COORD_MAX.
//'---------------------------------------------------------------------------------------
float random_float() {
    float x;
    x = (float) rand() * (float) 32767;
    x = fmod(x, COORD_MAX);
    return x;
}


//'---------------------------------------------------------------------------------------
//' Method    : write_to_file
//' Purpose   : This method will print points and clusters in a GNUPLOT format 1:2:3
//'---------------------------------------------------------------------------------------
void write_to_file(float punti[POINT_NUM][POINT_FEATURES]) {
    FILE *fPtr;
    char filePath[100] = {"G:\\file.dat"};
    char dataToAppend[1000];
    fPtr = fopen(filePath, "a");
    for (int i = 0; i < POINT_NUM; i++) {
        fprintf(fPtr, "%f %f %f\n", punti[i][0], punti[i][1], punti[i][2]);
    }
}


//'---------------------------------------------------------------------------------------
//' Method    : init_all
//' Purpose   : This method will init points and clusters
//'---------------------------------------------------------------------------------------
void init_all(float punti[POINT_NUM][POINT_FEATURES],float clusters[CLUSTER_NUM][CLUSTER_FEATURES]){
    for(int i = 0;i<POINT_NUM;i++){
        punti[i][0] = random_float();
        punti[i][1] = random_float();
        punti[i][2] = 0;
    }
    for (int j = 0; j < CLUSTER_NUM; j++) {
        clusters[j][0] = rand() % (POINT_NUM - 0) + 0;
        clusters[j][1] = 0;
        clusters[j][2] = 0;
        clusters[j][3] = 0;

    }
}


float clusters[CLUSTER_NUM][CLUSTER_FEATURES];  //<- here i define the cluster matrix.
float punti[POINT_NUM][POINT_FEATURES];         //<- here i define the point matrix.

int main() {                                    //<- program entry point.

    srand(time(NULL));
    init_all(punti,clusters);
    printf("Avvio algoritmo di clustering\n");
    clock_t begin = clock();
   for(int i = 0; i< IT_MAX;i++){
#pragma omp parallel for default(none) shared(punti, clusters) num_threads(NUM_THREAD)
        for (int j = 0; j < POINT_NUM; j++) {
            assignPoints(j, punti, clusters);
        }
        clusters_recomputeCenter(punti, clusters);
        remove_points_from_clusters(clusters);
        printf("it: %d\n", i);
    }
    clock_t end = clock();
    float time_spent = (float) (end - begin) / CLOCKS_PER_SEC;
    printf("Tempo : %f", time_spent);
    write_to_file(punti);
    return 0;
}
