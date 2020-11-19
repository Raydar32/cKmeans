#include <stdio.h>
#include <time.h>
#include <random>
#include <limits.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define COORD_MAX 300000
#define CLUSTER_NUM 20
#define POINT_NUM 10000
#define NUM_THREAD 12

using namespace std;


double distance(double x1, double x2, double y1, double y2) {
    return sqrt(((x1 - x2) * (x1 - x2)) + ((y1 - y2) * (y1 - y2)));
}


//Ricalcolo i centroidi di tutti i clusters,
bool clusters_recomputeCenter(double points[POINT_NUM][3],
                              double clusters[CLUSTER_NUM][4]) { //cluster <punto,sizeX,sizeY,n_points>
    int acc = 0;
    for (int i = 0; i < CLUSTER_NUM; i++) {
        double newX = clusters[i][1] / clusters[i][3];
        double newY = clusters[i][2] / clusters[i][3];
        int cluster_center_index = (int) clusters[i][0];
        double x = points[cluster_center_index][0];
        double y = points[cluster_center_index][1];
        double epsilon = 0.001;
        //x == newX && y == newY
        if (abs(x - newX) < epsilon && abs(y - newY) < epsilon) {
            acc = acc;
        } else {
            points[cluster_center_index][0] = newX;
            points[cluster_center_index][1] = newY;
            acc++;
        }
    }
    if (acc == 0) {
        return false;
    } else {
        return true;
    }
}

void remove_points_from_clusters(double clusters[CLUSTER_NUM][4]) { //<punto,sizeX,sizeY,n_points>
#pragma omp parallel for default(none) shared(clusters) num_threads(NUM_THREAD)
    for (int i = 0; i < CLUSTER_NUM; i++) {
        clusters[i][1] = 0;
        clusters[i][2] = 0;
        clusters[i][3] = 0;
    }
}


//Assegno i punti, qui ho un riferimento alla matrice,
//cluster <punto,sizeX,sizeY,n_points>
//punti <x,y,cluster>
void assignPoints(int id_punto1, double punti[POINT_NUM][3], double clusters[CLUSTER_NUM][4]) {
    int id_punto = id_punto1;
    double x_punto, x_cluster, y_punto, y_cluster = 0;
    x_punto = punti[id_punto][0];
    y_punto = punti[id_punto][1];
    int best_fit = 0;
    long distMax = LONG_MAX;
    for (int i = 0; i < CLUSTER_NUM; i++) {
        int cluster_index_point = clusters[i][0];
        x_cluster = punti[cluster_index_point][0];
        y_cluster = punti[cluster_index_point][1];
        if (distance(x_punto, x_cluster, y_punto, y_cluster) < distMax) {
            best_fit = i;
            distMax = distance(x_punto, x_cluster, y_punto, y_cluster);
        }
    }
    punti[id_punto][2] = best_fit;
    clusters[best_fit][1] = clusters[best_fit][1] + x_punto;
    clusters[best_fit][2] = clusters[best_fit][2] + y_punto;
    clusters[best_fit][3] = clusters[best_fit][3] + 1;
};


double clusters[CLUSTER_NUM][4]; //cluster <punto,sizeX,sizeY,n_points>
double punti[POINT_NUM][3]; //punti <x,y,cluster>


double random_double() {
    double x;
    x = (double) rand() * (double) 32767;
    x = fmod(x, COORD_MAX);
    return x;
}

void write_to_file(double punti[POINT_NUM][3]) {
    FILE *fPtr;
    char filePath[100] = {"G:\\file.dat"};
    char dataToAppend[1000];
    fPtr = fopen(filePath, "a");
    for (int i = 0; i < POINT_NUM; i++) {
        fprintf(fPtr, "%f %f %f\n", punti[i][0], punti[i][1], punti[i][2]);
    }
}

int main() {
    srand(time(NULL));
    //Inizializzo le matrici dei punti e dei clusters

    for (int i = 0; i < POINT_NUM; i++) {
        punti[i][0] = random_double();
        punti[i][1] = random_double();
        punti[i][2] = 0;
    }

    for (int j = 0; j < CLUSTER_NUM; j++) {
        clusters[j][0] = rand() % (POINT_NUM - 0) + 0;
        clusters[j][1] = 0;
        clusters[j][2] = 0;
        clusters[j][3] = 0;

    }

    clock_t begin = clock();

    printf("Avvio algoritmo di clustering\n");
    int it = 0;
    bool flag = true;
    while (it < 20 && flag) {
#pragma omp parallel for default(none) shared(punti, clusters, flag) num_threads(NUM_THREAD)
        for (int j = 0; j < POINT_NUM; j++) {
            assignPoints(j, punti, clusters);
        }
        flag = clusters_recomputeCenter(punti, clusters);
        remove_points_from_clusters(clusters);
        printf("it: %d\n", it);
        it++;
    }
    clock_t end = clock();
    double time_spent = (double) (end - begin) / CLOCKS_PER_SEC;
    printf("Tempo : %f", time_spent);
    write_to_file(punti);
    return 0;
}
