#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#ifndef SPKMEANS_H_   
#define SPKMEANS_H_


double** allocate_matrix(int N, int d);
double* allocate_vector(int d);
int* allocate_int_vector(int d);
void destroy_matrix(double** matrix, int N);
void destroy_vector(double* vector);
void destroy_int_vector(int* vector);

/*---------------------------------------------------------------------------------------*/

void find_N_d(char* filename, int* N_d);
void parse_file(char* filename, double** vectors, int N, int d);
void print_matrix(double** matrix, int N, int d);
void print_diag_matrix(double* diag, int N);
void print_vector(double* vec, int d);

/*--------------------------------------computation of W, D and L_norm-----------------------------------------------------*/

void compute_W_D(double** X, double** W, double* D, int N, int d);
void compute_L_norm(double** L_norm, double** W, double* D, int N);

/*-------------------------------------Jacobi---------------------------------------------------*/
void jacobi_algorithm(double** V, double** A, int N);

/*-------------------------------------Eigenmap Heuristic---------------------------------------------------*/

int find_k(double* E, int* ID, int N, int K);

/*-------------------------------------Compute U---------------------------------------------------*/

void compute_U(double** U, double** V, int* ID, int N, int K);
void normalize(double** U, int N, int K);

/*-------------------------------------K-Means-------------------------------------------------*/

double distance(double* x, double* mu, int d);
void vectorized_plus_equal(double* vector1, double* vector2, int d);
void copy_vector_to_vector(double* vector1, double* vector2, int d);
void zero_a_vector(double* vector, int d);
int check_vectors_equility(double* vector1, double* vector2, int d);
int find_closest_cluster(double* vector, double** centroids, int K, int d);
void initialize_centroids(double** centroids, double** U, int K, int d);
void kmeans(double** U, double** centroids, int N, int d, int K);


#endif