#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "spkmeans.h"

#define invalid_input "Invalid Input!"
#define error_occured "An Error Has Occured"


enum goal{spk, wam, ddg, lnorm, jacobi};


double** allocate_matrix(int N, int d){
    double** matrix;
    int i;
    matrix = (double**)calloc(N, sizeof(double*));
    assert(matrix != NULL);
    for (i = 0; i < N; i++) {
        matrix[i] = (double*) calloc(d, sizeof(double));
        assert(matrix[i] != NULL);
    }
    return matrix;
}

double* allocate_vector(int d){
    double* vector;
    vector = (double*)calloc(d,sizeof(double));
    assert(vector != NULL);
    return vector;
}

int* allocate_int_vector(int d){
    int* vector;
    vector = (int*)calloc(d,sizeof(int));
    assert(vector != NULL);
    return vector;
}

void destroy_matrix(double** matrix, int N){
    int i;
    for (i=0; i < N; i++){
        free(matrix[i]);
    }
    free(matrix);
}

void destroy_vector(double* vector){
    free(vector);
}

void destroy_int_vector(int* vector){
    free(vector);
}


/*----------------------------------------------------------------------------------------*/

void find_N_d(char* filename, int* N_d){ /* assumig that there is no empty line at the end */
    FILE* file;
    char c;

    file = fopen(filename, "r");
    if (file != NULL) {
        while ((c = fgetc(file)) != EOF) {
            if(c=='\n')
                N_d[0]++;
            if (N_d[0] == 0)
            {
                if (c == ',')
                    N_d[1]++;
            }
        }
        fclose(file);

        N_d[0]++;   /* no new line char at the end of the last line */
        N_d[1]++;   /* no comma at the end of the last column */
    }
    else{
        printf(error_occured);
        exit(0);
    }
}



void parse_file(char* filename, double** vectors, int N, int d) { /*extracting the points from the file*/
    FILE* file;
    double number;
    int i;
    int j;

    file = fopen(filename, "r");

    if (file == NULL) {
        printf(error_occured);
        exit(0);
    }

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < d; j++){
            fscanf(file, "%lf", &number);
            if (!(i == N-1 && j == d-1)) {
                fgetc(file);
            }
            vectors[i][j] = number;
        }
    }
}


static void print_number(double number){    /* eliminate situations where -0 is printed */
    if ((number < 0) && (number > -0.00005)){
        printf("0.0000");
    }
    else{
        printf("%.4f", number);
    }
}


void print_matrix(double** matrix, int N, int d){
    int i;
    int j;
    for (i = 0; i < N; i++){
        for (j = 0; j < d; j++){
            if (j < d-1) {
                print_number(matrix[i][j]);
                printf(",");
            }
            else{
                print_number(matrix[i][j]);
            }

        }
        printf("\n");
    }
}


void print_transposed_matrix(double** matrix, int N, int d){
    int i;
    int j;
    for (i = 0; i < N; i++){
        for (j = 0; j < d; j++){
            if (j < d-1) {
                print_number(matrix[j][i]);
                printf(",");
            }
            else{
                print_number(matrix[j][i]);
            }

        }
        printf("\n");
    }
}

void print_diag_matrix(double* diag, int N){
    int i;
    int j;
    for (i = 0; i < N; i++){
        for (j = 0; j < N; j++){
            if (i == j){
                if (j < N - 1){
                    print_number(diag[i]);
                    printf(",");
                }
                else{
                    print_number(diag[i]);
                }
            }
            else{
                if (j < N - 1){
                    printf("0.0000,");
                }
                else{
                    printf("0.0000");
                }
            }
        }
        printf("\n");
    }
}

void print_vector(double* vec, int d){
    int j;
    for (j = 0; j < d; j++){
        if (j < d - 1) {
            print_number(vec[j]);
            printf(",");
        }
        else{
            print_number(vec[j]);
        }
    }
    printf("\n");
}


void print_int_vector(int* vec, int d){
    int j;
    for (j = 0; j < d; j++){
        if (j < d - 1) {
            printf("%d,", vec[j]);
        }
        else{
            printf("%d", vec[j]);
        }
    }
    printf("\n");
}

/*--------------------------------------computation of W, D and L_norm-----------------------------------------------------*/

static double calc_euclidian_dist(double* x, double* y, int d){
    int i;
    double dist;
    dist = 0;
    for (i = 0; i < d; i++){
        dist += (x[i] - y[i]) * (x[i] - y[i]);
    }
    dist = sqrt(dist);
    return dist;
}

static double calc_dist(double* x, double* y, int d){
    double exponent;
    double dist;
    exponent = calc_euclidian_dist(x, y, d) / 2;
    dist = exp((-1) * exponent);
    return dist;
}


void compute_W_D(double** X, double** W, double* D, int N, int d){
    int i;
    int j;
    double weight;
    for(i = 0; i < N; i++){
        for (j = i+1; j < N; j++){
            weight = calc_dist(X[i], X[j], d);
            W[i][j] = weight;
            D[i] += weight;
            W[j][i] = weight;
            D[j] += weight;
        }
    }
}

static void update_D_to_minus_half_exp(double* D, int N){
    int i;
    for (i = 0; i < N; i++){
        D[i] = 1 / (sqrt(D[i]));
    }
}

void compute_L_norm(double** L_norm, double** W, double* D, int N){
    int i;
    int j;
    update_D_to_minus_half_exp(D, N);
    for (i = 0; i < N; i++){
        for (j = i; j < N; j ++){
            if (i == j){
                L_norm[i][j] = 1;
            }
            else{
                L_norm[i][j] = (-1) * (D[i] * D[j] * W[i][j]);
                L_norm[j][i] = L_norm[i][j];
            }
        }
    }
}


/*-------------------------------------Jacobi---------------------------------------------------*/


static int is_diag(double** A, int N){
    int i;
    int j;
    for (i = 0; i < N; i++){
        for (j = i + 1; j < N; j++){
            if (A[i][j] != 0){
                return 0;
            }
        }
    }
    return 1;
} 

static void find_max_index(double** A, int N, int* maxi){
    double aij_absulote_value;
    double max_element;
    int i = 0;
    int j = 1;
    maxi[0] = i;
    maxi[1] = j;
    max_element = fabs(A[0][1]);

    for (i = 0; i < N; i++){
        for (j = i + 1; j < N; j++){
            aij_absulote_value = fabs(A[i][j]);
            if (aij_absulote_value > max_element){
                max_element = aij_absulote_value;
                maxi[0] = i;
                maxi[1] = j;
            }
        }
    }
}

static double compute_theta(double** A, int* maxi){
    int i = maxi[0];
    int j = maxi[1];
    double theta;
    theta = A[j][j] - A[i][i];
    theta = theta / (2 * A[i][j]);
    return theta;
}



static double compute_t(double theta){
    double t;
    double mechane;
    double mone;
    
    mechane = (fabs(theta) + (sqrt((theta*theta) + 1)));
    if (theta < 0){
        mone = -1;
    }
    else{
        mone = 1;
    }
    t = mone / mechane;
    return t;
}

static double compute_c(double t){
    double c;
    c = 1 / (sqrt(t*t + 1));
    return c;
}

static double compute_s(double c, double t){
    double s;
    s = t * c;
    return s;
}

static void update_A(double** A, int N, double c, double s, int* maxi){
    int i = maxi[0];
    int j = maxi[1];
    double a_ii = A[i][i];
    double a_jj = A[j][j];
    double a_ij = A[i][j];
    double a_ri;
    double a_rj;
    int r;

    for (r = 0; r < N; r++){
        if (r != i && r != j){
            a_ri = A[r][i];
            a_rj = A[r][j];
            A[r][i] = c * a_ri - s * a_rj;
            A[r][j] = c * a_rj + s * a_ri;

            A[i][r] = A[r][i];
            A[j][r] = A[r][j];
        }
    }
    A[i][i] = c*c * a_ii + s*s * a_jj - 2 * s * c * a_ij;
    A[j][j] = s*s * a_ii + c*c * a_jj + 2 * s * c * a_ij;    
    A[i][j] = 0;
    A[j][i] = 0;
}

static void update_V(double** V, int N, double c, double s, int* maxi){
    int i = maxi[0];
    int j = maxi[1];
    int r;
    double v_ri;
    double v_rj;
    for (r = 0; r < N; r++){
        v_ri = V[r][i];
        v_rj = V[r][j];
        V[r][i] = c * v_ri - s * v_rj;
        V[r][j] = s * v_ri + c * v_rj;
    }
}

static double calc_off_squared(double** A, int N){
    int i;
    int j;
    double sum_of_squars = 0;

    for (i = 0; i < N; i++){
        for (j = 0; j < N; j++){
            if (i != j){
                sum_of_squars = sum_of_squars + (A[i][j] * A[i][j]);
            } 
        }
    }
    return sum_of_squars;
}


void jacobi_algorithm(double** V, double** A, int N){
    int i;
    int iter;
    int maxi[2] = {0, 0};
    double theta;
    double t;
    double c;
    double s;
    double prev_off;
    double curr_off;
    const double epsilon = 1.0e-15;
    const int MAX_ITER = 100;

    for (i = 0; i < N; i++){    /* initialize V to be Identity matrix */
        V[i][i] = 1;
    }

    curr_off = calc_off_squared(A, N);

    for (iter = 0; iter < MAX_ITER; iter++){
        if (!(is_diag(A, N) == 0)){
            break;
        }         
        find_max_index(A, N, maxi);
        theta = compute_theta(A, maxi);
        t = compute_t(theta);
        c = compute_c(t);
        s = compute_s(c,t);
        update_A(A, N, c, s, maxi);
        update_V(V, N, c, s, maxi);

        prev_off = curr_off;
        curr_off = calc_off_squared(A, N);

        if (fabs(curr_off - prev_off) <= epsilon){
            break;
        }    
    }
}


/*-------------------------------------Eigenmap Heuristic---------------------------------------------------*/

static void swap(double* E, int* ID, int left, int right){
    double tmp1;
    int tmp2;

    tmp1 = E[left];
    E[left] = E[right];
    E[right] = tmp1;

    tmp2 = ID[left];
    ID[left] = ID[right];
    ID[right] = tmp2;
}


static int compare(double* E, int* ID, int index1, int index2){ /* defining new comparison relation between two elements, in order to make our sorting algorithm stable */ 
    if (E[index1] < E[index2]){
        return 0;
    }
    if (E[index1] > E[index2]){
        return 1;
    }
    else{
        if (ID[index1] < ID[index2]){
            return 0;
        }
        if (ID[index1] > ID[index2]){
            return 1;
        }
    }
    return 0;
}


static int partition(double* E, int* ID, int N, int left, int right){
    int pivot_index;
    int comp;

    pivot_index = left;
    
    while (left < right){
        
        comp = compare(E, ID, left, pivot_index);
        while (left < N && (comp == 0)){
            left++;
            comp = compare(E, ID, left, pivot_index);
        }

        comp = compare(E, ID, right, pivot_index);
        while (comp == 1){
            right--;
            comp = compare(E, ID, right, pivot_index);
        }

        if (left < right){
            swap(E, ID, left, right);
        }
    }
    swap(E, ID, right, pivot_index);
    return right;
}

static void quick_sort(double* E, int* ID, int N, int left, int right){
    int index;
    if (left < right){
        index = partition(E, ID, N, left, right);
        quick_sort(E, ID, N, left, index - 1); 
        quick_sort(E, ID, N, index + 1, right); 
    }
}


static int compute_k_given_sorted_eigenvalues(double* E, int N){
    double max_delta = -1;
    int max_i = -1;
    int i;
    double delta;
    for (i = 0; i < (N / 2); i++){
        delta = E[i+1] - E[i];
        if (delta > max_delta){
            max_delta = delta;
            max_i = i + 1;
        }
    }
    return max_i;
}


int find_k(double* E, int* ID, int N, int K){
    quick_sort(E, ID, N, 0, N-1);    
    if (K == 0){
        return compute_k_given_sorted_eigenvalues(E, N);
    }
    return K;
}


static void create_compact_repr_for_diag_mat(double** A, double* E, int N){
    int i;
    for (i = 0; i < N; i++){
        E[i] = A[i][i];
    }
}

static void fill_ID_vector(int* ID, int N){
    int i;
    for (i = 0; i < N; i++){
        ID[i] = i;
    }
}


/*-------------------------------------Compute U---------------------------------------------------*/


void compute_U(double** U, double** V, int* ID, int N, int K){
    int i;
    int j;
    int col_index;

    for (j = 0; j< K; j++){
        col_index = ID[j];
        for (i = 0; i < N; i++){
            U[i][j] = V[i][col_index];
        }
    }
}


void normalize(double** U, int N, int K){ /* normalizing each row of U in place - no need for new matrix allocation (T) */ 
    int i;
    int j;
    double row_norm;

    for (i = 0; i < N; i++){
        row_norm = 0;
        for (j = 0; j < K; j++){
            row_norm = row_norm + (U[i][j] * U[i][j]);
        }
        row_norm = sqrt(row_norm);

        for (j = 0; j < K; j++){
            U[i][j] = (U[i][j] / row_norm);
        }
    }
}


/*-------------------------------------K-Means-------------------------------------------------*/

double distance(double* x, double* mu, int d){
    int i;
    double dist = 0;
    for (i = 0; i < d; i++){
        dist += (x[i] - mu[i]) * (x[i] - mu[i]);
    }
    return dist;
}

void vectorized_plus_equal(double* vector1, double* vector2, int d){ /* add the second vector to the first, and update the first in place */ 
    int j;
    for (j = 0 ; j < d; j++){
        vector1[j] = vector1[j] + vector2[j];
    }
}

void copy_vector_to_vector(double* vector1, double* vector2, int d){    /* updating vector1 to be vector2 in place */
    int j;
    for (j = 0; j < d; j++){
        vector1[j] = vector2[j];
    }
}

void zero_a_vector(double* vector, int d){
    int j;
    for (j = 0; j < d; j++){
        vector[j] = 0;
    }
}

int check_vectors_equility(double* vector1, double* vector2, int d){    /* return 1 if the vectors are stricly equal */ 
    int j;
    for (j=0; j < d; j++){
        if (vector1[j] != vector2[j]){
            return 0;
        }
    }
    return 1;
}

int find_closest_cluster(double* vector, double** centroids, int K, int d){
    int argmin = -1;
    double dist;
    double min_dist;
    int j;

    for (j = 0; j < K; j++){
        dist = distance(vector, centroids[j], d);
        if (j == 0){ /*inialization of min_dist*/
            min_dist = dist;
            argmin = 0;
        }
        if (dist < min_dist){
            min_dist = dist;
            argmin = j;
        }
    }
    return argmin;

}

void initialize_centroids(double** centroids, double** U, int K, int d){
    int i;
    for (i = 0; i < K; i++){
        copy_vector_to_vector(centroids[i], U[i], d);
    }
}



void kmeans(double** U, double** centroids, int N, int d, int K){   /* update centroids to be final kmeans clusters centroids */ 
    const int MAX_ITER = 300;
    double** clusters_sum;
    int* clusters_size;
    double* last_centroid;
    int m;
    int i;
    int j;
    int k;
    int closest_cluster;
    int has_change_done;


    /* allocate memory */
    clusters_sum = allocate_matrix(K, d);
    clusters_size = allocate_int_vector(K);
    last_centroid = allocate_vector(d);


    for (m = 0; m < MAX_ITER; m++){

        /*initialize auxilary data*/
        for (i = 0; i < K; i++){
            clusters_size[i] = 0;
            zero_a_vector(clusters_sum[i], K);
        }

        for (i = 0; i < N; i++){
            closest_cluster = find_closest_cluster(U[i], centroids, K, d);
            clusters_size[closest_cluster]++;
            vectorized_plus_equal(clusters_sum[closest_cluster], U[i], d);
        }

        has_change_done = 0;
        for (k = 0; k < K; k++){
            copy_vector_to_vector(last_centroid, centroids[k], d);
            
            for (j = 0; j < d; j++){
                centroids[k][j] = clusters_sum[k][j] / clusters_size[k];
            }

            if (!(check_vectors_equility(centroids[k], last_centroid, d))){
                has_change_done = 1;
            }
        }

        if (!(has_change_done)){
            break;
        }

    }

    destroy_matrix(clusters_sum, K);
    destroy_int_vector(clusters_size);
    destroy_vector(last_centroid);
}





/*-------------------------------------Main---------------------------------------------------*/


int main(int argc, char** argv) {
    double** X;
    double** W;
    double* D;
    double** L_norm; 
    double** A; /* jacobi eigenvalues matrix */
    double** V; /* jacobi eigenvectors matrix */
    double* E;
    int* ID;
    double** U;
    int N;
    int d;
    int N_d[2] = {0, 0};
    char* file_name;
    char* k_char;
    char tmp_char;
    int K;
    char* input_goal_str;
    enum goal input_goal;
    double** centroids;

    /* check amount of arguments validity */
    if (argc != 4){
        printf(invalid_input);
        exit(0);
    }

    /* parse K and check validity */
    k_char = argv[1];
    while (k_char != NULL && *k_char != '\0'){
        tmp_char = *k_char;
        if (tmp_char < '0' || tmp_char > '9'){
            printf(invalid_input);
            exit(0);
        }
        k_char++;
    }

    K = atoi(argv[1]);

    /* parse GOAL and check validity */
    input_goal_str = argv[2];
    if (!strcmp(input_goal_str, "spk")){
        input_goal = spk;
    }
    else if (!strcmp(input_goal_str, "wam")){
        input_goal = wam;
    }
    else if (!strcmp(input_goal_str, "ddg")){
        input_goal = ddg;
    }
    else if (!strcmp(input_goal_str, "lnorm")){
        input_goal = lnorm;
    }
    else if (!strcmp(input_goal_str, "jacobi")){
        input_goal = jacobi;
    }
    else{
        printf(invalid_input);
        exit(0);
    }


    /*parse file_name */
    file_name = argv[3];

    /* parse N and d */
    find_N_d(file_name, N_d);
    N = N_d[0];
    d = N_d[1];


    /* create the data matrix X */
    X = allocate_matrix(N, d);
    parse_file(file_name, X, N, d);


    if (input_goal == jacobi){
        A = X;
        V = allocate_matrix(N, N);
        jacobi_algorithm(V, A, N);

        E = allocate_vector(N);
        create_compact_repr_for_diag_mat(A, E, N);

        print_vector(E, N);
        print_transposed_matrix(V, N, N);

        destroy_matrix(X, N);
        destroy_matrix(V, N);
        destroy_vector(E);
        exit(0);
    }

    /* compute W and D */
    W = allocate_matrix(N, N);
    D = allocate_vector(N);
    compute_W_D(X, W, D, N, d);
    
    if (input_goal == wam){
        print_matrix(W, N, N);
        destroy_matrix(W, N);
        destroy_vector(D);
        destroy_matrix(X, N);
        exit(0);
    }

    if (input_goal == ddg){
        print_diag_matrix(D, N);
        destroy_matrix(W, N);
        destroy_vector(D);
        destroy_matrix(X, N);
        exit(0);
    }

    /* compute L_norm */
    L_norm = allocate_matrix(N, N);
    compute_L_norm(L_norm, W, D, N);

    if (input_goal == lnorm){
        print_matrix(L_norm, N, N);
        destroy_matrix(W, N);
        destroy_vector(D);
        destroy_matrix(X, N);
        destroy_matrix(L_norm, N);
        exit(0);
    }

    /* compute the eigenvectors matrix (V) and eigenvalue (A) */
    A = L_norm;
    V = allocate_matrix(N, N);
    jacobi_algorithm(V, A, N);

    E = allocate_vector(N);
    create_compact_repr_for_diag_mat(A, E, N);

    /* find number of clusters (K) */
    ID = allocate_int_vector(N);
    fill_ID_vector(ID, N);
    
    K = find_k(E, ID, N, K);

    /* compute U */
    U = allocate_matrix(N, K);
    compute_U(U, V, ID, N, K);
    normalize(U, N, K);

    /* find centroids */
    centroids = allocate_matrix(K, K);
    initialize_centroids(centroids, U, K, K);

    /* run k-means */
    kmeans(U, centroids, N, K, K);
    print_matrix(centroids, K, K);
    destroy_matrix(W, N);
    destroy_vector(D);
    destroy_matrix(X, N);
    destroy_matrix(A, N);
    destroy_matrix(V, N);
    destroy_matrix(U, N);
    destroy_vector(E);
    destroy_int_vector(ID);
    destroy_matrix(centroids, K);

    return 0;
}