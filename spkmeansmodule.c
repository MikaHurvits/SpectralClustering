#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "spkmeans.h"


/* ----------------------------------------------------- begin ------------------------------------------------------------------ */


static PyObject* matrix_from_C_to_Python(double** matrix, int N, int d){
    PyObject* matrix_python;
    int i;
    int j;

    matrix_python = (PyObject*) PyList_New(0);
    for (i = 0; i < N; i++){
        PyObject* row = (PyObject*) PyList_New(d);
        for (j = 0; j < d; j++){
            PyList_SET_ITEM(row, j, PyFloat_FromDouble(matrix[i][j]));
        }
        PyList_Append(matrix_python, row);
    }
    return matrix_python;
}

static void matrix_from_Python_to_C(double** X, PyObject* matrix, int N, int d){
    int i;
    int j;
    PyObject* vector;
    PyObject* feature;
    
    for (i = 0; i < N; i++){
    vector = PyList_GetItem(matrix, i);
        for (j = 0; j < d; j++){
            feature = PyList_GetItem(vector, j);
            if (!PyFloat_Check(feature)){
                printf("not float");
                break;
            }
            X[i][j] = PyFloat_AsDouble(feature);
            if (X[i][j]  == -1 && PyErr_Occurred()){
                printf("Something bad ...\n");
                break;
            }
        }
    }
}

static PyObject* vector_from_C_to_Python(double* vector,int d){
    PyObject* vector_python;
    int j;

    vector_python = (PyObject*) PyList_New(d);
    for (j = 0; j < d; j++){
        PyList_SET_ITEM(vector_python, j, PyFloat_FromDouble(vector[j]));
    }
    return vector_python;
}


static void vector_from_Python_to_C(double* vector_C, PyObject* vector_python, int d){
    int j;
    PyObject* feature;

    for (j = 0; j < d; j++){
        feature = PyList_GetItem(vector_python, j);
        if (!PyFloat_Check(feature)){
            printf("not float");
            exit(0);
        }
        vector_C[j] = PyFloat_AsDouble(feature);
        if (vector_C[j]  == -1 && PyErr_Occurred()){    
            printf("Something bad ...\n");
            exit(0);
        }
    }
}

static void int_vector_from_Python_to_C(int* vector_C, PyObject* vector_python, int d){
    int j;
    PyObject* feature;

    for (j = 0; j < d; j++){
        feature = PyList_GetItem(vector_python, j);
        if (!PyLong_Check(feature)){
            printf("not int");
            exit(0);
        }
        vector_C[j] = PyLong_AsLong(feature);
        if (vector_C[j]  == -1 && PyErr_Occurred()){    
            printf("Something bad ...\n");
            exit(0);
        }
    }
}

/* ---------------------------------------------------------------------------------------------------------- */



static PyObject* compute_W(PyObject *self, PyObject *args){
    PyObject* data_points;
    PyObject* W_python;

    int N;
    int d;            
    double** X;
    double** W;
    double* D;


    if (!PyArg_ParseTuple(args,"Oii", &data_points, &N, &d)){
        printf("parsing failed\n");
        return NULL;
    }
    if (!PyList_Check(data_points)){
        printf("not a list\n");
        return NULL;
    }

    // create the data matrix X
    X = allocate_matrix(N, d);

    // copy data_points to X
    matrix_from_Python_to_C(X, data_points, N, d);

    // compute W and D 
    W = allocate_matrix(N, N);
    D = allocate_vector(N);
    compute_W_D(X, W, D, N, d);

    // preperation for returning W to python
    W_python = matrix_from_C_to_Python(W, N, N);

    destroy_matrix(W, N);
    destroy_vector(D);
    destroy_matrix(X, N);

    return W_python;
    
}


static PyObject* compute_D(PyObject *self, PyObject *args){
    PyObject* data_points;
    PyObject* D_python;

    int N;
    int d;            
    double** X;
    double** W;
    double* D;


    if (!PyArg_ParseTuple(args,"Oii", &data_points, &N, &d)){
        printf("parsing failed\n");
        return NULL;
    }
    if (!PyList_Check(data_points)){
        printf("not a list\n");
        return NULL;
    }

    // create the data matrix X
    X = allocate_matrix(N, d);

    // copy data_points to X
    matrix_from_Python_to_C(X, data_points, N, d);

    // compute W and D 
    W = allocate_matrix(N, N);
    D = allocate_vector(N);
    compute_W_D(X, W, D, N, d);

    // preperation for returning W to python
    D_python = vector_from_C_to_Python(D, N);

    destroy_matrix(W, N);
    destroy_vector(D);
    destroy_matrix(X, N);

    return D_python;
}


static PyObject* compute_Lnorm(PyObject *self, PyObject *args){
    PyObject* W_python;
    PyObject* D_python;
    PyObject* lnorm_python;

    int N;
    double** L_norm;
    double** W;
    double* D;


    if (!PyArg_ParseTuple(args,"OOi", &W_python, &D_python, &N)){
        printf("parsing failed\n");
        return NULL;
    }
    if (!PyList_Check(W_python) || !PyList_Check(D_python)){
        printf("not a list\n");
        return NULL;
    }

    // create W and D in C-language
    W = allocate_matrix(N, N);
    D = allocate_vector(N);

    // copy data to W and D
    matrix_from_Python_to_C(W, W_python, N, N);
    vector_from_Python_to_C(D, D_python, N);

    // compute L_norm
    L_norm = allocate_matrix(N, N);
    compute_L_norm(L_norm, W, D, N);

    // preperation for returning W to python
    lnorm_python = matrix_from_C_to_Python(L_norm, N, N);

    destroy_matrix(W, N);
    destroy_vector(D);
    destroy_matrix(L_norm, N);

    return lnorm_python;
}


/* -------------------------- Jacobi -------------------------------------------- */


static PyObject* run_jacobi_compute_V(PyObject *self, PyObject *args){
    PyObject* V_python;
    PyObject* A_python;

    int N;
    double** A;
    double** V;


    if (!PyArg_ParseTuple(args,"Oi", &A_python, &N)){
        printf("parsing failed\n");
        return NULL;
    }
    if (!PyList_Check(A_python)){
        printf("not a list\n");
        return NULL;
    }

    // create A in C-language
    A = allocate_matrix(N, N);

    // copy data to A
    matrix_from_Python_to_C(A, A_python, N, N);
    
    // compute V matrix
    V = allocate_matrix(N, N);
    jacobi_algorithm(V, A, N);

    // preperation for returning W to python
    V_python = matrix_from_C_to_Python(V, N, N);

    destroy_matrix(A, N);
    destroy_matrix(V, N);

    return V_python;
}


static PyObject* run_jacobi_compute_A(PyObject *self, PyObject *args){
    PyObject* A_python;

    int N;
    double** A;
    double** V;


    if (!PyArg_ParseTuple(args,"Oi", &A_python, &N)){
        printf("parsing failed\n");
        return NULL;
    }
    if (!PyList_Check(A_python)){
        printf("not a list\n");
        return NULL;
    }

    // create A in C-language
    A = allocate_matrix(N, N);

    // copy data to A
    matrix_from_Python_to_C(A, A_python, N, N);
    
    // compute V matrix
    V = allocate_matrix(N, N);
    jacobi_algorithm(V, A, N);

    // preperation for returning W to python
    A_python = matrix_from_C_to_Python(A, N, N);

    destroy_matrix(A, N);
    destroy_matrix(V, N);

    return A_python;
}

/* -------------------------- find K and compute U -------------------------------------------- */

static PyObject* run_compute_normalized_U(PyObject *self, PyObject *args){
    PyObject* E_python;
    PyObject* ID_python;
    PyObject* V_python;
    PyObject* U_python;

    double* E;
    int* ID;
    double** V;
    double** U;

    int N;
    int K;


    if (!PyArg_ParseTuple(args,"OOOii", &E_python, &ID_python, &V_python, &N, &K)){
        printf("parsing failed\n");
        return NULL;
    }
    if ((!PyList_Check(E_python)) || (!PyList_Check(ID_python)) || (!PyList_Check(V_python)) ){
        printf("not a list\n");
        return NULL;
    }
    
    E = allocate_vector(N);
    vector_from_Python_to_C(E, E_python, N);
    ID = allocate_int_vector(N);
    int_vector_from_Python_to_C(ID, ID_python, N);
    
    K = find_k(E, ID, N, K);

    V = allocate_matrix(N, N);
    matrix_from_Python_to_C(V, V_python, N, N);

    U = allocate_matrix(N, K);
    compute_U(U, V, ID, N, K);
    normalize(U, N, K);

    U_python = matrix_from_C_to_Python(U, N, K);
    destroy_vector(E);
    destroy_int_vector(ID);
    destroy_matrix(V, N);
    destroy_matrix(U, N);

    return U_python;

}


/* -------------------------- KMeans -------------------------------------------- */

static PyObject* run_kmeans(PyObject *self, PyObject *args){
    PyObject* U_python;
    PyObject* centroids_python;

    double** U;
    double** centroids;

    int N;
    int K;
    int d;


    if (!PyArg_ParseTuple(args,"OOiii", &U_python, &centroids_python, &N, &K, &d)){
        printf("parsing failed\n");
        return NULL;
    }
    if ((!PyList_Check(U_python)) || (!PyList_Check(centroids_python))){
        printf("not a list\n");
        return NULL;
    }

    U = allocate_matrix(N, K);
    matrix_from_Python_to_C(U, U_python, N, K);

    centroids = allocate_matrix(K, d);
    matrix_from_Python_to_C(centroids, centroids_python, K, d);

    kmeans(U, centroids, N, K, d);

    centroids_python = matrix_from_C_to_Python(centroids, K, d);

    destroy_matrix(U, N);
    destroy_matrix(centroids, K);

    return centroids_python;
}




static PyMethodDef capiMethods[] = {
        {"compute_W",(PyCFunction) compute_W, METH_VARARGS, PyDoc_STR("computes W")},
        {"compute_D",(PyCFunction) compute_D, METH_VARARGS, PyDoc_STR("computes D")},
        {"compute_Lnorm",(PyCFunction) compute_Lnorm, METH_VARARGS, PyDoc_STR("compute L_norm")},
        {"run_jacobi_compute_V",(PyCFunction) run_jacobi_compute_V, METH_VARARGS, PyDoc_STR("computes eigenvectors")},
        {"run_jacobi_compute_A",(PyCFunction) run_jacobi_compute_A, METH_VARARGS, PyDoc_STR("compute eigenvalues")},
        {"run_compute_normalized_U",(PyCFunction) run_compute_normalized_U, METH_VARARGS, PyDoc_STR("finds K, computes U and normalizes")},
        {"run_kmeans",(PyCFunction) run_kmeans, METH_VARARGS, PyDoc_STR("compute last centroids")},
        {NULL, NULL, 0, NULL}
    };

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "spkmeansmodule", NULL, -1, capiMethods};


PyMODINIT_FUNC PyInit_spkmeansmodule(void){
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m){
        return NULL;
    }
    return m;
}