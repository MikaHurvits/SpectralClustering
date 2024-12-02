import sys
import numpy as np
import pandas as pd
from enum import Enum
import csv
import spkmeansmodule as spectral_clustering


invalid_input = "Invalid Input!"
error_occured = "An Error Has Occured"

class Goal(Enum):
    spk = 1
    wam = 2
    ddg = 3
    lnorm = 4
    jacobi = 5


def is_csv(filename): #check if the file is of type csv
    identifier = filename[-3:]
    return identifier == 'csv'


def parse_txt_file(filename):
    data_points = []  
    with open(filename, 'r') as f:
        for line in f:
            point = line.split(",")
            data_points.append([float(x) if x[0] != '-' else -1*float(x[1:]) for x in point])
    return data_points


def parse_csv_file(filename):
    data_points = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            data_points.append([float(x) for x in row])
    return data_points


def print_indices(indices):
    str_indices = [str(ind) for ind in indices]
    print(",".join(str_indices))


def print_matrix(centroids): 
    str_centroids = [[str(format(x,'.4f')) if (x >= 0 or x <= -0.00005) else "0.0000" for x in centroids[i]] for i in range(len(centroids))] # eliminate situations where -0 is printed
    for i in range(len(centroids)):
        print(",".join(str_centroids[i]))

def print_transposed_matrix(matrix):
    matrix = np.array(matrix)
    matrix = matrix.T
    matrix = [row.tolist() for row in matrix]
    print_matrix(matrix)

def print_diagonal_matrix(vector):
    N = len(vector)
    mat = [[0 for j in range(N)] for i in range(N)]
    for i in range(N):
        mat[i][i] = vector[i]
    print_matrix(mat)


def print_diagonal_matrix_as_vector(matrix):
    N = len(matrix)
    diag = [str(format(matrix[i][i], '.4f')) if (matrix[i][i] >=0 or matrix[i][i] <= -0.0005) else "0.0000" for i in range(N)] # eliminate situations where -0 is printed
    print(",".join(diag))


def K_input_checks(k):
    if not k.isdigit():
        print(invalid_input)
        sys.exit(0)

    k = int(k)

    if k < 0:
        print(invalid_input)
        sys.exit(0)

    if k != k // 1:
        print(invalid_input)
        sys.exit(0)
    
    return k


def kmeans_pp_initiallization(data, N, K):
    # kmeans++ for getting initial centroids 
    np.random.seed(0)
    rand_ind = np.random.choice([i for i in range(N)])
    initial_centroids = [data[rand_ind]]
    initial_inds = [rand_ind]
    for z in range(K-1):
        probs = np.zeros(N)
        for i in range(N):
            x_i = data[i]
            dists = [np.linalg.norm(x_i - mu_j) ** 2 for mu_j in initial_centroids]
            D_i = min(dists)
            probs[i] = D_i
        probs /= np.sum(probs)
        ind = np.random.choice(N, 1, p=probs)[0]

        next_mu = data[ind]
        initial_centroids.append(next_mu)
        initial_inds.append(ind)

    initial_centroids = [mu.tolist() for mu in initial_centroids]  
    return initial_centroids, initial_inds



def main():
    argv = sys.argv
    argc = len(argv)

    if argc != 4:
        print(invalid_input)
        sys.exit(0)

    # parsing K
    k = argv[1]
    K = K_input_checks(k)

    # parsing goal
    input_goal_str = argv[2]
    input_goal = None
    
    if input_goal_str == "spk":
        input_goal = Goal.spk
    
    elif input_goal_str == "wam":
        input_goal = Goal.wam
    
    elif input_goal_str == "ddg":
        input_goal = Goal.ddg

    elif input_goal_str == "lnorm":
        input_goal = Goal.lnorm
    
    elif input_goal_str == "jacobi":
        input_goal = Goal.jacobi
    
    else:
        print(invalid_input)
        sys.exit(0)


    #parsing file name 
    file_name = argv[3]

    if is_csv(file_name):
        data = np.array(parse_csv_file(file_name))
    else:
        data = np.array(parse_txt_file(file_name))

    N, d = data.shape
    data = [data[i].tolist() for i in range(N)]

    if K >= N:
        print(invalid_input)
        sys.exit(0)
    
    # code flow:

    if input_goal == Goal.jacobi:
        V = spectral_clustering.run_jacobi_compute_V(data, N)
        A = spectral_clustering.run_jacobi_compute_A(data, N)
        print_diagonal_matrix_as_vector(A)
        print_transposed_matrix(V)
        sys.exit(0)

    W = spectral_clustering.compute_W(data, N, d)
    D = spectral_clustering.compute_D(data, N, d)

    if input_goal == Goal.wam:
        print_matrix(W)
        sys.exit(0)
    
    if input_goal == Goal.ddg:
        print_diagonal_matrix(D)
        sys.exit(0)

    L_norm = spectral_clustering.compute_Lnorm(W, D, N)

    if input_goal == Goal.lnorm:
        print_matrix(L_norm)
        sys.exit(0)
    
    V = spectral_clustering.run_jacobi_compute_V(L_norm, N)
    A = spectral_clustering.run_jacobi_compute_A(L_norm, N)

    # create E and ID
    E = [A[i][i] for i in range(N)]
    ID = [i for i in range(N)]

    
    # find normalized U 
    U = spectral_clustering.run_compute_normalized_U(E, ID, V, N, K)
    K = len(U[0])
    

    U_numpy = np.array(U)
    initial_centroids, initial_inds = kmeans_pp_initiallization(U_numpy, N, K)
    centroids = spectral_clustering.run_kmeans(U, initial_centroids, N, K, K)

    print_indices(initial_inds)
    print_matrix(centroids)


if __name__ == "__main__":
    main()

