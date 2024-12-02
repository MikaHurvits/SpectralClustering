# Spectral Clustering Module

This project implements **Spectral Clustering**, a powerful clustering technique leveraging eigenvalues of graph Laplacians for dimensionality reduction before applying k-means clustering. 

<img width="476" alt="image" src="https://github.com/user-attachments/assets/7ee3d13a-9a73-4492-99f9-52b3b43d15f1">


The project provides a Python interface for performing spectral clustering along with supporting computations.


## Features

- Compute Weighted Adjacency Matrix (WAM)
- Compute Diagonal Degree Matrix (DDG)
- Compute Normalized Graph Laplacian (Lnorm)
- Perform Eigen decomposition using the Jacobi algorithm
- Determine optimal cluster count \(k\) and normalized matrix \(U\)
- Implement k-means clustering for final cluster assignments

## Structure

- **`spkmeansmodule.c`**: Implements the core computational logic in C for efficiency and exposes it as a Python extension module.
- **`spkmeans.h`**: Header file defining functions and structures used in the C implementation.
- **`spkmeans.py`**: Provides a Python interface for interacting with the C module and a main program to execute clustering workflows.
- **`setup.py`**: Configuration script for building and installing the Python extension.

## Installation

### Prerequisites

- Python 3.x
- GCC or a compatible C compiler
- Required Python libraries: `numpy`, `pandas`, `invoke`

### Build Instructions

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```
2. Build and install the Python module:
   ```bash
   python setup.py install
   ```

## Usage

### Command-line Interface

Run the main program with:
```bash
python spkmeans.py <k> <goal> <input_filename>
```

#### Parameters:

- `<k>`: The number of clusters. Specify `0` to let the program determine \(k\) automatically.
- `<goal>`: Specifies the computational goal:
  - `spk`: Spectral clustering
  - `wam`: Weighted adjacency matrix
  - `ddg`: Diagonal degree matrix
  - `lnorm`: Normalized graph Laplacian
  - `jacobi`: Eigen decomposition
- `<input_filename>`: Path to the input file (CSV or TXT format) containing the data points.

#### Example:

```bash
python spkmeans.py 3 spk data.csv
```

### Output

The output varies based on the chosen goal. For example:
- `wam`, `ddg`, and `lnorm` produce respective matrices.
- `spk` outputs the cluster centroids and the initial indices used for clustering.

### Library Usage

Import and use the provided module in Python scripts:

```python
import spkmeansmodule as spk

# Example: Compute Weighted Adjacency Matrix
W = spk.compute_W(data_points, N, d)
```

## File Formats

### Input

- Files should contain data points, each on a new line.
- Supported formats: CSV and TXT.

### Output

- Matrices and vectors are printed in a comma-separated format with up to 4 decimal points.


