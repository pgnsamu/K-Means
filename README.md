
# k-Means clustering algorithm

University Project of @AlexStingaci and @pgnsamu

Read the handout and use the sequential code as reference to study.
Use the other source files to parallelize with the proper programming model.

Use the input files in the test_files directory for your first tests.

## Table of Contents
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Testing](#testing)
- [License](#license)

## Installation

Step-by-step instructions on how to get the development environment running.

```bash
# Example command
git clone https://github.com/pgnsamu/KMEANS.git
cd KMEANS
```

## Configuration
Edit the first lines in the Makefile to set your preferred compilers and flags
for both the sequential code and for each parallel programming model: 
OpenMP, MPI, and CUDA.

To see a description of the Makefile options execute:
```bash
make help 
```

## Usage
There are 4 files that will be compiled whenever you use 
```bash
make all
```
After that you will need to run the executable of KMEANS.c and set the output in output2D.inp using this method then you can check if the results are fine with the sequential program 
```bash
./KMEANS_seq test_files/input100D2.inp 1000 3 50 0.01 output2D.inp
```
For running parallel programs

OMP
```bash
./KMEANS_omp test_files/input100D2.inp 1000 3 50 0.01 output2D2.inp
```
MPI
```bash
mpirun -np 4 ./KMEANS_mpi test_files/input100D2.inp 1000 3 50 0.01 output2D2.inp
```

For MACOS
```bash
export OMPI_CC=gcc-14   # For Open MPI
export MPICH_CC=gcc-14  # For MPICH
```

MPI+OMP
-
MAKEFILE
```bash
mpicc -fopenmp -o KMEANS_mpi_omp KMEANS_mpi_omp.c -lm
```
run
```bash
mpirun -np 4 ./KMEANS_mpi_omp test_files/input100D2.inp 1000 3 50 0.01 output2D.inp
```


Unlimited Test
```bash
ulimit -s unlimited
```


If you wanna run the program with debug variables you can do
```bash
make debug 
```

## Testing
We are planning to use github autotest, but for now before pushing you need to run tests by yourself.

The test scripts are confronto.py and extractTime.py. Before running confronto.py you need to modify num_esecuzioni default value that is 10
```python
num_esecuzioni = 10
```

execute the sequential program (with the command above), then you have to run the program modified (still using the command above) and last you have to run confronto.py
```bash
python confronto.py
```
this will check if the result of the parallel program are the same of the sequential program and save time of each execution in tempi_esecuzione.txt

## License

EduHPC 2023: Peachy assignment

(c) 2022 Diego García-Álvarez, Arturo Gonzalez-Escribano
Group Trasgo, Universidad de Valladolid (Spain)
