# HPC-Project-Parallel-K-Means
Spring 2023 High Performance Computing course (Georg Stadler)

Input Dataset - [Supermarket dataset for predictive marketing 2023](https://www.kaggle.com/datasets/hunter0007/ecommerce-dataset-for-predictive-marketing-2023?datasetId=2772962&sortBy=dateRun&tab=profile)

Steps to run the program on NYU's Greene HPC servers:
1. Login to Greene with ```ssh <netID>@greene.hpc.nyu.edu```
2. Create a new directory to store the project files, eg: ```mkdir project```
3. ```cd project```
4. Clone the project from this github repo into your new project directory: ```https://github.com/metallicmay/HPC-Project-Parallel-K-Means.git```
5. Load gcc module to compile main program: ```module load openmpi/gcc/4.0.5```
6. Compile the main program: ```mpic++ -std=c++11 -g k_means.cpp -o k_means```
7. Once kmeans executable is created, run the sbatch file: ```sbatch k_means.sbatch```
8. Check progress of the job using ```squeue -u <netID>```
9. Once it is done, read the output file: ```cat k_means.out```
10. Here is a sample output:
  ```Initialization of k centers takes 0.000022 seconds.
Initialization of k centers takes 0.000016 seconds.
Initialization of k centers takes 0.000017 seconds.
Initialization of k centers takes 0.000026 seconds.
Initialization of k centers takes 0.000020 seconds.
Sequential k means takes 9.890023e+00 s, 24 iterations
The flop rate is 0.367552 GFlops /s
The bandwith is 0.035938 GB/s
The centers of the k clusters identified by the sequential approach are:
(6.500588, 2.710744, 13.564159, 4.273994)
(71.791704, 2.842772, 13.023796, 9.915726)
(41.450571, 2.780801, 13.152742, 9.961178)
(22.223521, 2.730122, 13.381494, 9.845230)
(6.583953, 2.730925, 13.529109, 16.154328)
Parallel k means takes 2.330375e+00 s, 20 iterations
The flop rate is 1.299899GFlops /s
The bandwith is 0.152521 GB/s
The centers of the k clusters identified by the parallel approach are:
(5.843857, 2.743931, 17.069232, 9.862713)
(70.226851, 2.832166, 13.032586, 9.919858)
(39.498838, 2.781942, 13.166429, 9.964459)
(20.086927, 2.719495, 13.397795, 9.941531)
(5.877426, 2.701545, 10.063363, 9.966399)
