// module load openmpi/gcc/4.0.5
// module load mpi/openmpi-x86_64 gcc-8.1
// mpic++ -std=c++11 -g k_means.cpp -o k_means
// mpirun -np 20 valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes -s ./k_means 4
// mpirun -np 20 valgrind --suppressions=$PREFIX/share/openmpi/openmpi-valgrind.supp ./k_means 4
// mpirun -np 20 ./k_means 4

#include <stdio.h>
#include <cstdlib>
#include <cmath>
#include <mpi.h>
#include <iostream>
#include <vector>

int maxiter = 10000;
double convergence_bar = 1e-5;

double get_dist_squared(double i1, double j1, double i2, double j2)
{
    // calculate Euclidean distance between (i1, j1) and (i2, j2)
    return (i1 - i2) * (i1 - i2) + (j1 - j2) * (j1 - j2);
}

int cluster_match(const std::vector<std::vector<double>> &k_centers, double dow, double hod)
{
    int closest_index = 0;
    double shortest_dist_squared = get_dist_squared(dow, hod, k_centers[0][0], k_centers[0][1]);
    for (int i = 1; i < k_centers.size(); i++)
    {
        double new_dist_squared = get_dist_squared(dow, hod, k_centers[i][0], k_centers[i][1]);
        if (new_dist_squared < shortest_dist_squared)
        {
            shortest_dist_squared = new_dist_squared;
            closest_index = i;
        }
    }
    return closest_index;
}

std::vector<std::vector<double>> sequential_k_means(std::vector<std::vector<double>> old_k_centers, const std::vector<int> &dataset, int N)
{
    int k = old_k_centers.size();
    bool converged;
    int iter = 0;
    do
    {
        iter++;
        std::vector<int> num_members;
        std::vector<std::vector<double>> new_k_centers(old_k_centers);
        for (int i = 0; i < k; i++)
        {
            num_members.push_back(1);
        }

        for (int i = 0; i < N; i++)
        {
            int closest_index = cluster_match(old_k_centers, dataset[2 * i], dataset[2 * i + 1]);
            // record how the center should be updated
            new_k_centers[closest_index][0] = (new_k_centers[closest_index][0] * num_members[closest_index] + dataset[2 * i]) / (num_members[closest_index] + 1);
            new_k_centers[closest_index][1] = (new_k_centers[closest_index][1] * num_members[closest_index] + dataset[2 * i + 1]) / (num_members[closest_index] + 1);
            num_members[closest_index]++;
        }
        // calculate by how much each center has moved. Declare convergence if none of the centers moved more than 1e-2
        converged = true;
        for (int i = 0; i < k; i++)
        {
            if (get_dist_squared(old_k_centers[i][0], old_k_centers[i][1], new_k_centers[i][0], new_k_centers[i][1]) > convergence_bar)
            {
                converged = false;
                break;
            }
        }
        old_k_centers = new_k_centers;
        /*
        printf("After iteration # %d, the centers of the k clusters identified by the sequential approach are:\n", iter);
        for (int i = 0; i < k; i++)
        {
            printf("(%f, %f)\n", old_k_centers[i][0], old_k_centers[i][1]);
        }
        */
    } while (not converged and iter <= maxiter);
    printf("The sequential k means takes %d iterations.\n", iter);
    return old_k_centers;
}

std::vector<std::vector<double>> parallel_k_means(std::vector<std::vector<double>> old_k_centers, const std::vector<int> &dataset, int N)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int total;
    MPI_Comm_size(MPI_COMM_WORLD, &total);

    int elements_per_proc = floor(double(N) / total);
    std::vector<int> local_dataset(elements_per_proc * 2);

    bool converged = true;
    int iter = 0;
    do
    {
        iter++;
        MPI_Scatter(dataset.data(), 2 * elements_per_proc, MPI_INT, local_dataset.data(),
                    2 * elements_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<int> num_members;
        std::vector<std::vector<double>> new_k_centers(old_k_centers);
        int k = old_k_centers.size();
        for (int i = 0; i < k; i++)
        {
            num_members.push_back(1);
        }

        for (int i = 0; i < elements_per_proc; i++)
        {
            int closest_index = cluster_match(old_k_centers, local_dataset[2 * i], local_dataset[2 * i + 1]);
            // record how the center should be updated
            new_k_centers[closest_index][0] = (new_k_centers[closest_index][0] * num_members[closest_index] + local_dataset[2 * i]) / (num_members[closest_index] + 1);
            new_k_centers[closest_index][1] = (new_k_centers[closest_index][1] * num_members[closest_index] + local_dataset[2 * i + 1]) / (num_members[closest_index] + 1);
            num_members[closest_index]++;
        }

        // calculuate displacement in each center within each thread and allreduce
        std::vector<double> local_displacement0, local_displacement1;
        std::vector<double> global_displacement0(k), global_displacement1(k);
        for (int i = 0; i < k; i++)
        {
            local_displacement0.push_back(new_k_centers[i][0] - old_k_centers[i][0]);
            local_displacement1.push_back(new_k_centers[i][1] - old_k_centers[i][1]);
        }
        MPI_Allreduce(local_displacement0.data(), global_displacement0.data(), k, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_displacement1.data(), global_displacement1.data(), k, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        // calculate by how much each center has moved. Declare convergence if none of the centers moved more than 1e-2. broadcast the convergence result to all threads
        converged = true;

        for (int i = 0; i < k; i++)
        {
            new_k_centers[i][0] = old_k_centers[i][0] + global_displacement0[i] / total;
            new_k_centers[i][1] = old_k_centers[i][1] + global_displacement1[i] / total;
            if (get_dist_squared(old_k_centers[i][0], old_k_centers[i][1], new_k_centers[i][0], new_k_centers[i][1]) > convergence_bar)
            {
                converged = false;
            }
        }
        old_k_centers = new_k_centers;
        /*
        if (!rank)
        {
            printf("After iteration # %d, the centers of the k clusters identified by the parallel approach are:\n", iter);
            for (int i = 0; i < k; i++)
            {
                printf("(%f, %f)\n", old_k_centers[i][0], old_k_centers[i][1]);
            }
        }
        */

    } while (not converged and iter <= maxiter);
    if (rank == 0)
        printf("The parallel k means takes %d iterations.\n", iter);
    return old_k_centers;
}

int main(int argc, char **argv)
{
    // read data from the kaggle dataset and store the order_dow and order_hour_of_day respectively in order into dow and hod
    // int N = 2019501; // change this if some data have missing order_dow or order_hour_of_day

    if (argc < 2)
    {
        printf("Usage: mpirun -#processes ./k_means k \n");
        abort();
    }

    int N = 2000000;
    std::vector<int> dataset;
    for (int i = 0; i < N; i++)
    {
        dataset.push_back(i % 7);
        dataset.push_back(i % 24);
    }

    // initialize k centers
    int k = atoi(argv[1]);
    std::vector<std::vector<double>> initial_k_centers;
    for (int i = 0; i < k; i++)
    {
        int index;
        bool far_enough;
        do
        {
            index = rand() % N;
            far_enough = true;
            for (int j = 0; j < i; j++)
            {
                if (get_dist_squared(initial_k_centers[j][0], initial_k_centers[j][1], dataset[2 * index], dataset[2 * index + 1]) <= 1)
                {
                    far_enough = false;
                    break;
                }
            }
        } while (not far_enough);
        // printf("center %d is (%d, %d)\n", i, dataset[2 * index], dataset[2 * index + 1]);
        initial_k_centers.push_back({double(dataset[2 * index]), double(dataset[2 * index + 1])});
    }
    // printf("managed to initialize k centers\n");

    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int total;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    double tt = MPI_Wtime();
    std::vector<std::vector<double>> sequential_k_means_centers = sequential_k_means(initial_k_centers, dataset, N);
    tt = MPI_Wtime() - tt;
    if (!rank)
    {
        printf("Sequential k means takes %e s\n", tt);
        printf("The centers of the k clusters identified by the sequential approach are:\n");
        for (int i = 0; i < k; i++)
        {
            printf("(%f, %f)\n", sequential_k_means_centers[i][0], sequential_k_means_centers[i][1]);
        }
    }

    tt = MPI_Wtime();
    std::vector<std::vector<double>> parallel_k_means_centers = parallel_k_means(initial_k_centers, dataset, N);
    tt = MPI_Wtime() - tt;

    if (!rank)
    {
        printf("parallel k means takes %e s\n", tt);
        printf("The centers of the k clusters identified by the parallel approach are:\n");
        for (int i = 0; i < k; i++)
        {
            printf("(%f, %f)\n", parallel_k_means_centers[i][0], parallel_k_means_centers[i][1]);
        }
    }
    MPI_Finalize();
}