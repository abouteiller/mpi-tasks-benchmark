cmake_minimum_required(VERSION 3.12)
project(irate)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
add_executable(mpirate mpirate.c common.c)
target_link_libraries(mpirate PRIVATE MPI::MPI_C OpenMP::OpenMP_C)
