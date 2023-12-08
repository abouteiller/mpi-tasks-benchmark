cmake_minimum_required(VERSION 3.19)
project(mpi-tasks-benchmark)

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    option(GIT_SUBMODULE "Fetch the submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_FAILED)
        if(GIT_SUBMOD_FAILED)
            message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_FAILED}, please checkout submodules")
        endif()
        add_subdirectory(parsec)
        find_package(parsec-submodule NO_DEFAULT_PATH PATHS ${CMAKE_CURRENT_BINARY_DIR}/parsec REQUIRED)
    endif()
endif()

add_subdirectory(irregular-gemm-parsec)



