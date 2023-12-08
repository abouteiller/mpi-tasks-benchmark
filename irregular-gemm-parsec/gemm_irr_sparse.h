#ifndef GEMM_SPARSE_H
#define GEMM_SPARSE_H
/*
 * Copyright (c) 2020-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT
 *
 */

#include "parsec.h"
#include <math.h>
#include <complex.h>

#include "irr_bs_tm.h"

#define DGEMM_IRR_SPARSE_GENB 102

#if defined(IGGOP_HAVE_HIP)
#include <hipblas.h>
#endif /* defined(IGGOP_HAVE_HIP) */
#if defined(IGGOP_HAVE_CUDA)
#include <cublas.h>
#endif /* defined(IGGOP_HAVE_HIP) */

/**
 * A full execution plan for the irregular GEMM block-sparse.
 */
typedef struct gemm_irr_sparse_plan_s gemm_irr_sparse_plan_t;

int gemm_irr_sparse_first_gemm(gemm_irr_sparse_plan_t *plan, int m, int n);
int gemm_irr_sparse_next_gemm(gemm_irr_sparse_plan_t *plan, int m, int n, int k);
int gemm_irr_sparse_prev_gemm(gemm_irr_sparse_plan_t *plan, int m, int n, int k);
int gemm_irr_sparse_nb_gemm(gemm_irr_sparse_plan_t *plan, int m, int n);
int gemm_irr_sparse_gemm_k(gemm_irr_sparse_plan_t *plan, int m, int n, int ik);

/**
 * Return true iff (p, q) is my rank
 */
int gemm_irr_sparse_plan_my_rank(const gemm_irr_sparse_plan_t *plan, int p, int q);

/**
 * What is the last col index for which rank r needs to instantiate an element of B
 */
int gemm_irr_sparse_plan_B_last_col(const gemm_irr_sparse_plan_t *plan, int p, int q);

/**
 * Given col index in, rank r instantiates B(*, gemm_irr_sparse_plan_B_col(plan, p, q, in))
 */
int gemm_irr_sparse_plan_B_col(const gemm_irr_sparse_plan_t *plan, int p, int q, int in);

/**
 * What is the last row index for which rank (p, q) needs to instantiate B(*, n)
 */
int gemm_irr_sparse_plan_B_last_row(const gemm_irr_sparse_plan_t *plan, int p, int q, int n);

/**
 * Given row index ik, on rank r, rank r instantiates B(gemm_irr_sparse_plan_B_row(plan, p, q, n, ik), n)
 */
int gemm_irr_sparse_plan_B_row(const gemm_irr_sparse_plan_t *plan, int p, int q, int n, int ik);

/**
 * Describe the plan to the FILE
 */
void gemm_irr_sparse_genB_describe_plan(const gemm_irr_sparse_plan_t *plan, FILE *f);

/**
 * Dump what GEMM will execute when and where in FILE
 */
void gemm_irr_sparse_genB_output_plan(const gemm_irr_sparse_plan_t *plan, FILE *f);

/**
 * Gets on what rank B(k, *) will be used for GEMMs
 *   (assumes that B is 1D-cyclic distributed amongst ranks and GPUs)
 */
int gemm_irr_sparse_genB_rank(gemm_irr_sparse_plan_t *plan, int k);

/**
 * Gets on what GPU B(k, *) will be used for GEMMs
 *   (assumes that B is 1D-cyclic distributed amongst ranks and GPUs)
 */
int gemm_irr_sparse_genB_gpu(gemm_irr_sparse_plan_t *plan, int k);

/**
 * Gets during which column-phase B(k, *) will be used for GEMMS
 *   (assumes that B is 1D-cyclic distributed amongst ranks and GPUs,
 *    and assumes that each column of B is only generated once,
 *    all tiles of the column being generated at the same time)
 */
int gemm_irr_sparse_genB_column_phase(gemm_irr_sparse_plan_t *plan, int k);

/**
 * Returns the index of the last column-phase for local rank on GPU g
 */
int gemm_irr_sparse_max_column_phase(gemm_irr_sparse_plan_t *plan, int g);

/**
 * Returns the number of columns in column phase cp
 */
int gemm_irr_sparse_nb_columns_of_column_phase(gemm_irr_sparse_plan_t *plan, int g, int cp);

/**
 * Returns the i-th column in column phase cp
 */
int gemm_irr_sparse_column_of_column_phase(gemm_irr_sparse_plan_t *plan, int g, int cp, int i);

/**
 * Returns the index of the last row-phase in column-phase cp of GPU g on local rank
 */
int gemm_irr_sparse_max_row_phase(gemm_irr_sparse_plan_t *plan, int g, int cp);

/**
 * Returns the index of the last GEMM in row-phase rp, belonging to column-phase cp
 * that runs on GPU g of local rank (see example below)
 */
int gemm_irr_sparse_max_gemm_of_row_phase(gemm_irr_sparse_plan_t *plan, int g, int cp, int rp);

/**
 * Given an index of GEMM, ig, in row-phase rp belonging to column-phase cp that runs on
 * GPU g of local rank, returns the actual row of that GEMM (see example below)
 */
int gemm_irr_sparse_gemm_m_of_row_phase(gemm_irr_sparse_plan_t *plan, int g, int cp, int rp, int ig);

/**
 * Given an index of GEMM, ig, in row-phase rp belonging to column-phase cp that runs on
 * GPU g of local rank, returns the actual column of that GEMM (see example below)
 */
int gemm_irr_sparse_gemm_n_of_row_phase(gemm_irr_sparse_plan_t *plan, int g, int cp, int rp, int ig);

/**
 * Given an index of GEMM, ig, in row-phase rp belonging to column-phase cp that runs on
 * GPU g of local rank, returns the actual k of that GEMM (see example below)
 */
int gemm_irr_sparse_gemm_k_of_row_phase(gemm_irr_sparse_plan_t *plan, int g, int cp, int rp, int ig);

/* Example: consider that during row-phase 3 of column-phase 1 running on GPU 0 of local rank,
 *  the following GEMMs are scheduled:
 *     { GEMM(16, 12, 8), GEMM(16, 12, 9), GEMM(16, 12, 10),
 *       GEMM(17, 12, 7), GEMM(17, 12, 32), GEMM(17, 12, 4), GEMM(17, 12, 66), GEMM(17, 12, 11) }
 *   gemm_irr_sparse_max_gemm_of_row_phase(plan, 0, 1, 3) = 7
 *   gemm_irr_sparse_gemm_m_of_row_phase(plan, 0, 1, 3, 3) = 17
 *   gemm_irr_sparse_gemm_k_of_row_phase(plan, 0, 1, 3, 3) = 7
 */

#if 0
/**
 * Create a dummy random plan to test */
gemm_irr_sparse_plan_t *gemm_irr_sparse_create_random_plan(irr_bs_tm_t *A, irr_bs_tm_t *Bgen, int my_rank, int P, int world_size, int nb_gpus);

/**
 * Create a simple 'balanced' plan to test:
 *   each row-phase aim at holding a block of a x b gemms, with the last phase of each column phase holding less
 */
gemm_irr_sparse_plan_t *gemm_irr_sparse_create_simple_plan(irr_bs_tm_t *A, irr_bs_tm_t *Bgen, int my_rank, int P, int world_size, int nb_gpus, int a, int b);
#endif

/**
 * Create a 'smart plan...
 *  Sort things, and try to load balance between the GPUs
 *  Maximize ram_per_gpu usage (can exceed ram_per_gpu per the largest GEMM in the worst case)
 *     ram_per_gpu is in number of elements, not in bytes
 *  Try to put b elements of the column in parallel in the same row phase
 */
gemm_irr_sparse_plan_t *gemm_irr_sparse_create_smart_plan(irr_bs_tm_t *A, irr_bs_tm_t *Bgen, int my_rank, int P, int world_size, int nb_gpu, size_t ram_per_gpu, size_t gpu_alloc_grain, double part_for_B, int b, int RL, int CL, int GL, parsec_hash_table_t *gemm_per_mn);

/**
 * Destroy a plan
 */
void gemm_irr_sparse_destroy_plan(gemm_irr_sparse_plan_t *plan);

/**
 * Given a GEMM(m, n, k), what row-rank executes it
 */
int gemm_irr_sparse_row_rank_of_gemm(gemm_irr_sparse_plan_t *plan, int m, int n, int k);

/**
 * Given a GEMM(m, n, k), what col-rank executes it
 */
int gemm_irr_sparse_col_rank_of_gemm(gemm_irr_sparse_plan_t *plan, int m, int n, int k);

/**
 * Given a GEMM(m, n, k), what GPU executes it
 */
int gemm_irr_sparse_gpu_of_gemm(gemm_irr_sparse_plan_t *plan, int m, int n, int k);

/**
 * Given a GEMM(m, n, k), what column-phase does it belong to
 *   (implicitely, what column-phase of rank r and GPU g)
 */
int gemm_irr_sparse_column_phase_of_gemm(gemm_irr_sparse_plan_t *plan, int m, int n, int k);

/**
 * Given a GEMM(m, n, k), what row-phase does it belong to
 *   (implicitely, what row-phase of rank r, GPU g, and column-phase cp)
 */
int gemm_irr_sparse_row_phase_of_gemm(gemm_irr_sparse_plan_t *plan, int m, int n, int k);

/**
 * Compute the tile size of a tile of C
 */
int gemm_irr_sparse_C_tile_count(const irr_bs_tm_t *descA, const irr_bs_tm_t *genB, int m, int n);


parsec_taskpool_t*dgemm_irr_sparse_New( double alpha, irr_bs_tm_t *A,
                                        irr_bs_tm_t *Bgen, double beta,
                                        irr_bs_tm_t *C, int P, int Q,
                                        gemm_irr_sparse_plan_t *plan,
                                        int nb_gpus, int *dev_index,
                                        int RL, int CL, int GL,
                                        int show_progress );
void dgemm_irr_sparse_Destruct( parsec_taskpool_t *tp );


typedef struct gemm_irr_sparse_plan_gemms_at_mn_s {
    parsec_hash_table_item_t ht_item;
    int m;
    int n;
    int nb_k;
    int k[1];
} gemm_irr_sparse_plan_gemm_at_mn_t;

typedef struct gemm_irr_sparse_plan_gemm_s gemm_irr_sparse_plan_gemm_t;
typedef struct gemm_irr_sparse_plan_row_phase_s gemm_irr_sparse_plan_row_phase_t;
typedef struct gemm_irr_sparse_plan_column_s gemm_irr_sparse_plan_column_t;
typedef struct gemm_irr_sparse_plan_column_phase_s gemm_irr_sparse_plan_column_phase_t;
typedef struct gemm_irr_sparse_plan_gpu_s gemm_irr_sparse_plan_gpu_t;

struct gemm_irr_sparse_plan_gemm_s {
    gemm_irr_sparse_plan_row_phase_t *rp;
    int gemm_index;
    int m;
    int n;
    int k;
    int ik;
    parsec_hash_table_item_t ht_item;
};

struct gemm_irr_sparse_plan_row_phase_s {
    gemm_irr_sparse_plan_column_phase_t *cp;
    int rp_index;
    int nb_gemm;
    gemm_irr_sparse_plan_gemm_t *gemm;
};

struct gemm_irr_sparse_plan_column_s {
    gemm_irr_sparse_plan_column_phase_t *cp;
    int n;
    parsec_hash_table_item_t ht_item;
};

struct gemm_irr_sparse_plan_column_phase_s {
    gemm_irr_sparse_plan_gpu_t *gpu;
    int nb_col, col_size;
    gemm_irr_sparse_plan_column_t *col;
    int cp_index;
    int nb_rp;
    gemm_irr_sparse_plan_row_phase_t *rp;
};

struct gemm_irr_sparse_plan_gpu_s {
    gemm_irr_sparse_plan_t *plan;
    int gpu_index;
    int nb_cp;
    gemm_irr_sparse_plan_column_phase_t *cp;
};

struct gemm_irr_sparse_plan_s {
    irr_bs_tm_t *A;
    irr_bs_tm_t *Bgen;
    int p;
    int q;
    int *col_ranks;
    int *local_cols;
    int nb_local_cols;
    int *row_ranks;
    int *local_rows;
    int nb_local_rows;
    int nb_gpu;
    int64_t nb_local_gemms;
    int64_t nb_local_genB;
    gemm_irr_sparse_plan_gpu_t *gpu;
    parsec_hash_table_t cp_per_column;
    parsec_hash_table_t gemm_per_mnk;
    parsec_hash_table_t *gemm_per_mn;
};

typedef struct {
#if defined(IGGOP_HAVE_HIP)
    hipblasHandle_t hipblas_handle;
#endif /* IGGOP_HAVE_HIP */
#if defined(IGGOP_HAVE_CUDA)
    cublasHandle_t cublas_handle;
#endif /* IGGOP_HAVE_CUDA */
} handles_infokey_t;
void *create_infokey_handles(void *obj, void *_n);
void destroy_infokey_handles(void *obj, void *_n);

#if defined(IGGOP_HAVE_HIP)
#define ROCBLAS_CHECK_ERROR(STR, ERROR, CODE) \
    do { \
        rocblas_status __error = (rocblas_status) (ERROR); \
        if(rocblas_status_success != __error) { \
            parsec_warning( "%s:%d %s%s", __FILE__, __LINE__, \
                            (STR), rocblas_status_to_string(__error)); \
            CODE; \
        } \
    } while(0)

/* For some reason the error values are not the same... */
#define HIPBLAS_CHECK_ERROR(STR, ERROR, CODE) \
    do { \
        hipblasStatus_t __error = (hipblasStatus_t) (ERROR); \
        if(HIPBLAS_STATUS_SUCCESS != __error) { \
            parsec_warning( "%s:%d %s%s", __FILE__, __LINE__, \
                            (STR), hipblasStatusToString(__error)); \
            CODE; \
        } \
    } while(0)
#endif /* defined(IGGOP_HAVE_HIP) */

#endif
