/*
 * Copyright (c) 2020-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT
 *
 * @generated d Thu Dec  7 16:50:48 2023
 *
 */

#include <errno.h>
#include <string.h>
#include <time.h>

#include "parsec.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "irr_bs_tm.h"
#include "gemm_irr_sparse.h"
#include "dgemm_irr_sparse_genB.h"

#if defined(IGGOP_HAVE_CUDA)
#include "parsec/mca/device/cuda/device_cuda_internal.h"
#endif /* defined(IGGOP_HAVE_CUDA) */
#if defined(IGGOP_HAVE_HIP)
#include "parsec/mca/device/hip/device_hip_internal.h"
#endif /* defined(IGGOP_HAVE_HIP) */

#include "parsec/utils/mca_param.h"
static parsec_matrix_block_cyclic_t TrivDist;
static int TrivDistInitialized = 0;

static parsec_data_t *TrivDist_data_of(parsec_data_collection_t *d, ...)
{
    (void)d;
    assert(0);
    return NULL;
}

static parsec_data_t *TrivDist_data_of_key(parsec_data_collection_t *d, parsec_data_key_t key)
{
    (void)d;
    (void)key;
    assert(0);
    return NULL;
}

/**
 *******************************************************************************
 *
 *  dgemm_irr_sparse_New - Generates the taskpool that performs one of the following
 *  matrix-matrix operations for irregular sparse tiling,
 *
 *    \f[ C = \alpha [op( A )\times op( B )] + \beta C,
 *
 *  where op( X ) is one of
 *
 *    op( X ) = X  or op( X ) = X' or op( X ) = g( X' )
 *
 *  alpha is scalar, and A, B and C are matrices, with op( A )
 *  an m by k matrix, op( B ) a k by n matrix and C an m by n matrix.
 *
 *******************************************************************************
 *
 * @param[in] alpha
 *          alpha specifies the scalar alpha
 *
 * @param[in] A
 *          Descriptor of the distributed matrix A.
 *
 * @param[in] B
 *          Descriptor of the distributed matrix B.
 *
 * @param[in] beta
 *          beta specifies the scalar beta
 *
 * @param[out] C
 *          Descriptor of the distributed matrix C.
 *          On exit, the data described by C contain the matrix (
 *          alpha*op( A )*op( B ) )
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The parsec taskpool describing the operation that can be
 *          enqueued in the runtime with parsec_enqueue(). It, then, needs to be
 *          destroy with dgemm_irr_sparse_Destruct();
 ******************************************************************************/
parsec_taskpool_t*
dgemm_irr_sparse_New( double alpha, irr_bs_tm_t *A,
                      irr_bs_tm_t *B,
                      double beta,
                      irr_bs_tm_t *C,
                      int P, int Q,
                      gemm_irr_sparse_plan_t *plan,
                      int nb_gpus, int *dev_index,
                      int RL, int CL, int GL,
                      int show_progress)
{
    parsec_taskpool_t* dgemm_handle = NULL;
    parsec_arena_datatype_t *adt;

    if( TrivDistInitialized == 0 ) {
        TrivDistInitialized = 1;
        assert(A->grid.rank < P*Q);
        parsec_matrix_block_cyclic_init(&TrivDist, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,
                                  A->grid.rank,
                                  1,   1,   /* Tile size */
                                  P,   Q,   /* Global matrix size (what is stored)*/
                                  0,   0,   /* Staring point in the global matrix */
                                  P,   Q,   /* Submatrix size (the one concerned by the computation */
                                  P,   Q,   /* Process grid*/
                                  1,   1,   /* Cyclicity */
                                  0,   0    /* Origin of cycle for process grid */ );
        TrivDist.super.super.data_of = TrivDist_data_of;
        TrivDist.super.super.data_of_key = TrivDist_data_of_key;
        TrivDist.super.super.key_base = strdup("TrivDist");
        TrivDist.super.super.key_dim = strdup("");
#if defined(PARSEC_DEBUG_VERBOSE)
        int r = 0;
        for(int p = 0; p < P; p++) {
            for(int q = 0; q < Q; q++) {
                assert( r == TrivDist.super.super.rank_of((parsec_data_collection_t*)&TrivDist, p, q) );
                r++;
            }
        }
#endif
    }
    assert(P == TrivDist.grid.rows);
    assert(Q == TrivDist.grid.cols);

    parsec_dgemm_irr_sparse_genB_taskpool_t *handle =
        parsec_dgemm_irr_sparse_genB_new(DGEMM_IRR_SPARSE_GENB, alpha, beta,
                                         A,
                                         B,
                                         C,
                                         &TrivDist,
                                         plan,
                                         nb_gpus, dev_index);
    adt = &handle->arenas_datatypes[PARSEC_dgemm_irr_sparse_genB_DEFAULT_ADT_IDX];
    handle->_g_RL = RL;
    handle->_g_CL = CL;
    handle->_g_GL = GL;
    if(show_progress) {
        fprintf(stderr, "Showing progress on rank %d\n", A->grid.rank);
        handle->_g_flops = 0;
    } else {
        handle->_g_flops = -1;
    }
    dgemm_handle = (parsec_taskpool_t*)handle;
    handle->_g_handles_infokey = parsec_info_lookup(&parsec_per_stream_infos, "GPU::HANDLES", NULL);

    parsec_datatype_t mtype;
    parsec_type_create_contiguous(1, parsec_datatype_double_t, &mtype);

    parsec_arena_datatype_construct(adt, sizeof(double),
                                    PARSEC_ARENA_ALIGNMENT_SSE,
                                    mtype);

    return dgemm_handle;
}

/**
 *******************************************************************************
 *  dgemm_irr_sparse_Destruct - Free the data structure associated to an taskpool
 *  created with dgemm_irr_sparse_New().
 *
 *******************************************************************************
 *
 * @param[in,out] tp
 *          On entry, the taskpool to destroy.
 *          On exit, the taskpool cannot be used anymore.
 *
 ******************************************************************************/
void
dgemm_irr_sparse_Destruct( parsec_taskpool_t *tp )
{
    parsec_dgemm_irr_sparse_genB_taskpool_t *dgemm_taskpool = (parsec_dgemm_irr_sparse_genB_taskpool_t *)tp;
    if( dgemm_taskpool->_g_gemm_type == DGEMM_IRR_SPARSE_GENB ) {
        if( dgemm_taskpool->_g_flops > 0 )
            fprintf(stderr, "GFlops done: %10.0g\n", (double)dgemm_taskpool->_g_flops / 1e9);
        if (NULL != dgemm_taskpool->arenas_datatypes[PARSEC_dgemm_irr_sparse_genB_DEFAULT_ADT_IDX].arena)
            parsec_del2arena( &dgemm_taskpool->arenas_datatypes[PARSEC_dgemm_irr_sparse_genB_DEFAULT_ADT_IDX] );
    }
    parsec_taskpool_free(tp);
}
