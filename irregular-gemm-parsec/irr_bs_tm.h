#ifndef _IRREGULAR_TILED_MATRIX_H_
#define _IRREGULAR_TILED_MATRIX_H_
/*
 * Copyright (c) 2016-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <assert.h>
#include "parsec/parsec_config.h"
#include "parsec/data.h"
#include "parsec/data_internal.h"
#include "parsec/data_distribution.h"
#include "parsec/data_dist/matrix/grid_2Dcyclic.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"

BEGIN_C_DECLS

#define BLOCK_SPARSE_IRREGULAR_BLOCKS_MATRIX_DC_TYPE 0x10

typedef struct irregular_tile_s {
    int                rank;
    int                vpid;
} irregular_tile_t;

typedef struct block_sparsity_s {
    int        nb_nnz_row;         /**< Number of non-zero rows */
    int        nb_nnz_col;         /**< Number of non-zero columns */
    int       *nnz_rows_index;     /**< Index (in global space) of non-zero rows (indexed in 0-nb_nnz_rows-1) */
    int       *nnz_cols_index;     /**< Index (in global space) of non-zero cols (indexed in 0-nb_nnz_cols-1) */
    int       *nb_nnz_cols_in_row; /**< Number of non-zero columns in a non-zero row (indexed in global space) */
    int       *nb_nnz_rows_in_col; /**< Number of non-zero rows in a non-zero column (indexed in global space) */
    uint32_t  *sparsity;           /**< Full sparsity bit-array (indexed in global space) */
} block_sparsity_t;

typedef enum irr_tm_dynamic_mode_e {
    IRR_TM_STATIC,             /**< The matrix is static: if the data is not set before use, create an error */
    IRR_TM_GENERATED,          /**< The matrix is generated: never store data in the local_data_map, but provide the meta-data.
                                *   generate_tile can be called whenever a tile needs instantiation (without regard to the
                                *   distribution). */
    IRR_TM_CREATE_ON_DEMAND    /**< The matrix is created on demand: the first data_of that is called creates
                                *   a local data_t/data_copy_t pair, store them in the local_data_map and
                                *   initializes this to 0. NB: if two ranks do this for the same data,
                                *   no error will be generated until the consolidation call... */
} irr_tm_dynamic_mode_t;

typedef struct irr_bs_tm_s {
    parsec_data_collection_t     super;           /**< inherited class */
    parsec_grid_2Dcyclic_t       grid;            /**< processes grid */
    irregular_tile_t            *data_map;        /**< map of the meta data of tiles */
    parsec_data_t              **local_data_map;  /**< map of the local data */
    int                         *Mtiling;         /**< array of size lmt giving the tile size */
    int                         *Ntiling;         /**< array of size lnt giving the tile size */
    parsec_matrix_type_t         mtype;           /**< precision of the matrix */
    parsec_matrix_storage_t      storage;    /**< storage of the matrix   */
    int                          dtype;           /**< Distribution type of descriptor */
    int                          bsiz;            /**< size in elements incl padding of a tile - derived parameter */
    unsigned int                 lm;              /**< number of rows of the entire matrix */
    unsigned int                 ln;              /**< number of columns of the entire matrix */
    unsigned int                 llm;
    unsigned int                 lln;

    irr_tm_dynamic_mode_t        dynamic_mode;    /**< see above the definition of irr_tm_dynamic_mode_t */
    block_sparsity_t            *sparsity;        /**< the matrix can be block-sparse, if this is not NULL */
    int                        (*generate_tile)   /**< The matrix can be implicit, in which case it defines this function */
        (struct irr_bs_tm_s *g,                    /*  to generate tile (i, j) data, and that function is called on */
         int i, int j, parsec_data_copy_t *dc);    /*  temporary memory, after it is allocated. */
    void                        *generate_tile_arg; /**< Storage for the generate_tile function global parameters */

    int                          lmt;             /**< number of tile rows of the entire matrix */
    int                          lnt;             /**< number of tile columns of the entire matrix */
    int                          i;               /**< row tile index to the beginning of the submatrix */
    int                          j;               /**< column tile index to the beginning of the submatrix */
    int                          m;               /**< number of rows of the submatrix - derived parameter */
    int                          n;               /**< number of columns of the submatrix - derived parameter */
    int                          mt;              /**< number of tile rows of the submatrix */
    int                          nt;              /**< number of tile columns of the submatrix */
    int                          nb_local_tiles;  /**< number of tile handled locally */
    unsigned int                 max_mb;          /**< maximum value of mb */
    unsigned int                 max_tile;        /**< size of the biggest tile */
    void *(*future_resolve_fct)(void*,void*,void*); /**< Function to use to resolve future if this is needed */
} irr_bs_tm_t;

/**
 * Initialize an empty matrix in A: create a block sparsity information,
 * and set all the blocks to empty.
 */
void irr_tm_sparsity_init(irr_bs_tm_t *A);

/**
 * Free resources allocated by the block sparsity information in A
 */
void irr_tm_sparsity_destroy(irr_bs_tm_t *A);

/**
 * Mark block (m, n) as full in A.
 * It is incorrect to call this if the block sparsity information is not allocated.
 */
void irr_tm_sparse_block_set_full(const irr_bs_tm_t *A, int m, int n);

void irr_tm_sparse_block_finalize(irr_bs_tm_t *A);

/**
 * Returns true iff block (m, n) as full in A.
 * It is incorrect to call this if the block sparsity information is not allocated.
 */
int  irr_tm_sparse_block_is_full (const irr_bs_tm_t *A, int m, int n);

/**
 * Returns true iff block (m, n) as empty in A.
 * It is incorrect to call this if the block sparsity information is not allocated.
 */
int  irr_tm_sparse_block_is_empty(const irr_bs_tm_t *A, int m, int n);

/**
 * Returns n such that A(m, n) is not empty, and | { A(m, n') / n' < n and A(m, n') is not empty } | = ik
 */
int irr_tm_sparse_block_col_from_index(const irr_bs_tm_t *A, int m, int ik);

/**
 * Returns m such that A(m, n) is not empty, and | { A(m', n) / m' < m and A(m', n) is not empty } | = im
 */
int irr_tm_sparse_block_row_from_index(const irr_bs_tm_t *A, int im, int n);

/**
 * Initialize the sparsity structure with a very simple heuristic to reach
 * the desired density
 */
void irregular_tiled_matrix_initialize_simple_random_sparsity(irr_bs_tm_t *A, float density, unsigned int *seed);

/**
 * Initialize the sparsity structure with a dense matrix (for testing purposes)
 */
void irregular_tiled_matrix_initialize_simple_dense_sparsity(irr_bs_tm_t *A);

/**
 * Returns the number of elements in desc(m, n)
 */
int irr_bs_tm_get_tile_count(const irr_bs_tm_t *desc, int m, int n);

/**
 * Returns the owner of desc(i, j)
 */
unsigned int irr_bs_tm_tile_owner(const irr_bs_tm_t *desc, int i, int j);

/**
 * Creates an irregularly tiled block-sparse tiled matrix
 */
void irr_bs_tm_init(
    irr_bs_tm_t* dc,
    parsec_matrix_type_t mtype,
    unsigned int nodes, unsigned int myrank,
    /* global number of rows/cols */
    unsigned int lm, unsigned int ln,
    /* global number of tiles */
    unsigned int lmt, unsigned int lnt,
    /* tiling of the submatrix */
    int*Mtiling, int* Ntiling,
    /* first tile of the submatrix */
    unsigned int i, unsigned int j,
    /* number of tiles of the submatrix */
    unsigned int mt, unsigned int nt,
    unsigned int P,
    irr_tm_dynamic_mode_t dynamic_mode,
    void *(*future_resolve_fct)(void*,void*,void*));

typedef int (*irr_bs_tm_init_op_t)( struct parsec_execution_stream_s *es,
                                    irr_bs_tm_t *desc1,
                                    int m, int n,
                                    void *args );

/**
 * Destroys an irregularly tiled block-sparse tiled matrix
 */
void irr_bs_tm_destroy(irr_bs_tm_t* dc);

/**
 * Set the data for dc(i, j):
 *   mark dc(i, j) as full
 *   assign dc(i, j) to rank and vpid
 *   remember that dc(i, j) is of size mb x nb
 *   and if actual_data is not null,
 *     create a data copy on device 0 for actual_data,
 *     create a data_t for dc(i, j),
 *     and make the data_t point to the data copy,
 */
void irr_bs_tm_set_data(irr_bs_tm_t *dc, void *actual_data, int i, int j, int mb, int nb, int vpid, int rank);

#if defined(PARSEC_HAVE_MPI)
int irr_bs_tm_consolidate_with_MPI(irr_bs_tm_t *dc, void *pcomm);
#endif

void irr_bs_tm_build(irr_bs_tm_t *dc);

END_C_DECLS

#endif /* _IRREGULAR_TILED_MATRIX_H_ */
