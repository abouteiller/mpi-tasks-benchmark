/*
 * Copyright (c) 2016-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/mca/device/device.h"
#include "parsec/parsec_internal.h"
#include "parsec/vpmap.h"
#include "parsec.h"
#include "parsec/data.h"
#include "irr_bs_tm.h"

#include <math.h>
#define _GNU_SOURCE         /* See feature_test_macros(7) */
#include <stdio.h>
int asprintf(char **strp, const char *fmt, ...);
#include <stdlib.h>

#ifdef PARSEC_HAVE_MPI
#include <mpi.h>
#endif /* PARSEC_HAVE_MPI */

int irr_tm_sparse_block_is_full(const irr_bs_tm_t *A, int m, int n)
{
    uint32_t w;
    int c = n * A->mt + m;
    assert(m >= 0);
    assert(m < A->mt);
    assert(n >= 0);
    assert(n < A->nt);
    w = A->sparsity->sparsity[ c/32 ];
    return (w & (1 << (c % 32))) != 0;
}

int irr_tm_sparse_block_is_empty(const irr_bs_tm_t *A, int m, int n)
{
    return !irr_tm_sparse_block_is_full(A, m, n);
}

void irr_tm_sparse_block_set_full(const irr_bs_tm_t *A, int m, int n)
{
    uint32_t w;
    int c = n * A->mt + m;
    int r = m * A->nt + n;
    int i;
    assert(m >= 0);
    assert(m < A->mt);
    assert(n >= 0);
    assert(n < A->nt);
    if( A->sparsity->nb_nnz_cols_in_row[m] == 0 )
        A->sparsity->nb_nnz_row++;
    A->sparsity->nb_nnz_cols_in_row[m]++;
    if( A->sparsity->nb_nnz_rows_in_col[n] == 0 )
        A->sparsity->nb_nnz_col++;
    A->sparsity->nb_nnz_rows_in_col[n]++;

    w = 1 << (c % 32);
    A->sparsity->sparsity[ c/32 ] |= w;
}

void irr_tm_sparse_block_finalize(irr_bs_tm_t *A)
{
  int m, im, n, in;
    A->sparsity->nnz_rows_index = (int*)malloc(A->sparsity->nb_nnz_row * sizeof(int));
    for(m = 0, im = 0; m < A->mt; m++) {
       assert(im < A->sparsity->nb_nnz_row);
       if(A->sparsity->nb_nnz_cols_in_row[m] > 0)
          A->sparsity->nnz_rows_index[im++] = m;
    }
    A->sparsity->nnz_cols_index = (int*)malloc(A->sparsity->nb_nnz_col * sizeof(int));
    for(n = 0, in = 0; n < A->nt; n++) {
       assert(in < A->sparsity->nb_nnz_col);
       if(A->sparsity->nb_nnz_rows_in_col[n] > 0)
          A->sparsity->nnz_cols_index[in++] = n;
    }
}

int irr_tm_sparse_block_col_from_index(const irr_bs_tm_t *A, int m, int ik)
 {
    int n;
    for(n = 0; n < A->nt; n++) {
        if( irr_tm_sparse_block_is_full(A, m, n) ) {
            if(0 == ik) {
                return n;
            }
            ik--;
        }
    }
    assert(0);
    return -1;
 }

int irr_tm_sparse_block_row_from_index(const irr_bs_tm_t *A, int im, int n)
 {
    int m;
    for(m = 0; m < A->mt; m++) {
        if( irr_tm_sparse_block_is_full(A, m, n) ) {
            if(0 == im) {
                return m;
            }
            im--;
        }
    }
    assert(0);
    return -1;
 }

void irr_tm_sparsity_init(irr_bs_tm_t *A)
{
    assert(NULL == A->sparsity);
    A->sparsity = (block_sparsity_t*)malloc(sizeof(block_sparsity_t));
    A->sparsity->nb_nnz_row = 0;
    A->sparsity->nnz_rows_index = NULL;
    A->sparsity->nb_nnz_cols_in_row = (int*)calloc(A->mt, sizeof(int));
    A->sparsity->nb_nnz_col = 0;
    A->sparsity->nnz_cols_index = NULL;
    A->sparsity->nb_nnz_rows_in_col = (int*)calloc(A->nt, sizeof(int));
    A->sparsity->sparsity = (uint32_t*)calloc((A->mt * A->nt + 31) / 32, sizeof(uint32_t));
}

void irr_tm_sparsity_destroy(irr_bs_tm_t *A)
{
    free(A->sparsity->nnz_rows_index);
    free(A->sparsity->nb_nnz_cols_in_row);
    free(A->sparsity->nnz_cols_index);
    free(A->sparsity->nb_nnz_rows_in_col);
    free(A->sparsity->sparsity);
    free(A->sparsity);
    A->sparsity = NULL;
}

void irregular_tiled_matrix_initialize_simple_random_sparsity(irr_bs_tm_t *A, float density, unsigned int *seed)
{
    size_t filling, size;
    int m, moff, n, noff, done;
    assert(A->sparsity == NULL);
    irr_tm_sparsity_init(A);
    size = 0;
    for(m = 0; m < A->mt; m++) {
        for(n = 0; n < A->nt; n++) {
            size += A->Mtiling[m] * A->Ntiling[n];
        }
    }
    filling = 0;
    done = 1;
    while( done && (float)filling/(float)size < density ) {
        moff = (int)floor(A->mt * (double)rand_r(seed)/(double)RAND_MAX);
        noff = (int)floor(A->nt * (double)rand_r(seed)/(double)RAND_MAX);
        done = 0;
        for(n = 0; !done && n < A->nt; n++) {
            for(m = 0; !done && m < A->mt; m++) {
                if( irr_tm_sparse_block_is_empty(A, (m+moff)%A->mt, (n+noff)%A->nt) ) {
                    m = (m+moff)%A->mt;
                    n = (n+noff)%A->nt;
                    filling += A->Mtiling[m] * A->Ntiling[n];
                    irr_tm_sparse_block_set_full(A, m, n);
                    done = 1;
                }
            }
        }
    }

    /* Sanity check: at least one element per column */
    for(n = 0; n < A->nt; n++) {
        for(m = 0; m < A->mt; m++) {
            if( irr_tm_sparse_block_is_full(A, m, n) )
                break;
        }
        if(A->mt == m) {
            moff = (int)floor(A->mt * (double)rand_r(seed)/(double)RAND_MAX);
            irr_tm_sparse_block_set_full(A, moff, n);
        }
    }

	/* Sanity check: at least one element per row */
	for(m = 0; m < A->mt; m++) {
	    for(n = 0; n < A->nt; n++) {
            if( irr_tm_sparse_block_is_full(A, m, n) )
                break;
        }
        if(A->nt == n) {
            noff = (int)floor(A->nt * (double)rand_r(seed)/(double)RAND_MAX);
            irr_tm_sparse_block_set_full(A, m, noff);
        }
    }

    irr_tm_sparse_block_finalize(A);
}

void irregular_tiled_matrix_initialize_simple_dense_sparsity(irr_bs_tm_t *A)
{
    int m, n;
    assert(A->sparsity == NULL);
    irr_tm_sparsity_init(A);
    for(n = 0; n < A->nt; n++) {
        for(m = 0; m < A->mt; m++) {
            assert( irr_tm_sparse_block_is_empty(A, m, n) );
            irr_tm_sparse_block_set_full(A, m, n);
        }
    }
}

static uint32_t       irregular_tiled_matrix_rank_of(     parsec_data_collection_t* dc, ...);
static uint32_t       irregular_tiled_matrix_rank_of_key( parsec_data_collection_t* dc, parsec_data_key_t key);
static int32_t        irregular_tiled_matrix_vpid_of(     parsec_data_collection_t* dc, ...);
static int32_t        irregular_tiled_matrix_vpid_of_key( parsec_data_collection_t* dc, parsec_data_key_t key);
static parsec_data_t* irregular_tiled_matrix_data_of(     parsec_data_collection_t* dc, ...);
static parsec_data_t* irregular_tiled_matrix_data_of_key( parsec_data_collection_t* dc, parsec_data_key_t key);
static parsec_data_key_t irregular_tiled_matrix_coord_to_key(parsec_data_collection_t* dc, ...);
static void           irregular_tiled_matrix_key_to_coord(parsec_data_collection_t* dc, parsec_data_key_t key, int *i, int *j);

static irregular_tile_t *get_tile(irr_bs_tm_t* desc, int i, int j)
{
    int pos;
    irregular_tile_t *tile;
    i += desc->i;
    j += desc->j;
    /* Row major column storage */
    pos = (desc->lnt * i) + j;
    assert(0 <= pos && pos < desc->lmt * desc->lnt);

    tile = desc->data_map+pos;
    if(IRR_TM_CREATE_ON_DEMAND == desc->dynamic_mode &&
       tile->rank == -1) {
        desc->nb_local_tiles++;

        /* We don't have the es... We can't find the VPID... */
        tile->vpid = 0;
        tile->rank = desc->super.myrank;
    }
    return tile;
}

static parsec_data_t *get_data(irr_bs_tm_t* desc, int i, int j)
{
    int pos;
    void *actual_data = NULL;
    parsec_data_t *data = NULL;
    parsec_data_copy_t *data_copy = NULL;
    parsec_datatype_t dtt;

    assert(0 == desc->i);
    assert(0 == desc->j);
    /* Row major column storage */
    pos = (desc->lnt * i) + j;
    assert(0 <= pos && pos < desc->lmt * desc->lnt);

    if( IRR_TM_STATIC == desc->dynamic_mode ) {
        assert(NULL != desc->local_data_map[pos]);
        return desc->local_data_map[pos];
    }

    assert( IRR_TM_GENERATED != desc->dynamic_mode /* We should never call data_of on a generated tile */ );
    assert( IRR_TM_CREATE_ON_DEMAND == desc->dynamic_mode /* There should not be many more cases... */ );

    posix_memalign(&actual_data, PARSEC_ARENA_ALIGNMENT_SSE,
                   desc->Mtiling[i]*desc->Ntiling[j]*parsec_datadist_getsizeoftype(desc->mtype));
    data = PARSEC_OBJ_NEW(parsec_data_t);
    data->owner_device = 0;
    data->key = pos;
    data->dc = &desc->super;
    data->nb_elts = desc->Mtiling[i]*desc->Ntiling[j]*parsec_datadist_getsizeoftype(desc->mtype);

    parsec_translate_matrix_type(desc->mtype, &dtt);
    data_copy = parsec_data_copy_new(data, 0, dtt, PARSEC_DATA_FLAG_PARSEC_MANAGED);
    data_copy->device_private = actual_data;

    if( !parsec_atomic_cas_ptr(&desc->local_data_map[pos], NULL, data) ) {
        parsec_data_copy_detach(data, data_copy, 0);
        free(actual_data);
        PARSEC_OBJ_RELEASE(data_copy);
        PARSEC_OBJ_RELEASE(data);
        data = desc->local_data_map[pos];
    } else {
        get_tile(desc, i, j); /* This will set the tile's rank and mark the tile as full */
    }

    return data;
}

static uint32_t irregular_tiled_matrix_rank_of(parsec_data_collection_t* dc, ...)
{
    int i, j;
    va_list ap;
    irr_bs_tm_t* desc;
    irregular_tile_t* t;

    va_start(ap, dc);
    i = va_arg(ap, int);
    j = va_arg(ap, int);
    va_end(ap);

    desc = (irr_bs_tm_t*)dc;

    if(IRR_TM_GENERATED == desc->dynamic_mode)
        return dc->myrank; /* Generated data collections are ubiquitous */

    t = get_tile(desc, i, j);

    assert(NULL != t);
    return t->rank;
}

static int32_t irregular_tiled_matrix_vpid_of(parsec_data_collection_t* dc, ...)
{
    int i, j;
    va_list ap;

    irr_bs_tm_t* desc = (irr_bs_tm_t*)dc;

    va_start(ap, dc);
    i = va_arg(ap, int);
    j = va_arg(ap, int);
    va_end(ap);

    irregular_tile_t* t = get_tile(desc, i, j);

    assert(NULL != t);
    return t->vpid;
}

static parsec_data_t* irregular_tiled_matrix_data_of(parsec_data_collection_t* dc, ...)
{
    int i, j, pos;
    va_list ap;
    parsec_data_t *data = NULL;

    irr_bs_tm_t* desc = (irr_bs_tm_t*)dc;

    va_start(ap, dc);
    i = va_arg(ap, int);
    j = va_arg(ap, int);
    va_end(ap);

#if defined(DISTRIBUTED)
    assert(dc->myrank == irregular_tiled_matrix_rank_of(dc, i, j) ||
           (IRR_TM_CREATE_ON_DEMAND == desc->dynamic_mode && -1 == irregular_tiled_matrix_rank_of(dc, i, j)));
#endif

    return get_data(desc, i, j);
}

static void irregular_tiled_matrix_key_to_coord(parsec_data_collection_t* dc, parsec_data_key_t key, int *i, int *j)
{
    irr_bs_tm_t* desc = (irr_bs_tm_t*)dc;
    *i = key / desc->lnt;
    *j = key % desc->lnt;
}

static uint32_t irregular_tiled_matrix_rank_of_key(parsec_data_collection_t* dc, parsec_data_key_t key)
{
    int i, j;
    irregular_tiled_matrix_key_to_coord(dc, key, &i, &j);
    return irregular_tiled_matrix_rank_of(dc, i, j);
}

static int32_t irregular_tiled_matrix_vpid_of_key(parsec_data_collection_t* dc, parsec_data_key_t key)
{
    int i, j;
    irregular_tiled_matrix_key_to_coord(dc, key, &i, &j);
    return irregular_tiled_matrix_vpid_of(dc, i, j);
}

static parsec_data_t* irregular_tiled_matrix_data_of_key(parsec_data_collection_t* dc, parsec_data_key_t key)
{
    int i, j;
    irregular_tiled_matrix_key_to_coord(dc, key, &i, &j);
    return irregular_tiled_matrix_data_of(dc, i, j);
}

static parsec_data_key_t irregular_tiled_matrix_coord_to_key(struct parsec_data_collection_s *dc, ...)
{
    irr_bs_tm_t* desc = (irr_bs_tm_t*)dc;
    int i, j;
    va_list ap;

    va_start(ap, dc);
    i = va_arg(ap, unsigned int);
    j = va_arg(ap, unsigned int);
    va_end(ap);

    i += desc->i;
    j += desc->j;

    parsec_data_key_t k = (i * desc->lnt) + j;

    return k;
}

static int irregular_tiled_matrix_key_to_string(parsec_data_collection_t *dc, parsec_data_key_t key, char * buffer, uint32_t buffer_size)
{
    unsigned int m, n;
    int res;
    irr_bs_tm_t* desc = (irr_bs_tm_t*)dc;

    m = key / desc->lnt;
    n = key % desc->lnt;
    res = snprintf(buffer, buffer_size, "%s(%u, %u)", dc->key_base, m, n);
    if (res < 0)
        parsec_warning("Wrong key_to_string for tile (%u, %u) key: %u", m, n, key);
    return 0;
}

unsigned int irr_bs_tm_tile_owner(const irr_bs_tm_t *desc, int i, int j)
{
    int rows_tile = desc->grid.rows;
    int cols_tile = desc->grid.cols;

    int iP = i%rows_tile;
    int iQ = j%cols_tile;

    return iP*desc->grid.cols+iQ;
}

int irr_bs_tm_get_tile_count(const irr_bs_tm_t *desc, int m, int n)
{
    assert(0 <= m && m < desc->mt && 0 <= n && n < desc->nt);
    uint64_t res = desc->Mtiling[m] * desc->Ntiling[n];
    uint64_t max = (uint64_t)desc->m * (uint64_t)desc->n;
    (void)res;
    (void)max;
    assert(0 < res && res <= max);
    return res;
}

void irr_bs_tm_init(irr_bs_tm_t* ddesc,
                    parsec_matrix_type_t mtype,
                    unsigned int nodes, unsigned int myrank,
                    unsigned int lm, unsigned int ln,
                    unsigned int lmt, unsigned int lnt,
                    int* Mtiling, int* Ntiling,
                    unsigned int i, unsigned int j,
                    unsigned int mt, unsigned int nt,
                    unsigned int P,
                    irr_tm_dynamic_mode_t dynamic_mode,
                    void *(*future_resolve_fct)(void*, void *, void *))
{
    unsigned int k;
    parsec_data_collection_t *d = (parsec_data_collection_t*)ddesc;

    if(nodes < P)
        parsec_fatal("Block Cyclic Distribution:\tThere are not enough nodes (%d) to make a process grid with P=%d", nodes, P);

    int Q = nodes / P;
    if(nodes != P*Q)
        parsec_warning("Block Cyclic Distribution:\tNumber of nodes %d doesn't match the process grid %dx%d", nodes, P, Q);

    parsec_data_collection_init(d, nodes, myrank);

    d->rank_of     = irregular_tiled_matrix_rank_of;
    d->rank_of_key = irregular_tiled_matrix_rank_of_key;
    d->vpid_of     = irregular_tiled_matrix_vpid_of;
    d->vpid_of_key = irregular_tiled_matrix_vpid_of_key;
    d->data_of     = irregular_tiled_matrix_data_of;
    d->data_of_key = irregular_tiled_matrix_data_of_key;
    d->data_key    = irregular_tiled_matrix_coord_to_key;

    d->key_to_string = irregular_tiled_matrix_key_to_string;

    parsec_grid_2Dcyclic_init(&ddesc->grid, myrank, P, Q, 1, 1, 0, 0);

    ddesc->data_map = (irregular_tile_t*)calloc(lmt*lnt, sizeof(irregular_tile_t));
    ddesc->local_data_map = (parsec_data_t**)calloc(lmt*lnt, sizeof(parsec_data_t*));
    if( IRR_TM_CREATE_ON_DEMAND == dynamic_mode ) {
        for(int idx = 0; idx < lmt*lnt; idx++)
            ddesc->data_map[idx].rank = -1;
    }

    ddesc->Mtiling = (int*)malloc(lmt*sizeof(int));
    ddesc->Ntiling = (int*)malloc(lnt*sizeof(int));

    for (k = 0; k < lmt; ++k) ddesc->Mtiling[k] = Mtiling[k];
    for (k = 0; k < lnt; ++k) ddesc->Ntiling[k] = Ntiling[k];

    ddesc->mtype = mtype;
    parsec_translate_matrix_type(ddesc->mtype, &ddesc->super.default_dtt);
    ddesc->storage = matrix_Tile;
    ddesc->dtype = BLOCK_SPARSE_IRREGULAR_BLOCKS_MATRIX_DC_TYPE;
    ddesc->bsiz = lm*ln;
    ddesc->lm = lm;
    ddesc->ln = ln;
    ddesc->lmt = lmt;
    ddesc->lnt = lnt;
    ddesc->mt = mt;
    ddesc->nt = nt;
    assert(0 == i);
    ddesc->i = i;
    assert(0 == j);
    ddesc->j = j;

    ddesc->m = 0;
    for (k = 0; k < lmt; ++k) ddesc->m += Mtiling[k];
    ddesc->lm = ddesc->m;
    ddesc->n = 0;
    for (k = 0; k < lnt; ++k) ddesc->n += Ntiling[k];
    ddesc->ln = ddesc->n;

    ddesc->nb_local_tiles = 0;
    ddesc->max_mb = 0;
    ddesc->max_tile = 0;

    for (i = 0; i < lmt; ++i) {
        if (Mtiling[i] > ddesc->max_mb)
            ddesc->max_mb = Mtiling[i];
        for (j = 0; j < lnt; ++j)
            if (Mtiling[i]*Ntiling[j] > ddesc->max_tile)
                /* Worst case scenario */
                ddesc->max_tile = Mtiling[i]*Ntiling[j];
    }
    ddesc->future_resolve_fct = future_resolve_fct;
    asprintf(&ddesc->super.key_dim, "(%d, %d)", mt, nt);

    ddesc->sparsity = NULL;
    ddesc->generate_tile = NULL;
    ddesc->generate_tile_arg = NULL;

    ddesc->dynamic_mode = dynamic_mode;
}

void irregular_tiled_matrix_destroy_data(irr_bs_tm_t* ddesc)
{
    int i, j;
    for (i = 0; i < ddesc->lmt; ++i)
        for (j = 0; j < ddesc->lnt; ++j)
            if (ddesc->super.myrank == irregular_tiled_matrix_rank_of(&ddesc->super, i, j)) {
                uint32_t idx = ((parsec_data_collection_t*)ddesc)->data_key((parsec_data_collection_t*)ddesc, i, j);
                if( NULL != ddesc->local_data_map[idx] ) {
                    if( ddesc->dynamic_mode == IRR_TM_CREATE_ON_DEMAND ) {
                        parsec_data_copy_t *data_copy = parsec_data_get_copy(ddesc->local_data_map[idx], 0);
                        parsec_data_copy_detach(ddesc->local_data_map[idx], data_copy, 0);
                        free(data_copy->device_private);
                        PARSEC_OBJ_RELEASE(data_copy);
                    }
                    parsec_data_destroy(ddesc->local_data_map[idx]);
                }
            }
    if(NULL != ddesc->sparsity) {
        irr_tm_sparsity_destroy(ddesc);
    }
}

void irr_bs_tm_destroy(irr_bs_tm_t* ddesc)
{
    irregular_tiled_matrix_destroy_data(ddesc);

    if (ddesc->data_map)       free(ddesc->data_map);
    if (ddesc->local_data_map) free(ddesc->local_data_map);
    if (ddesc->Mtiling)        free(ddesc->Mtiling);
    if (ddesc->Ntiling)        free(ddesc->Ntiling);

    parsec_data_collection_destroy((parsec_data_collection_t*)ddesc);
    if (ddesc->super.key_dim)  free(ddesc->super.key_dim);
}

void irr_bs_tm_insert_data(irr_bs_tm_t *dc, parsec_data_t *data, int m, int n, int mb, int nb)
{
    uint32_t idx = ((parsec_data_collection_t*)dc)->data_key((parsec_data_collection_t*)dc, m, n);
    assert(dc->local_data_map[ idx ] == NULL);
    PARSEC_OBJ_RETAIN(data);
    dc->local_data_map[ idx ] = data;
    dc->nb_local_tiles++;
    dc->data_map[idx].vpid = 0;
    dc->data_map[idx].rank = ((parsec_data_collection_t*)dc)->myrank;
    if( dc->Mtiling[m] == -1 )
        dc->Mtiling[m] = mb;
    assert(dc->Mtiling[m] == mb);
    if( dc->Ntiling[n] == -1 )
        dc->Ntiling[n] = nb;
    assert(dc->Ntiling[n] == nb);
    assert(irr_tm_sparse_block_is_empty(dc, m, n));
    irr_tm_sparse_block_set_full(dc, m, n);
}

#if defined(PARSEC_HAVE_MPI)
int irr_bs_tm_consolidate_with_MPI(irr_bs_tm_t *dc, void *pcomm)
{
    MPI_Comm comm = *(MPI_Comm*)pcomm;
    int *local = (int*)malloc(dc->lmt*dc->lnt*sizeof(int));
    int *global = (int*)malloc(dc->lmt*dc->lnt*sizeof(int));
    int i, m, n, err = 0;

    if( dc->dynamic_mode != IRR_TM_CREATE_ON_DEMAND ) {
        parsec_warning("irr_bs_tm_consolidate_with_MPI can only be used on IRR_TM_CREATE_ON_DEMAND matrices\n");
        return -1;
    }
    for(i = 0; i < dc->lmt*dc->lnt; i++)
        local[i] = dc->data_map[i].rank;
    MPI_Allreduce(local, global, dc->lmt*dc->lnt, MPI_INT, MPI_MAX, comm);
    for(i = 0; i < dc->lmt*dc->lnt; i++) {
        if( dc->data_map[i].rank > 0 && global[i] != dc->data_map[i].rank ) {
            parsec_warning("Error in use of IRR_TM_CREATE_ON_DEMAND: ranks %d and %d (at least) both created the data at (%d, %d): cannot consolidate without loosing one.\n",
                           global[i], dc->data_map[i].rank, i / dc->lnt, i % dc->lnt);
            err++;
        }
        dc->data_map[i].rank = global[i];
    }
    free(local);
    free(global);

    if( NULL != dc->sparsity ) {
        for(m = 0; m < dc->mt; m++) {
            for(n = 0; n < dc->nt; n++) {
                i = irregular_tiled_matrix_coord_to_key(&dc->super, m, n);
                if( dc->data_map[i].rank != -1 &&
                    irr_tm_sparse_block_is_empty(dc, m, n) )
                    irr_tm_sparse_block_set_full(dc, m, n);
            }
        }
    }
    if( err > 0 )
        return -1;
    return 0;
}
#endif

/* sets up the tile by constructing a new object, then filling specific fields with input parameter */
void irr_bs_tm_set_data(irr_bs_tm_t *ddesc, void *actual_data, int i, int j, int mb, int nb, int vpid, int rank)
{
    uint32_t idx = irregular_tiled_matrix_coord_to_key(&ddesc->super, i, j);

    if (NULL != actual_data) {
        parsec_data_create(ddesc->local_data_map+idx,(parsec_data_collection_t*)ddesc, idx, actual_data,
                           mb*nb*parsec_datadist_getsizeoftype(ddesc->mtype), PARSEC_DATA_FLAG_PARSEC_MANAGED);
        ddesc->nb_local_tiles++;
    }
    ddesc->data_map[idx].vpid = vpid;
    ddesc->data_map[idx].rank = rank;

    assert ((uint32_t)rank == ((parsec_data_collection_t*)ddesc)->myrank || actual_data == NULL);
}
