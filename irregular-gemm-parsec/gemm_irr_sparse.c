/*
 * Copyright (c) 2020-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT
 *
 */

#include "gemm_irr_sparse.h"

#include "parsec/class/parsec_hash_table.h"
#include <string.h>
#define  _GNU_SOURCE
#include <stdlib.h>
#include "parsec/utils/show_help.h"

#if defined(IGGOP_HAVE_HIP)
#include "parsec/mca/device/hip/device_hip.h"
#endif /* defined(IGGOP_HAVE_HIP) */

void qsort_r(void *base, size_t nmemb, size_t size,
           int (*compar)(const void *, const void *, void *),
           void *arg);

static parsec_key_t gemm_irr_sparse_plan_gemm_key(gemm_irr_sparse_plan_t *plan, int m, int n, int k)
{
    parsec_key_t key = 0;
#if defined(PARSEC_DEBUG_NOISIER)
    {
        unsigned long long int ml = m;
        unsigned long long int nl = n;
        unsigned long long int kl = k;
        unsigned long long int antl = plan->A->nt;
        unsigned long long int bntl = plan->Bgen->nt;
        assert( sizeof(unsigned long long int) == sizeof(parsec_key_t) );
        assert( antl < ULLONG_MAX / bntl ); // A->nt * B->nt does not overflow
        assert( m < ULLONG_MAX / (antl * bntl) ); // m * A->nt * B->nt does not overflow
        assert( n < ULLONG_MAX / antl ); // n * A->nt does not overflow
        assert( m*antl*bntl + n*antl >= m*antl*bntl ); //partial sum does not overflow
        assert( m*antl*bntl + n*antl + k >= m*antl*bntl + n*antl ); // full sum does not overflow
    }
#endif
    key += (parsec_key_t)m * (parsec_key_t)plan->A->nt * (parsec_key_t)plan->Bgen->nt;
    key += (parsec_key_t)n * (parsec_key_t)plan->A->nt;
    key += (parsec_key_t)k;
    return key;
}

int gemm_irr_sparse_first_gemm(gemm_irr_sparse_plan_t *plan,
                               int m, int n)
{
    parsec_key_t key = (parsec_key_t)m * (parsec_key_t)plan->Bgen->nt + (parsec_key_t)n;
    gemm_irr_sparse_plan_gemm_at_mn_t *gemm = parsec_hash_table_nolock_find(plan->gemm_per_mn, key);
    if(NULL == gemm || gemm->nb_k == 0)
        return -1;
    return gemm->k[0];
}

int gemm_irr_sparse_next_gemm(gemm_irr_sparse_plan_t *plan,
                              int m, int n, int k)
{
    parsec_key_t key = gemm_irr_sparse_plan_gemm_key(plan, m, n, k);
    gemm_irr_sparse_plan_gemm_t *gemm = parsec_hash_table_nolock_find(&plan->gemm_per_mnk, key);
    assert(NULL != gemm);
    return gemm_irr_sparse_gemm_k(plan, m, n, gemm->ik+1);
}

int gemm_irr_sparse_prev_gemm(gemm_irr_sparse_plan_t *plan,
                              int m, int n, int k)
{
    parsec_key_t key = gemm_irr_sparse_plan_gemm_key(plan, m, n, k);
    gemm_irr_sparse_plan_gemm_t *gemm = parsec_hash_table_nolock_find(&plan->gemm_per_mnk, key);
    assert(NULL != gemm);
    return gemm_irr_sparse_gemm_k(plan, m, n, gemm->ik-1);
}

int gemm_irr_sparse_nb_gemm(gemm_irr_sparse_plan_t *plan,
                            int m, int n)
{
    parsec_key_t key = (parsec_key_t)m * (parsec_key_t)plan->Bgen->nt + (parsec_key_t)n;
    int ik;
    gemm_irr_sparse_plan_gemm_at_mn_t *gemm = parsec_hash_table_nolock_find(plan->gemm_per_mn, key);
    if(NULL == gemm)
        return 0;
    return gemm->nb_k;
}

int gemm_irr_sparse_gemm_k(gemm_irr_sparse_plan_t *plan,
                           int m, int n, int ik)
{
    parsec_key_t key = (parsec_key_t)m * (parsec_key_t)plan->Bgen->nt + (parsec_key_t)n;
    gemm_irr_sparse_plan_gemm_at_mn_t *gemm = parsec_hash_table_nolock_find(plan->gemm_per_mn, key);
    assert(NULL != gemm);
    if(ik < 0) return -1;
    if(ik >= gemm->nb_k) return -1;
    return gemm->k[ik];
}




int gemm_irr_sparse_plan_my_rank(const gemm_irr_sparse_plan_t *plan, int p, int q)
{
    return plan->p == p && plan->q == q;
}

int gemm_irr_sparse_plan_B_last_col(const gemm_irr_sparse_plan_t *plan, int p, int q)
{
    return plan->nb_local_cols-1;
}

int gemm_irr_sparse_plan_B_col(const gemm_irr_sparse_plan_t *plan, int p, int q, int in)
{
    assert(in>=0);
    assert(in < plan->nb_local_cols);
    return plan->local_cols[in];
}

int gemm_irr_sparse_plan_B_last_row(const gemm_irr_sparse_plan_t *plan, int p, int q, int n)
{
    int m, l = -1;
    /* B has its rows replicated everywhere, so the only question is how many tiles in column n */
    return plan->Bgen->sparsity->nb_nnz_rows_in_col[n]-1;
}

int gemm_irr_sparse_plan_B_row(const gemm_irr_sparse_plan_t *plan, int p, int q, int n, int ik)
{
    return  irr_tm_sparse_block_row_from_index(plan->Bgen, ik, n);
}

void gemm_irr_sparse_genB_describe_plan(const gemm_irr_sparse_plan_t *plan, FILE *f)
{
    int ig, icp, in, irp, igemm;
    gemm_irr_sparse_plan_gpu_t *gpu;
    gemm_irr_sparse_plan_column_phase_t *cp;
    gemm_irr_sparse_plan_row_phase_t *rp;
    gemm_irr_sparse_plan_gemm_t *gemm;
    char str[256];
    int first;

    fprintf(f, "### There are %d GPUs on rank (%d, %d)\n", plan->nb_gpu, plan->p, plan->q);
    for(ig = 0; ig < plan->nb_gpu; ig++) {
        gpu = &plan->gpu[ig];
        fprintf(f, "#### Rank (%d, %d), GPU %d: %d column phases\n", plan->p, plan->q, gpu->gpu_index, gpu->nb_cp);

        for(icp = 0; icp < gpu->nb_cp; icp++) {
            cp = &gpu->cp[icp];
            str[0]='\0';
            for(in = 0; in < cp->nb_col; in++) {
                snprintf(str+strlen(str), 256-strlen(str), "%s%d (%d wide)", in==0?"":", ", cp->col[in].n, plan->Bgen->Ntiling[cp->col[in].n]);
            }
            fprintf(f, "##### Rank (%d, %d), GPU %d, Column-Phase %d: concerns %d columns [%s] and has %d row-phases\n", plan->p, plan->q, gpu->gpu_index, cp->cp_index, cp->nb_col, str, cp->nb_rp);

            for(irp = 0; irp < cp->nb_rp; irp++) {
                rp = &cp->rp[irp];
                first = 1;
                str[0] = '\0';
                for(igemm = 0; strlen(str) < 252 && igemm < rp->nb_gemm; igemm++) {
                    gemm = &rp->gemm[igemm];
                    snprintf(str+strlen(str), 256-strlen(str), "%s(%d,%d,%d)", first ? "" : ", ", gemm->m, gemm->n, gemm->k);
                    first = 0;
                }
                if( strlen(str) >= 252 )
                    sprintf(str+252, "...");
                fprintf(f, "###### Rank (%d, %d, GPU %d, Column-Phase %d, Row-Phase %d: %d GEMMS: [%s]\n", plan->p, plan->q, gpu->gpu_index, cp->cp_index, rp->rp_index, rp->nb_gemm, str);
            }
        }
    }
}

void gemm_irr_sparse_genB_output_plan(const gemm_irr_sparse_plan_t *plan, FILE *f)
{
    int ig, icp, in, irp, igemm;
    gemm_irr_sparse_plan_gpu_t *gpu;
    gemm_irr_sparse_plan_column_phase_t *cp;
    gemm_irr_sparse_plan_row_phase_t *rp;
    gemm_irr_sparse_plan_gemm_t *gemm;

    for(ig = 0; ig < plan->nb_gpu; ig++) {
        gpu = &plan->gpu[ig];
        for(icp = 0; icp < gpu->nb_cp; icp++) {
            cp = &gpu->cp[icp];
            for(irp = 0; irp < cp->nb_rp; irp++) {
                rp = &cp->rp[irp];
                for(igemm = 0; igemm < rp->nb_gemm; igemm++) {
                    gemm = &rp->gemm[igemm];
                    fprintf(f, "cp= %d rp= %d p= %d q= %d g= %d m= %d n= %d k= %d mb= %d nb= %d kb= %d Arank=%d\n",
                            icp, irp, plan->p, plan->q, gpu->gpu_index, gemm->m, gemm->n, gemm->k,
                            plan->A->Mtiling[gemm->m], plan->Bgen->Ntiling[gemm->n], plan->A->Ntiling[gemm->k],
                            plan->A->super.rank_of(&plan->A->super, gemm->m, gemm->k));
                }
            }
        }
    }
}

static parsec_key_fn_t default_hash_functions = {
    .key_equal = parsec_hash_table_generic_64bits_key_equal,
    .key_print = parsec_hash_table_generic_64bits_key_print,
    .key_hash  = parsec_hash_table_generic_64bits_key_hash
};

static int compar_intarray(const void *_a, const void *_b, void *_t)
{
    int a = *(int*)_a;
    int b = *(int*)_b;
    int *t = (int*)_t;
    return t[b]-t[a];
}

typedef struct {
    size_t bsize;
    size_t local_a_size;
} column_sort_arg_t;

static int compar_sizearray(const void *_a, const void *_b, void *_t)
{
    column_sort_arg_t *cmp = (column_sort_arg_t*)_t;
    int a = *(int*)_a;
    int b = *(int*)_b;
    if( cmp[a].local_a_size == cmp[b].local_a_size )
        return cmp[b].bsize-cmp[a].bsize;
    return cmp[b].local_a_size - cmp[a].local_a_size;
}

static int64_t gpu_tile_size(int mb, int nb, size_t elt_size, size_t gpu_ram_grain)
{
    return gpu_ram_grain * (( ((size_t)mb*(size_t)nb*elt_size) + gpu_ram_grain-1 ) / gpu_ram_grain);
}

gemm_irr_sparse_plan_t *gemm_irr_sparse_create_smart_plan(irr_bs_tm_t *A, irr_bs_tm_t *Bgen, int my_rank, int P, int world_size, int nb_gpu, size_t ram_per_gpu, size_t gpu_ram_grain, double part_for_B, int b, int RL, int CL, int GL, parsec_hash_table_t *gemm_per_mn)
{
    int ig, icp, irp, igemm, mm, m, kk, k, ik, i, ni, c, r, r_s;
    gemm_irr_sparse_plan_gpu_t *gpu;
    gemm_irr_sparse_plan_column_phase_t *cp;
    gemm_irr_sparse_plan_row_phase_t *rp;
    gemm_irr_sparse_plan_gemm_t *gemm;
    gemm_irr_sparse_plan_t *plan;
    int nb_col, col_idx, *local_cols, col;
    int *sorted;
    column_sort_arg_t *col_size;
    int nb_row, row_idx, *local_rows, rid;
    int nb_rp, rp_size, step;
    int nb_gemm, gemm_size;
    size_t base_ram_for_cp, *filling;
    int64_t ram_free;
    int p, q, Q;
    size_t elt_size = parsec_datadist_getsizeoftype(A->mtype);

    (void)GL;
    if(CL!=0) {
        parsec_warning("smart plan does not manage CL>0 yet\n");
    }

    plan = (gemm_irr_sparse_plan_t*)malloc(sizeof(gemm_irr_sparse_plan_t));
    plan->A = A;
    plan->Bgen = Bgen;
    plan->nb_gpu = nb_gpu;
    plan->gemm_per_mn = gemm_per_mn;
    plan->nb_local_gemms = 0;
    plan->nb_local_genB = 0;

    assert( (world_size % P) == 0);
    Q = world_size / P;
    p = my_rank / Q;
    q = my_rank % Q;
    plan->p = p;
    plan->q = q;

    /*
      char fn[512];
      snprintf(fn, 512, "/gpfs/alpine/csc312/scratch/herault/dbg-%d-%d.log", plan->p, plan->q);
      FILE *debug = fopen(fn, "w");
    */

    sorted = (int*)malloc(A->mt * sizeof(int));

    if(0) {
        for(c = 0; c < A->mt; c++)
            sorted[c] = c;
        qsort_r(sorted, A->mt, sizeof(int), compar_intarray, A->Mtiling);

        plan->row_ranks = (int*)calloc(A->mt, sizeof(int));
        c = 0;
        r = 0;
        step = 1;
        nb_row = (A->mt + (P-1))/P;
        local_rows = (int*)malloc(nb_row * sizeof(int));
        row_idx = 0;
        while(c < A->mt) {
            plan->row_ranks[ sorted[c] ] = r;
            if( r == p ) {
                local_rows[row_idx++] = sorted[c];
            }
            c++;
            if(P > 1) {
                if((r == P - 1) && (step == 1) ) {
                    step=-1;
                } else if((r == 0) && (step == -1) ) {
                    step=1;
                } else {
                    r+=step;
                }
            }
        }
    } else {
        plan->row_ranks = (int*)calloc(A->mt, sizeof(int));
        local_rows = (int*)malloc(A->mt * sizeof(int));
        row_idx = 0;
        for(c = 0; c < A->mt; c++) {
            r = A->super.rank_of(&A->super, c, 0) / Q;
            plan->row_ranks[c] = r;
            if( r == p ) {
                local_rows[row_idx++] = c;
                //fprintf(debug, "this rank manages row %d\n", c);
            }
        }
    }
    nb_row = row_idx;
    plan->local_rows = local_rows;
    plan->nb_local_rows = nb_row;

    col_size = (column_sort_arg_t*)calloc(Bgen->nt, sizeof(column_sort_arg_t));
    sorted = (int*)realloc(sorted, Bgen->nt * sizeof(int));
    for(ik = 0; ik < Bgen->sparsity->nb_nnz_row; ik++) {
        k = Bgen->sparsity->nnz_rows_index[ik];
        for(ni = 0; ni < Bgen->nt; ni++) {
            if(ik == 0) sorted[ni] = ni;
            if( irr_tm_sparse_block_is_full(Bgen, k, ni) ) {
                col_size[ni].bsize += (size_t)gpu_tile_size(Bgen->Mtiling[k], Bgen->Ntiling[ni], elt_size, gpu_ram_grain);
            }
        }
        for(m = 0; m < A->mt; m++) {
            if( irr_tm_sparse_block_is_full(A, m, k) &&
                A->super.rank_of(&A->super, m, k) == my_rank ) {
                for(ni = 0; ni < Bgen->nt; ni++) {
                    col_size[ni].local_a_size += (size_t)gpu_tile_size(A->Mtiling[m], A->Ntiling[k], elt_size, gpu_ram_grain);
                }
            }
        }
    }

    qsort_r(sorted, Bgen->nt, sizeof(int), compar_sizearray, col_size);

    plan->col_ranks = (int*)calloc(Bgen->nt, sizeof(int));
    c = 0;
    r = 0;
    step = 1;
    nb_col = (Bgen->nt + (Q-1)) / Q;
    local_cols = (int*)malloc(nb_col * sizeof(int));
    col_idx = 0;
    while(c < Bgen->nt) {
        plan->col_ranks[ sorted[c] ] = r;
        if( r == q ) {
            local_cols[col_idx++] = sorted[c];
            //fprintf(debug, "this rank manages column %d\n", sorted[c]);
            for(int ki = 0; ki < Bgen->mt; ki++) {
                if( irr_tm_sparse_block_is_full(Bgen, ki, sorted[c]) ) {
                    plan->nb_local_genB++;
                }
            }
        }
        c++;
        if(Q > 1) {
            if((r == Q - 1) && (step == 1) ) {
                step=-1;
            } else if((r == 0) && (step == -1) ) {
                step=1;
            } else {
                r+=step;
            }
        }
    }
    nb_col = col_idx;
    plan->local_cols = local_cols;
    plan->nb_local_cols = nb_col;

    parsec_hash_table_init(&plan->cp_per_column, offsetof(gemm_irr_sparse_plan_column_t, ht_item), 8, default_hash_functions, NULL);
    parsec_hash_table_init(&plan->gemm_per_mnk, offsetof(gemm_irr_sparse_plan_gemm_t, ht_item), 16, default_hash_functions, NULL);

    plan->gpu = (gemm_irr_sparse_plan_gpu_t*)malloc(plan->nb_gpu * sizeof(gemm_irr_sparse_plan_gpu_t));
    filling = (size_t*)malloc(plan->nb_gpu * sizeof(size_t));
    for(ig = 0; ig < nb_gpu; ig++) {
        gpu = &plan->gpu[ig];
        gpu->plan = plan;
        gpu->gpu_index = ig;
        gpu->nb_cp = 0;

        gpu->cp = (gemm_irr_sparse_plan_column_phase_t*)malloc((nb_col+1)*sizeof(gemm_irr_sparse_plan_column_phase_t));
        gpu->cp[0].col_size = 0;
        gpu->cp[0].col = NULL;
        gpu->cp[0].nb_col = 0;
        gpu->cp[0].cp_index = 0;
        gpu->cp[0].gpu = gpu;
        gpu->cp[0].nb_rp = 0;
        gpu->cp[0].rp = NULL;

        filling[ig] = 0;
    }

    /* We now do a best fit bin packing between all the GPUs */
    size_t ram_limit = ram_per_gpu * part_for_B;
    int nb_col_fitted = 0;
    memset(sorted, 0, nb_col*sizeof(int));
    while(nb_col_fitted < nb_col) {
        int fitted_one = 0;
        size_t size_to_fit;
        int col_to_fit;
        for(col_idx = 0; col_idx < nb_col; col_idx++) {
            int best_gpu = -1;
            size_t best_gpu_fit = 0;

            if( sorted[col_idx] ) continue;

            col_to_fit = local_cols[col_idx];
            size_to_fit = col_size[ col_to_fit ].bsize;
            for(row_idx = 0; row_idx < nb_row; row_idx++) {
                if( gemm_irr_sparse_first_gemm(plan, local_rows[row_idx], col_to_fit) != -1 ) {
                    size_to_fit += (size_t)gpu_tile_size(A->Mtiling[local_rows[row_idx]],
                                                         Bgen->Ntiling[col_to_fit],
                                                         elt_size, gpu_ram_grain);
                }
            }
            for(ig = 0; ig < nb_gpu; ig++) {
                if( filling[ig] + size_to_fit <= ram_limit &&
                    ram_limit - (filling[ig] + size_to_fit) >= best_gpu_fit ) {
                    best_gpu_fit = ram_limit - (filling[ig] + size_to_fit);
                    best_gpu = ig;
                }
            }
            if(best_gpu == -1) {
                continue; // Try to fit another col
            }

            filling[best_gpu] += size_to_fit;
            gpu = &plan->gpu[best_gpu];
            cp = &gpu->cp[gpu->nb_cp];

            cp->gpu = gpu;
            cp->cp_index = gpu->nb_cp;

            if(cp->col_size == cp->nb_col) {
                cp->col_size += 8;
                cp->col = (gemm_irr_sparse_plan_column_t*)realloc(cp->col, sizeof(gemm_irr_sparse_plan_column_t) * cp->col_size);
            }
            //fprintf(debug, "column %d is #%d of cp %d on gpu %d\n", col_to_fit, cp->nb_col, cp->cp_index, best_gpu);
            cp->col[cp->nb_col].n = col_to_fit;
            cp->col[cp->nb_col].ht_item.key = cp->col[cp->nb_col].n;
            cp->col[cp->nb_col].cp = cp;
            cp->nb_col++;
            nb_col_fitted++;
            fitted_one = 1;
            sorted[col_idx] = 1;
        }

        if( 0 == fitted_one ) {
            // We need to allcate a new CP for a GPU, as nothing fits.
            // We take the one with the least available room
            if(size_to_fit > ram_limit) {
                fprintf(stderr, "1 - GPU RAM SIZE for B (%zu bytes) TOO SMALL TO HOLD COLUMN %d of B (%zu bytes)\n", ram_limit, col_to_fit, size_to_fit);
                assert(size_to_fit <= ram_limit);
                exit(1);
            }

            assert(gpu->cp[gpu->nb_cp].nb_col > 0);
            for(ig=0; ig < nb_gpu; ig++) {
                gpu = &plan->gpu[ig];
                gpu->nb_cp++;
                filling[ig] = 0;

                gpu->cp[gpu->nb_cp].col_size = 0;
                gpu->cp[gpu->nb_cp].col = NULL;
                gpu->cp[gpu->nb_cp].nb_col = 0;
                gpu->cp[gpu->nb_cp].cp_index = 0;
                gpu->cp[gpu->nb_cp].gpu = gpu;
                gpu->cp[gpu->nb_cp].nb_rp = 0;
                gpu->cp[gpu->nb_cp].rp = NULL;
            }
        }
    }

    free(sorted);
    sorted = NULL;

    free(filling);
    filling = NULL;
    for(ig = 0; ig < nb_gpu; ig++) {
        gpu = &plan->gpu[ig];
        if(gpu->cp[gpu->nb_cp].nb_col > 0)
            gpu->nb_cp++;
        // We now register each column in the cp_per_column, as they are not going to be reallocated
        for(icp = 0; icp < gpu->nb_cp; icp++) {
            cp = &gpu->cp[icp];
            for(ni = 0; ni < cp->nb_col; ni++) {
                parsec_hash_table_nolock_insert(&plan->cp_per_column, &cp->col[ni].ht_item);
            }
        }
    }

    for(ig = 0; ig < plan->nb_gpu; ig++) {
        gpu = &plan->gpu[ig];
        for(icp = 0; icp < gpu->nb_cp; icp++) {
            cp = &gpu->cp[icp];

            rp = (gemm_irr_sparse_plan_row_phase_t *)malloc(sizeof(gemm_irr_sparse_plan_row_phase_t));
            irp = 0;
            rp_size = 1;
            rp->cp = cp;
            rp->rp_index = irp;
            rp->nb_gemm = 0;

            gemm = (gemm_irr_sparse_plan_gemm_t*)malloc(sizeof(gemm_irr_sparse_plan_gemm_t));
            gemm_size = 1;
            igemm = 0;

            base_ram_for_cp = 0;
            // All the columns of B
            for(col_idx = 0; col_idx < cp->nb_col; col_idx++) {
                base_ram_for_cp += col_size[ cp->col[col_idx].n ].bsize;
                // And the corresponding part of C:
                for(row_idx = 0; row_idx < nb_row; row_idx++) {
                    if( gemm_irr_sparse_first_gemm(plan, local_rows[row_idx], cp->col[col_idx].n) != -1 ) {
                        base_ram_for_cp += (size_t)gpu_tile_size(A->Mtiling[local_rows[row_idx]],
                                                                 Bgen->Ntiling[cp->col[col_idx].n],
                                                                 elt_size, gpu_ram_grain);
                    }
                }
            }

            ram_free = ((int64_t)ram_per_gpu - (int64_t)base_ram_for_cp) / (RL+1);
            if(ram_free <= 0) {
                fprintf(stderr, "2 - GPU RAM SIZE (%zu bytes) TOO SMALL TO HOLD %d COLUMNS of B and corresponding tiles of C (%ld bytes available)\n",
                        ram_per_gpu, cp->nb_col, ram_free);
                assert(0);
                exit(1);
            }

            for(row_idx = 0; row_idx < nb_row; row_idx += b) {
                int found_some_k = 1;
                for(int ik = 0; ik < A->nt && found_some_k; ik++) {
                    found_some_k = 0;
                    for(ni = 0; ni < cp->nb_col; ni++) {
                        for(rid = 0; rid < b && rid + row_idx < nb_row; rid++) {
                            m = local_rows[rid+row_idx];
                            parsec_key_t key = (parsec_key_t)m * (parsec_key_t)plan->Bgen->nt + (parsec_key_t)cp->col[ni].n;
                            gemm_irr_sparse_plan_gemm_at_mn_t *mn = parsec_hash_table_nolock_find(plan->gemm_per_mn, key);
                            if((NULL == mn) || (ik >= mn->nb_k))
                                continue;
                            found_some_k = 1;
                            k = mn->k[ik];

                            if( ram_free < gpu_tile_size(A->Mtiling[m], A->Ntiling[k], elt_size, gpu_ram_grain) ) {
                                // We go in a new RP
                                if(igemm == 0) {
                                    fprintf(stderr, "3 - GPU RAM SIZE (%zu bytes) TOO SMALL TO HOLD %d COLUMNS of B and corresponding tiles of C (%zu bytes left)\n",
                                            ram_per_gpu, cp->nb_col, ram_free);
                                    assert(0);
                                    exit(1);
                                }
                                rp[irp].cp = cp;
                                rp[irp].rp_index = irp;
                                rp[irp].nb_gemm = igemm;
                                rp[irp].gemm = gemm;

                                gemm = (gemm_irr_sparse_plan_gemm_t*)malloc(sizeof(gemm_irr_sparse_plan_gemm_t));
                                gemm_size = 1;
                                igemm = 0;

                                ram_free = ((int64_t)ram_per_gpu - (int64_t)base_ram_for_cp) / (RL+1)
                                    - gpu_tile_size(A->Mtiling[m],
                                                    Bgen->Ntiling[cp->col[ni].n],
                                                    elt_size, gpu_ram_grain);

                                irp++;
                                if( irp == rp_size ) {
                                    rp_size = rp_size + 10;
                                    rp = realloc(rp, rp_size * sizeof(gemm_irr_sparse_plan_row_phase_t));
                                }
                            }

                            gemm[igemm].gemm_index = nb_gemm;
                            gemm[igemm].rp = NULL; // We must set this later, when we know what real rp we are using
                            gemm[igemm].m = m;
                            gemm[igemm].n = cp->col[ni].n;
                            gemm[igemm].k = k;
                            gemm[igemm].ik = ik;
                            gemm[igemm].ht_item.key =  gemm_irr_sparse_plan_gemm_key(plan, m, cp->col[ni].n, k);
                            igemm++;
                            //fprintf(debug, "GEMM(%d, %d, %d) gets row phase %d, column phase %d on GPU %d\n", m, cp->col[ni].n, k, irp, cp->cp_index, ig);

                            ram_free -= gpu_tile_size(A->Mtiling[m], A->Ntiling[k], elt_size, gpu_ram_grain);

                            assert(ram_free >= 0);

                            if( igemm >= gemm_size ) {
                                gemm_size += 10;
                                gemm = realloc(gemm, gemm_size * sizeof(gemm_irr_sparse_plan_gemm_t));
                            }
                        }
                    }
                }
            }
            if( igemm > 0 ) {
                if( irp == rp_size ) {
                    rp_size = rp_size + 10;
                    rp = realloc(rp, rp_size * sizeof(gemm_irr_sparse_plan_row_phase_t));
                }
                rp[irp].cp = cp;
                rp[irp].rp_index = irp;
                rp[irp].nb_gemm = igemm;
                rp[irp].gemm = gemm;
                irp++;
            } else {
                free(gemm);
            }
            cp->nb_rp = irp;
            cp->rp = rp;

            /* Now that all pointers for gemm and rp are set, write the back pointers
             * and register the gemm in the hash table */
            for(irp = 0; irp < cp->nb_rp; irp++) {
                rp = &cp->rp[irp];
                for(igemm = 0; igemm < rp->nb_gemm; igemm++) {
                    gemm = &rp->gemm[igemm];
                    gemm->rp = rp;
                    parsec_hash_table_nolock_insert(&plan->gemm_per_mnk, &gemm->ht_item);
                    //if(NULL != debug) fprintf(debug, "GEMM(%d, %d, %d) inserted in table at %p for key %"PRIu64"\n", gemm->m, gemm->n, gemm->k, gemm, gemm->ht_item.key);
                    plan->nb_local_gemms++;
                }
            }
        }
    }
    /*
      if(NULL != debug)
        fclose(debug);
    */
    if( NULL != col_size )
        free(col_size);
    return plan;
}

#if 0
gemm_irr_sparse_plan_t *gemm_irr_sparse_create_simple_plan(irr_bs_tm_t *A, irr_bs_tm_t *Bgen, int my_rank, int P, int world_size, int nb_gpu, int a, int b)
{
    int ig, icp, irp, igemm, mm, m, kk, k, i, c;
    gemm_irr_sparse_plan_gpu_t *gpu;
    gemm_irr_sparse_plan_column_phase_t *cp;
    gemm_irr_sparse_plan_row_phase_t *rp;
    gemm_irr_sparse_plan_gemm_t *gemm;
    gemm_irr_sparse_plan_t *plan;
    int nb_col, *cols, col;
    int nb_rp, rp_size;
    int nb_gemm, gemm_size;
    int Q, p, q;

    plan = (gemm_irr_sparse_plan_t*)malloc(sizeof(gemm_irr_sparse_plan_t));
    plan->A = A;
    plan->Bgen = Bgen;
    plan->nb_gpu = nb_gpu;

    assert( (world_size % P) == 0);
    Q = world_size / P;
    p = my_rank % P;
    q = my_rank / P;
    plan->p = p;
    plan->q = q;

    plan->col_ranks = (int*)malloc(Bgen->nt * sizeof(int));
    for(i = 0; i < Bgen->nt; i++) {
        plan->col_ranks[i] = i % Q;
    }
    plan->row_ranks = (int*)malloc(A->mt * sizeof(int));
    for(i = 0; i < A->mt; i++) {
        plan->row_ranks[i] = i % P;
    }

    parsec_hash_table_init(&plan->cp_per_column, offsetof(gemm_irr_sparse_plan_column_t, ht_item), 8, default_hash_functions, NULL);
    parsec_hash_table_init(&plan->gemm_per_mnk, offsetof(gemm_irr_sparse_plan_gemm_t, ht_item), 8, default_hash_functions, NULL);

    plan->gpu = (gemm_irr_sparse_plan_gpu_t*)malloc(plan->nb_gpu * sizeof(gemm_irr_sparse_plan_gpu_t));
    cols = NULL;
    for(ig = 0; ig < nb_gpu; ig++) {
        gpu = &plan->gpu[ig];
        gpu->plan = plan;
        gpu->gpu_index = ig;
        gpu->nb_cp = Bgen->nt / (world_size * nb_gpu) + ((my_rank*nb_gpu + ig) < (Bgen->nt % (world_size * nb_gpu)));
        gpu->cp = (gemm_irr_sparse_plan_column_phase_t*)malloc(gpu->nb_cp * sizeof(gemm_irr_sparse_plan_column_phase_t));

        nb_col = gpu->nb_cp;
        cols = (int*)realloc(cols, gpu->nb_cp * sizeof(int));
        for(c = 0; c < gpu->nb_cp; c++) {
            cols[c] = world_size * nb_gpu * c + my_rank * nb_gpu + ig;
        }
        nb_col = 0;

        for(icp = 0; icp < gpu->nb_cp; icp++) {
            cp = &gpu->cp[icp];

            cp->gpu = gpu;
            cp->cp_index = icp;

            /* Select the next column */
            cp->n = cols[nb_col];
            nb_col++;

            cp->ht_item.key = cp->n;
            parsec_hash_table_nolock_insert(&plan->cp_per_column, &cp->ht_item);

            rp = (gemm_irr_sparse_plan_row_phase_t *)malloc(sizeof(gemm_irr_sparse_plan_row_phase_t));
            irp = 0;
            rp_size = 1;
            rp->cp = cp;
            rp->rp_index = irp;
            rp->nb_gemm = 0;

            gemm = (gemm_irr_sparse_plan_gemm_t*)malloc(sizeof(gemm_irr_sparse_plan_gemm_t));
            gemm_size = 1;
            igemm = 0;

            assert( 0 ); // This still needs to be fixed for PxQ grid

            for(mm = 0; mm < A->mt; mm+=a) {
                for(kk = 0; kk < A->nt; kk+=b) {
                    for(m = mm; m < mm+a && m < A->mt; m++) {
                        for(k = kk; k < kk+b && k < A->nt; k++) {
                            if( irr_tm_sparse_block_is_empty( A, m, k ) ||
                                irr_tm_sparse_block_is_empty( Bgen, k, cp->n) )
                                continue;
                            gemm[igemm].gemm_index = nb_gemm;
                            gemm[igemm].rp = NULL; // We must set this later, when we know what real rp we are using
                            gemm[igemm].m = m;
                            gemm[igemm].k = k;
                            gemm[igemm].ht_item.key = gemm_irr_sparse_plan_gemm_key(plan, m, cp->n, k);
                            igemm++;
                            if( igemm >= gemm_size ) {
                                gemm_size += 10;
                                gemm = realloc(gemm, gemm_size * sizeof(gemm_irr_sparse_plan_gemm_t));
                            }
                        }
                    }

                    if(igemm == 0)
                        continue;

                    if( irp == rp_size ) {
                        rp_size = rp_size + 10;
                        rp = realloc(rp, rp_size * sizeof(gemm_irr_sparse_plan_row_phase_t));
                    }

                    // We go in a new RP
                    rp[irp].cp = cp;
                    rp[irp].rp_index = irp;
                    rp[irp].nb_gemm = igemm;
                    rp[irp].gemm = gemm;

                    irp++;

                    gemm = (gemm_irr_sparse_plan_gemm_t*)malloc(sizeof(gemm_irr_sparse_plan_gemm_t));
                    gemm_size = 1;
                    igemm = 0;
                }
            }
            if( igemm > 0 ) {
                if( irp == rp_size ) {
                    rp_size = rp_size + 10;
                    rp = realloc(rp, rp_size * sizeof(gemm_irr_sparse_plan_row_phase_t));
                }
                rp[irp].cp = cp;
                rp[irp].rp_index = irp;
                rp[irp].nb_gemm = igemm;
                rp[irp].gemm = gemm;
                irp++;
            } else {
                free(gemm);
            }
            cp->nb_rp = irp;
            cp->rp = rp;

            /* Now that all pointers for gemm and rp are set, write the back pointers
             * and register the gemm in the hash table */
            for(irp = 0; irp < cp->nb_rp; irp++) {
                rp = &cp->rp[irp];
                for(igemm = 0; igemm < rp->nb_gemm; igemm++) {
                    gemm = &rp->gemm[igemm];
                    gemm->rp = rp;
                    parsec_hash_table_nolock_insert(&plan->gemm_per_mnk, &gemm->ht_item);
                }
            }
        }
    }
    if( NULL != cols )
        free(cols);

    return plan;
}

gemm_irr_sparse_plan_t *gemm_irr_sparse_create_random_plan(irr_bs_tm_t *A, irr_bs_tm_t *Bgen, int my_rank, int P, int world_size, int nb_gpu)
{
    int ig, icp, irp, igemm, m, k, i, c;
    gemm_irr_sparse_plan_gpu_t *gpu;
    gemm_irr_sparse_plan_column_phase_t *cp;
    gemm_irr_sparse_plan_row_phase_t *rp;
    gemm_irr_sparse_plan_gemm_t *gemm;
    gemm_irr_sparse_plan_t *plan;
    int nb_col, *cols, col;
    int nb_rp, rp_size;
    int nb_gemm, gemm_size;
    int Q, p, q;

    plan = (gemm_irr_sparse_plan_t*)malloc(sizeof(gemm_irr_sparse_plan_t));
    plan->A = A;
    plan->Bgen = Bgen;
    plan->nb_gpu = nb_gpu;

    assert( (world_size % P) == 0);
    Q = world_size / P;
    p = my_rank % P;
    q = my_rank / P;
    plan->p = p;
    plan->q = q;

    plan->col_ranks = (int*)malloc(Bgen->nt * sizeof(int));
    for(i = 0; i < Bgen->nt; i++) {
        plan->col_ranks[i] = i % Q;
    }
    plan->row_ranks = (int*)malloc(A->mt * sizeof(int));
    for(i = 0; i < A->mt; i++) {
        plan->row_ranks[i] = i % P;
    }

    parsec_hash_table_init(&plan->cp_per_column, offsetof(gemm_irr_sparse_plan_column_phase_t, ht_item), 8, default_hash_functions, NULL);
    parsec_hash_table_init(&plan->gemm_per_mnk, offsetof(gemm_irr_sparse_plan_gemm_t, ht_item), 8, default_hash_functions, NULL);

    plan->gpu = (gemm_irr_sparse_plan_gpu_t*)malloc(plan->nb_gpu * sizeof(gemm_irr_sparse_plan_gpu_t));
    cols = NULL;
    for(ig = 0; ig < nb_gpu; ig++) {
        gpu = &plan->gpu[ig];
        gpu->plan = plan;
        gpu->gpu_index = ig;
        gpu->nb_cp = Bgen->nt / (world_size * nb_gpu) + ((my_rank*nb_gpu + ig) < (Bgen->nt % (world_size * nb_gpu)));
        gpu->cp = (gemm_irr_sparse_plan_column_phase_t*)malloc(gpu->nb_cp * sizeof(gemm_irr_sparse_plan_column_phase_t));

        nb_col = gpu->nb_cp;
        cols = (int*)realloc(cols, gpu->nb_cp * sizeof(int));
        for(c = 0; c < gpu->nb_cp; c++) {
            cols[c] = world_size * nb_gpu * c + my_rank * nb_gpu + ig;
        }

        for(icp = 0; icp < gpu->nb_cp; icp++) {
            cp = &gpu->cp[icp];

            cp->gpu = gpu;
            cp->cp_index = icp;

            /* Select a random column */
            c = rand() % nb_col;
            cp->nb_col = 1;
            cp->col = (int*)malloc(sizeof(int)*cp->nb_col);
            cp->col[0] = cols[c];
            cols[c] = cols[nb_col-1];
            nb_col--;

            cp->ht_item.key = cp->n;
            parsec_hash_table_nolock_insert(&plan->cp_per_column, &cp->ht_item);

            rp = (gemm_irr_sparse_plan_row_phase_t *)malloc(sizeof(gemm_irr_sparse_plan_row_phase_t));
            irp = 0;
            rp_size = 1;
            rp->cp = cp;
            rp->rp_index = irp;
            rp->nb_gemm = 0;

            gemm = (gemm_irr_sparse_plan_gemm_t*)malloc(sizeof(gemm_irr_sparse_plan_gemm_t));
            gemm_size = 1;
            igemm = 0;

            assert( 0 ); // This still needs to be fixed for PxQ grid

            for(m = 0; m < A->mt; m++) {
                for(int k = 0; k < A->nt; k++) {
                    if( irr_tm_sparse_block_is_empty( A, m, k ) ||
                        irr_tm_sparse_block_is_empty( Bgen, k, cp->n) )
                        continue;
                    gemm[igemm].gemm_index = nb_gemm;
                    gemm[igemm].rp = NULL; // We must set this later, when we know what real rp we are using
                    gemm[igemm].m = m;
                    gemm[igemm].k = k;
                    gemm[igemm].ht_item.key = gemm_irr_sparse_plan_gemm_key(plan, m, cp->n, k);
                    igemm++;

                    if( (rand() % 5) == 0 ) {
                        if( irp == rp_size ) {
                            rp_size = rp_size + 10;
                            rp = realloc(rp, rp_size * sizeof(gemm_irr_sparse_plan_row_phase_t));
                        }

                        // We go in a new RP
                        rp[irp].cp = cp;
                        rp[irp].rp_index = irp;
                        rp[irp].nb_gemm = igemm;
                        rp[irp].gemm = gemm;

                        irp++;

                        gemm = (gemm_irr_sparse_plan_gemm_t*)malloc(sizeof(gemm_irr_sparse_plan_gemm_t));
                        gemm_size = 1;
                        igemm = 0;
                    } else {
                        if( igemm >= gemm_size ) {
                            gemm_size += 10;
                            gemm = realloc(gemm, gemm_size * sizeof(gemm_irr_sparse_plan_gemm_t));
                        }
                    }
                }
            }
            if( igemm > 0 ) {
                if( irp == rp_size ) {
                    rp_size = rp_size + 10;
                    rp = realloc(rp, rp_size * sizeof(gemm_irr_sparse_plan_row_phase_t));
                }
                rp[irp].cp = cp;
                rp[irp].rp_index = irp;
                rp[irp].nb_gemm = igemm;
                rp[irp].gemm = gemm;
                irp++;
            } else {
                free(gemm);
            }
            cp->nb_rp = irp;
            cp->rp = rp;

            /* Now that all pointers for gemm and rp are set, write the back pointers
             * and register the gemm in the hash table */
            for(irp = 0; irp < cp->nb_rp; irp++) {
                rp = &cp->rp[irp];
                for(igemm = 0; igemm < rp->nb_gemm; igemm++) {
                    gemm = &rp->gemm[igemm];
                    gemm->rp = rp;
                    parsec_hash_table_nolock_insert(&plan->gemm_per_mnk, &gemm->ht_item);
                }
            }
        }
    }
    if( NULL != cols )
        free(cols);

    return plan;
}
#endif

static void gemm_irr_sparse_remove_col_from_table(void *item, void*cb_data)
{
    gemm_irr_sparse_plan_column_t *col = (gemm_irr_sparse_plan_column_t*)item;
    parsec_hash_table_t *table = (parsec_hash_table_t*)cb_data;
    parsec_hash_table_nolock_remove(table, col->ht_item.key);
}

static void gemm_irr_sparse_remove_gemm_from_table(void *item, void*cb_data)
{
    gemm_irr_sparse_plan_gemm_t *gemm = (gemm_irr_sparse_plan_gemm_t*)item;
    parsec_hash_table_t *table = (parsec_hash_table_t*)cb_data;
    parsec_hash_table_nolock_remove(table, gemm->ht_item.key);
}

void gemm_irr_sparse_destroy_plan(gemm_irr_sparse_plan_t *plan)
{
    int ig, icp, irp, ni;
    gemm_irr_sparse_plan_gpu_t *gpu;
    gemm_irr_sparse_plan_column_phase_t *cp;
    gemm_irr_sparse_plan_row_phase_t *rp;

    parsec_hash_table_for_all(&plan->cp_per_column, gemm_irr_sparse_remove_col_from_table, &plan->cp_per_column);
    parsec_hash_table_fini(&plan->cp_per_column);
    parsec_hash_table_for_all(&plan->gemm_per_mnk, gemm_irr_sparse_remove_gemm_from_table, &plan->gemm_per_mnk);
    parsec_hash_table_fini(&plan->gemm_per_mnk);

    for(ig = 0; ig < plan->nb_gpu; ig++) {
        gpu = &plan->gpu[ig];

        for(icp = 0; icp < gpu->nb_cp; icp++) {
            cp = &gpu->cp[icp];

            for(irp = 0; irp < cp->nb_rp; irp++) {
                rp = &cp->rp[irp];
                free(rp->gemm);
            }
            free(cp->rp);
            free(cp->col);
        }
        free(gpu->cp);
    }
    free(plan->gpu);
    free(plan->col_ranks);
    free(plan->row_ranks);
    free(plan);
}

int gemm_irr_sparse_genB_col_rank(gemm_irr_sparse_plan_t *plan, int n)
{
    return plan->col_ranks[n];
}

int gemm_irr_sparse_genB_row_rank(gemm_irr_sparse_plan_t *plan, int m)
{
    return plan->row_ranks[m];
}

int gemm_irr_sparse_genB_gpu(gemm_irr_sparse_plan_t *plan, int k)
{
    gemm_irr_sparse_plan_column_t *col;
    col = parsec_hash_table_nolock_find(&plan->cp_per_column, k);
    assert(NULL != col);
    return col->cp->gpu->gpu_index;
}

int gemm_irr_sparse_genB_column_phase(gemm_irr_sparse_plan_t *plan, int k)
{
    int g, l;
    gemm_irr_sparse_plan_column_t *col;
    col = parsec_hash_table_nolock_find(&plan->cp_per_column, k);
    assert(NULL != col);
    return col->cp->cp_index;
}

int gemm_irr_sparse_max_column_phase(gemm_irr_sparse_plan_t *plan, int g)
{
    assert(g < plan->nb_gpu);
    assert(g >= 0);
    return plan->gpu[g].nb_cp-1;
}

int gemm_irr_sparse_nb_columns_of_column_phase(gemm_irr_sparse_plan_t *plan, int g, int cp)
{
    assert(g < plan->nb_gpu);
    assert(g >= 0);
    assert(cp < plan->gpu[g].nb_cp);
    int ret = plan->gpu[g].cp[cp].nb_col;
    return ret;
}

int gemm_irr_sparse_column_of_column_phase(gemm_irr_sparse_plan_t *plan, int g, int cp, int i)
{
    assert(g < plan->nb_gpu);
    assert(g >= 0);
    assert(cp < plan->gpu[g].nb_cp);
    assert(i >= 0);
    assert(i < plan->gpu[g].cp[cp].nb_col);
    int ret = plan->gpu[g].cp[cp].col[i].n;
    return ret;
}

int gemm_irr_sparse_max_row_phase(gemm_irr_sparse_plan_t *plan, int g, int cp)
{
    assert(g < plan->nb_gpu);
    assert(g >= 0);
    assert(cp < plan->gpu[g].nb_cp);
    assert(cp >= 0);
    return plan->gpu[g].cp[cp].nb_rp-1;
}

int gemm_irr_sparse_max_gemm_of_row_phase(gemm_irr_sparse_plan_t *plan, int g, int cp, int rp)
{
    assert(g < plan->nb_gpu);
    assert(g >= 0);
    assert(cp < plan->gpu[g].nb_cp);
    assert(cp >= 0);
    assert(rp < plan->gpu[g].cp[cp].nb_rp);
    assert(rp >= 0);
    return plan->gpu[g].cp[cp].rp[rp].nb_gemm-1;
}

int gemm_irr_sparse_gemm_m_of_row_phase(gemm_irr_sparse_plan_t *plan, int g, int cp, int rp, int ig)
{
    assert(g < plan->nb_gpu);
    assert(g >= 0);
    assert(cp < plan->gpu[g].nb_cp);
    assert(cp >= 0);
    assert(rp < plan->gpu[g].cp[cp].nb_rp);
    assert(rp >= 0);
    assert(ig <  plan->gpu[g].cp[cp].rp[rp].nb_gemm);
    assert(ig >= 0);
    return plan->gpu[g].cp[cp].rp[rp].gemm[ig].m;
}

int gemm_irr_sparse_gemm_n_of_row_phase(gemm_irr_sparse_plan_t *plan, int g, int cp, int rp, int ig)
{
    assert(g < plan->nb_gpu);
    assert(g >= 0);
    assert(cp < plan->gpu[g].nb_cp);
    assert(cp >= 0);
    assert(rp < plan->gpu[g].cp[cp].nb_rp);
    assert(rp >= 0);
    assert(ig <  plan->gpu[g].cp[cp].rp[rp].nb_gemm);
    assert(ig >= 0);
    return plan->gpu[g].cp[cp].rp[rp].gemm[ig].n;
}

int gemm_irr_sparse_gemm_k_of_row_phase(gemm_irr_sparse_plan_t *plan, int g, int cp, int rp, int ig)
{
    assert(g < plan->nb_gpu);
    assert(g >= 0);
    assert(cp < plan->gpu[g].nb_cp);
    assert(cp >= 0);
    assert(rp < plan->gpu[g].cp[cp].nb_rp);
    assert(rp >= 0);
    assert(ig <  plan->gpu[g].cp[cp].rp[rp].nb_gemm);
    assert(ig >= 0);
    return plan->gpu[g].cp[cp].rp[rp].gemm[ig].k;
}

int gemm_irr_sparse_row_rank_of_gemm(gemm_irr_sparse_plan_t *plan, int m, int n, int k)
{
    (void)n;
    (void)k;
    return plan->row_ranks[m];
}

int gemm_irr_sparse_col_rank_of_gemm(gemm_irr_sparse_plan_t *plan, int m, int n, int k)
{
    (void)m;
    (void)k;
    return plan->col_ranks[n];
}

int gemm_irr_sparse_gpu_of_gemm(gemm_irr_sparse_plan_t *plan, int m, int n, int k)
{
    gemm_irr_sparse_plan_gemm_t *gemm;
    parsec_key_t key = gemm_irr_sparse_plan_gemm_key(plan, m, n, k);
    gemm = parsec_hash_table_nolock_find(&plan->gemm_per_mnk, key);
    if(NULL == gemm) {
        volatile int loop = 1;
        char hostname[64];
        gethostname(hostname, 64);
        fprintf(stderr, "ssh -t %s gdb -p %d\n", hostname, getpid());
        while(loop==1) {
          sleep(1);
        }
    }
    assert(NULL != gemm);
    return gemm->rp->cp->gpu->gpu_index;
}

int gemm_irr_sparse_column_phase_of_gemm(gemm_irr_sparse_plan_t *plan, int m, int n, int k)
{
    gemm_irr_sparse_plan_gemm_t *gemm;
    parsec_key_t key = gemm_irr_sparse_plan_gemm_key(plan, m, n, k);
    gemm = parsec_hash_table_nolock_find(&plan->gemm_per_mnk, key);
    assert(NULL != gemm);
    return gemm->rp->cp->cp_index;
}

int gemm_irr_sparse_row_phase_of_gemm(gemm_irr_sparse_plan_t *plan, int m, int n, int k)
{
    gemm_irr_sparse_plan_gemm_t *gemm;
    parsec_key_t key = gemm_irr_sparse_plan_gemm_key(plan, m, n, k);
    gemm = parsec_hash_table_nolock_find(&plan->gemm_per_mnk, key);
    assert(NULL != gemm);
    return gemm->rp->rp_index;
}

int gemm_irr_sparse_C_tile_count(const irr_bs_tm_t *descA, const irr_bs_tm_t *genB, int m, int n)
{
    return descA->Mtiling[m] * genB->Ntiling[n];
}

#if defined(IGGOP_HAVE_HIP)
char *hipblas_error_to_string(hipblasStatus_t hipblas_status)
{
    switch(hipblas_status)
    {
        case HIPBLAS_STATUS_SUCCESS: return "HIPBLAS_STATUS_SUCCESS";
        case HIPBLAS_STATUS_NOT_INITIALIZED: return "HIPBLAS_STATUS_NOT_INITIALIZED";
        case HIPBLAS_STATUS_ALLOC_FAILED: return "HIPBLAS_STATUS_ALLOC_FAILED";
        case HIPBLAS_STATUS_INVALID_VALUE: return "HIPBLAS_STATUS_INVALID_VALUE";
        case HIPBLAS_STATUS_ARCH_MISMATCH: return "HIPBLAS_STATUS_ARCH_MISMATCH";
        case HIPBLAS_STATUS_MAPPING_ERROR: return "HIPBLAS_STATUS_MAPPING_ERROR";
        case HIPBLAS_STATUS_EXECUTION_FAILED: return "HIPBLAS_STATUS_EXECUTION_FAILED";
        case HIPBLAS_STATUS_INTERNAL_ERROR: return "HIPBLAS_STATUS_INTERNAL_ERROR";
        default: return "unknown HIPBLAS error";
    }
}
#endif /* defined(PARSEC_HAVE_HIP) */

void *create_infokey_handles(void *obj, void *_n)
{
    handles_infokey_t *new = malloc(sizeof(handles_infokey_t));

    (void)_n;

#if defined(IGGOP_HAVE_HIP)
    hipblasHandle_t hipblas_handle;
    hipblasStatus_t hipblas_status;
    parsec_hip_exec_stream_t *stream = (parsec_hip_exec_stream_t *)obj;

    /* No need to call hipSetDevice, as this has been done by PaRSEC before calling the task body */
    hipblas_status = hipblasCreate(&hipblas_handle);
    if(HIPBLAS_STATUS_SUCCESS != hipblas_status) {
        if( HIPBLAS_STATUS_ALLOC_FAILED == hipblas_status) {
            parsec_show_help("help-dplasma.txt", "gpu_alloc_failed", 1, "HIPBLAS");
        }
        parsec_fatal("Unable to create HIPBLAS Handle: %s", hipblas_error_to_string(hipblas_status));
        return NULL;
    }
    hipblas_status = hipblasSetStream(hipblas_handle, stream->hip_stream);
    assert(HIPBLAS_STATUS_SUCCESS == hipblas_status);

    new->hipblas_handle = hipblas_handle;
#endif

    return new;
}

void destroy_infokey_handles(void *_h, void *_n)
{
    handles_infokey_t *handles = (handles_infokey_t*)_h;
    (void)_n;
#if defined(IGGOP_HAVE_HIP)
    hipblasDestroy(handles->hipblas_handle);
#endif
    free(handles);
}
