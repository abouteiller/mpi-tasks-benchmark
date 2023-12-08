/*
 * Copyright (c) 2009-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @generated d Thu Dec  7 17:03:56 2023
 *
 */

#include <math.h>
#include <getopt.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "parsec.h"
#include "parsec/execution_stream.h"
#include "parsec/mca/pins/pins.h"
#include "parsec/profiling.h"
#include "parsec/utils/mca_param.h"

#include "irr_bs_tm.h"
#include "irr_bs_tm_init.h"
#include "gemm_irr_sparse.h"

#include "dgemm_irr_sparse_genB.h"

#if defined(IGGOP_HAVE_CUDA)
#include "parsec/mca/device/cuda/device_cuda_internal.h"
#endif /* defined(IGGOP_HAVE_CUDA) */
#if defined(IGGOP_HAVE_HIP)
#include "parsec/mca/device/hip/device_hip_internal.h"
#endif /* defined(IGGOP_HAVE_HIP) */

//static unsigned long long int Rnd64seed = 100;
#define Rnd64_A  6364136223846793005ULL
#define Rnd64_C  1ULL
#define RndF_Mul 5.4210108624275222e-20f
#define RndD_Mul 5.4210108624275222e-20
#define EPSILON  0.000001L

static parsec_key_fn_t default_hash_functions = {
    .key_equal = parsec_hash_table_generic_64bits_key_equal,
    .key_print = parsec_hash_table_generic_64bits_key_print,
    .key_hash  = parsec_hash_table_generic_64bits_key_hash
};

#define imax(a, b) (( (a) > (b) )?(a):(b))
#define imin(a, b) (( (a) < (b) )?(a):(b))

static void init_tiling(int *T, unsigned int *seed, int MT, int M, int mca_random_tiling)
{
    int t, w, l, d, r;
    int MB_min = M/MT/2;
    int MB_max = 2*M/MT;
    int MB = M/MT;
    (void)seed;
    (void)mca_random_tiling;

    for (t = 0; t < MT; ++t) T[t] = MB;
    if (M%MB != 0 && MT > 0) T[MT-1] = M%MB;
    /* good old regular tiling with smaller last tile */

    if (mca_random_tiling) {
        for(t = 0; t < MT*MT; t++) {
            w = rand_r(seed) % MT;
            l = rand_r(seed) % MT;
            if( T[w] >= MB_max )
                continue;
            if( T[l] <= MB_min )
                continue;
            r = rand_r(seed) % (MB_max-MB_min);
            d = imin(imin(r , MB_max - T[w] ), T[l] - MB_min);
            if(d == 0) continue;
            T[l] -= d;
            T[w] += d;
            assert(T[l] > 0 && T[l] <= MB_max);
            assert(T[w] > 0 && T[w] <= MB_max);
        }
    }
}

typedef struct init_irregular_tile_op_arg_s {
    int max_tile_size;
    int zero;
    int seed;
    double **storage;
} init_irregular_tile_op_arg_t;

static void fill_tile_with_random(double *array, int mts, int seed, int mt, int mb, int nb, int m, int n)
{
    int jump, i, j;
    jump = mts * ( m * mt + n ) + seed;
    for (j = 0; j < nb; ++j) {
        for (i = 0; i < mb; ++i) {
            array[i+j*mb] = 0.5f - jump * RndF_Mul;
            jump = Rnd64_A * jump + Rnd64_C;
#if defined(PRECISION_z) || defined(PRECISION_c)
            array[i+j*mb] += I*(.5f - jump * RndF_Mul);
            jump = Rnd64_A * jump + Rnd64_C;
#endif
        }
    }
}

static int init_irregular_tile_op( struct parsec_execution_stream_s *es,
                                   irr_bs_tm_t *M,
                                   int m, int n,
                                   void *args )
{
    init_irregular_tile_op_arg_t *arg = (init_irregular_tile_op_arg_t*)args;
    int mb = M->Mtiling[m];
    int nb = M->Ntiling[n];
    double *array;
    (void)es;

    if( irr_tm_sparse_block_is_empty(M, m, n) )
        return 0;

    posix_memalign((void**)&array, PARSEC_ARENA_ALIGNMENT_CL1, sizeof(double)*mb*nb);
    if( arg->zero ) {
        memset(array, 0, sizeof(double)*mb*nb);
    } else {
        fill_tile_with_random(array, arg->max_tile_size, arg->seed, M->mt, mb, nb, m, n);
    }
    uint32_t idx = ((parsec_data_collection_t*)M)->data_key((parsec_data_collection_t*)M, m, n);
    unsigned int rank = M->super.rank_of( &M->super, m ,n );
    irr_bs_tm_set_data(M, array, m, n, M->Mtiling[m], M->Ntiling[n], 0, rank);
    arg->storage[idx] = array;

    return 0;
}

static int generate_b_tile_function(struct irr_bs_tm_s *g, int m, int n, parsec_data_copy_t *dc)
{
    double *array = parsec_data_copy_get_ptr(dc);
    int mb = g->Mtiling[m];
    int nb = g->Ntiling[n];
    init_irregular_tile_op_arg_t *arg = (init_irregular_tile_op_arg_t*)g->generate_tile_arg;
    int seed = arg->seed;
    int mts = arg->max_tile_size;
    dc->original->dc = &g->super;          // For profiling purposes
    dc->original->key = (g->lnt * m) + n;  // For profiling purposes
    fill_tile_with_random(array, mts, seed, g->mt, mb, nb, m, n);
    return 0;
}

static void irr_sparse_set_data_distribution(irr_bs_tm_t *M)
{
    int m, n;
    for (m = 0; m < M->mt; m++) {
        for (n = 0; n < M->nt; n++) {
            uint32_t idx = ((parsec_data_collection_t*)M)->data_key((parsec_data_collection_t*)M, m, n);
            /*
             * int p = (m * M->grid.rows) / M->mt;
             * int q = (n * M->grid.cols) / M->nt;
             * unsigned int rank = p*M->grid.cols+q; */
            unsigned int rank = irr_bs_tm_tile_owner(M, m, n);
            /* We set the data at NULL for now, a parallel JDF running the init_irregular_tile_op on the
             * local tiles will provide data for the local part */
            irr_bs_tm_set_data(M, NULL, m, n, M->Mtiling[m], M->Ntiling[n], 0, rank);
        }
    }
}

static void fini_matrix(double **Mstorage, int nb)
{
    int i;
    for (i = 0; i < nb; ++i)
        free(Mstorage[i]);
}

static void print_matrix_data(irr_bs_tm_t* A, const char *Aid, double* checkA)
{
#if defined(PRECISION_z)
#define FORMAT " %f+i%f%s"
#elif defined(PRECISION_c)
#define FORMAT " %lf+i%lf%s"
#elif defined(PRECISION_d)
#define FORMAT " %lf%s"
#else
#define FORMAT " %f%s"
#endif

#if defined(PRECISION_z) || defined(PRECISION_c)
#define cmplx_print(z) (z), (z)
#else
#define cmplx_print(z) (z)
#endif

    /* print the matrix in scilab-friendly-ready-to-c/c format */
    int i, j;
    fprintf(stdout, "Matrix_%s = [\n", Aid);
    for (i = 0; i < A->m; i++)
        for (j = 0; j < A->n; ++j)
            fprintf(stdout, FORMAT, cmplx_print(checkA[i+A->m*j]),
                    (j!=A->n-1)?",":(i!=A->m-1)?";\n":"];\n");
}

/* prints meta deta of the matrix */
static void print_matrix_meta(irr_bs_tm_t* A)
{
    fprintf(stdout, "  Grid: %dx%d\n",A->grid.rows, A->grid.cols);
    fprintf(stdout, "  M=%d, N=%d, MT=%d, NT=%d\n", A->m, A->n, A->mt, A->nt);

    int i;
    fprintf(stdout, "  M tiling:");
    for (i = 0; i < A->mt; ++i) fprintf(stdout, " %d", A->Mtiling[i]);
    fprintf(stdout, "\n");
    fprintf(stdout, "  N tiling:");
    for (i = 0; i < A->nt; ++i) fprintf(stdout, " %d", A->Ntiling[i]);
    fprintf(stdout, "\n");

    fprintf(stdout, "  i=%d, j=%d, nb_local_tiles=%d\n", A->i, A->j, A->nb_local_tiles);
    fprintf(stdout, "  lm=%d, ln=%d, lmt=%d, lnt=%d\n", A->lm, A->ln, A->lmt, A->lnt);

    int m, n;
    fprintf(stdout, "    ");
    for(n = 0; n < A->nt; n++) {
        fprintf(stdout, " %2d", n);
    }
    fprintf(stdout, "\n");
    for(m = 0; m < A->mt; m++) {
        fprintf(stdout, "  %2d", m);
        for(n = 0; n < A->nt; n++) {
            if( irr_tm_sparse_block_is_full(A, m, n) )
                fprintf(stdout, " * ");
            else
                fprintf(stdout, "   ");
        }
        fprintf(stdout, "\n");
    }
}

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int info_solution = 0;
    unsigned int Aseed = 3872;
    unsigned int Bseed = 4674;
    unsigned int Tseed = 4242;
    unsigned int Cseed = 1789;
    double Adensity = 0.3;
    double Bdensity = 0.3;
    double bc_part = 0.3;
    int rank  = 0, my_q = 0;
    int nodes = 1;
    int cores = -1;
    int P = -1, Q = -1, M = -1, N = -1, K = -1, MT = -1, NT = -1, KT = -1;
    int MB = -1, NB = -1, KB = -1;
    double alpha = 1., beta = 0.0;
    char **pargv;
    int pargc;
    int verbose = 0;
    int mca_random_tiling;
    int show_progress = 0;
    int RL=0, CL=0, GL=0;
    int debug = -1;
    int doalarm = 0;
#if defined(PRECISION_z) || defined(PRECISION_c)
    alpha -= I * 0.32;
#endif

    while (1) {
        int r;
        int option_index = 0;
        int this_option_optind = optind ? optind : 1;
        static struct option long_options[] = {
            {"P",                    required_argument, 0,  'P' },
            {"Q",                    required_argument, 0,  'Q' },
            {"M",                    required_argument, 0,  'M' },
            {"N",                    required_argument, 0,  'N' },
            {"K",                    required_argument, 0,  'K' },
            {"MT",                   required_argument, 0,  'm' },
            {"NT",                   required_argument, 0,  'n' },
            {"KT",                   required_argument, 0,  'k' },
            {"Adensity",             required_argument, 0,  'A' },
            {"Bdensity",             required_argument, 0,  'B' },
            {"part-for-bc",          required_argument, 0,  'p' },
            {"row-phase-overlap",    required_argument, 0,  'r' },
            {"column-phase-overlap", required_argument, 0,  'c' },
            {"generate-B-lookahead", required_argument, 0,  'g' },
            {"debug",                required_argument, 0,  'd' },
            {"alarm",                      no_argument, 0,  'a' },
            {"help",                       no_argument, 0,  'h' },
            {"verbose",                    no_argument, 0,  'v' },
            {0,                                      0, 0,   0  }
        };

        r = getopt_long(argc, argv, "P:Q:M:N:K:m:n:k:A:B:p:r:c:g:d:hav",
                        long_options, &option_index);
        if (r == -1)
            break;

        switch (r) {
        case 'P':
            P = atoi(optarg);
            break;
        case 'Q':
            Q = atoi(optarg);
            break;
        case 'M':
            M = atoi(optarg);
            break;
        case 'N':
            N = atoi(optarg);
            break;
        case 'K':
            K = atoi(optarg);
            break;
        case 'A':
            Adensity = strtod(optarg, NULL);
            break;
        case 'B':
            Bdensity = strtod(optarg, NULL);
            break;
        case 'p':
            bc_part = strtod(optarg, NULL);
            break;
        case 'm':
            MT = atoi(optarg);
            break;
        case 'n':
            NT = atoi(optarg);
            break;
        case 'k':
            KT = atoi(optarg);
            break;
        case 'r':
            RL = atoi(optarg);
            break;
        case 'c':
            CL = atoi(optarg);
            break;
        case 'g':
            GL = atoi(optarg);
            break;
        case 'v':
            verbose = !verbose;
            break;
        case 'a':
            doalarm = 1;
            break;
    case 'd':
          debug = atoi(optarg);
          break;

        case '?':
        case 'h':
            fprintf(stderr, "Usage: %s [options] -- [parsec options]\n"
                    "Where options are:\n"
                    "  [P|Q]: process grid size\n"
                    "  [M|N|K]: problem size\n"
                    "  [MT|NT|KT]: number of tiles\n"
                    "  [A|B]: set the density of matrices A or B\n"
                    "  [p]: set the part of the GPU memory dedicated to B and C storage\n"
                    "  [r]: set the overlap of row phases\n"
                    "  [c]: set the overlap of column phases\n"
                    "  [g]: set the look ahead for the generation of B columns\n"
                    "  [v]: active verbose mode\n"
                    "  [h]: print this help\n"
                    "  ==\n"
                    "  --mca gemm_random_tiling 1: set random tiling on\n"
                    "  --mca gemm_import_tiledarray 1 --mca gemm_ta_path /path/to/file: set tiling based on file\n"
                    "  --mca gemm_show_progress 1: display the number of flops achieved by each rank since the beginning every 60 seconds\n",
                    argv[0]);
            break;

        default:
            printf("?? getopt returned character code 0%o ??\n", r);
        }
    }

#if defined(DISTRIBUTED)
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nodes);
    }
#endif

    pargc = argc - optind + 1;
    pargv = (char**)malloc((pargc+2)*sizeof(char*));
    pargv[0] = argv[0];
    for(int i = 1; i < pargc; i++) {
      pargv[i] = argv[i+optind-1];
    }
    pargv[pargc] = NULL;

    /* Initialize PaRSEC */
    parsec = parsec_init(cores, &pargc, &pargv);
    free(pargv);

    parsec_info_register(&parsec_per_stream_infos, "GPU::HANDLES",
                         destroy_infokey_handles, NULL,
                         create_infokey_handles, NULL,
                         NULL);

    int tiledarraycase;
    parsec_mca_param_reg_int_name("gemm", "import_tiledarray", "Boolean for importing TiledArray testcase.", false, false, 0, &tiledarraycase);
    fprintf(stderr, "MCA gemm_import_tiledarray = %d\n", tiledarraycase);

    parsec_mca_param_reg_int_name("gemm", "show_progress", "Boolean for displaying on stderr the number of flops achieved by each rank every 60s", false, false, 0, &show_progress);

    if(!tiledarraycase) {
        if( M == -1 )
            M = N;
        if( K == -1 )
            K = N;
        if( MT == -1 )
            MT = NT;
        if( KT == -1 )
            KT = NT;
        MB = M/MT;
        NB = N/NT;
        KB = K/KT;
        if( imin(imin(imin(MT, KT), imin(NT, M)), imin(N, K) ) <= 0 ) {
            fprintf(stderr, "One of M[%d], N[%d], K[%d], MT[%d], NT[%d], KT[%d] at least is not defined (or defined to 0 or -1).\n",
                    M, N, K, MT, NT, KT);
            exit(1);
        }
        if( Adensity <= 0.0 || Adensity > 1.0 || Bdensity <= 0.0 || Bdensity > 1.0 ) {
            fprintf(stderr, "Required density of %g for A and %g for B is not achievable\n", Adensity, Bdensity);
            exit(1);
        }
    }

    if(rank == -1 || (debug == -2)) {
      volatile int loop = 1;
      char hostname[64];
      gethostname(hostname, 64);
      fprintf(stderr, "ssh -t %s gdb -p %d\n", hostname, getpid());
      while(loop) {
        sleep(1);
      }
    }

    if(P == -1) {
      if(Q == -1) {
        Q = 1;
      }
      P = nodes/Q;
    } else {
      if(Q == -1) {
        Q = nodes/P;
      }
    }
    if( P * Q != nodes ) {
      fprintf(stderr, "*** P = %d, Q = %d, world size = %d *** Bailing out.\n", P, Q, nodes);
      exit(1);
    }

    my_q = rank % Q;

    int *Mtiling = NULL;
    int *Ktiling = NULL;
    int *Ntiling = NULL;

    block_sparsity_t *Asparsity = NULL;
    block_sparsity_t *Bsparsity = NULL;
    block_sparsity_t *Csparsity = NULL;

    if ( 0 < tiledarraycase ) { /* Read from file */
        FILE *Aptr, *Bptr, *Cptr;
        char *ta_path;

        parsec_mca_param_reg_string_name("gemm", "ta_path",
                                         "File describing TiledArray data shape and distribution.\n",
                                         false, false,
                                         "", &ta_path);

        char Afile[256], Bfile[256], Cfile[256];
        sprintf(Afile, "%s/Adist.mat", ta_path);
        sprintf(Bfile, "%s/Bdist.mat", ta_path);
        sprintf(Cfile, "%s/Cdist.mat", ta_path);

        tiledarraycase = 0;
        if (NULL == (Aptr = fopen(Afile, "r"))) {
            parsec_warning("File Adist.mat not found in %s. Provide a correct path.\nFalling back to random test.\n", ta_path);
            exit(1);
        } else if (NULL == (Bptr = fopen(Bfile, "r"))) {
            parsec_warning("File Bdist.mat not found in %s. Provide a correct path.\nFalling back to random test.\n", ta_path);
            exit(1);
        } else if (NULL == (Cptr = fopen(Cfile, "r"))) {
            parsec_warning("File Cdist.mat not found in %s. Provide a correct path.\nFalling back to random test.\n", ta_path);
            exit(1);
        } else {
            tiledarraycase = 1;
            /* Read TiledArray test case */
            unsigned int amt, ant, bmt, bnt, cmt, cnt, mb, nb, kb, i, j, p;
            int k;
            irr_bs_tm_t tempA;
            irr_bs_tm_t tempB;
            irr_bs_tm_t tempC;

            fscanf(Aptr, "%u %u\n", &amt, &ant);
            fscanf(Bptr, "%u %u\n", &bmt, &bnt);
            fscanf(Cptr, "%u %u\n", &cmt, &cnt);
            MT = amt;
            KT = ant;
            NT = bnt;
            if( bmt != ant ) {
                fprintf(stderr, "Tiling is not compatible between the two files (ANT=%d, BMT=%d), aborting\n", ant, bmt);
                exit(1);
            }
            if( cmt != amt ) {
                fprintf(stderr, "Tiling is not compatible between the two files (CMT=%d, AMT=%d), aborting\n", cmt, amt);
                exit(1);
            }
            if( cnt != bnt) {
                fprintf(stderr, "Tiling is not compatible between the two files (CNT=%d, BNT=%d), aborting\n", cnt, bnt);
                exit(1);
            }

            fprintf(stderr, "MT = %d, NT = %d, KT = %d\n", MT, NT, KT);

            tempA.mt = amt;
            tempA.nt = ant;
            tempA.sparsity = NULL;
            irr_tm_sparsity_init(&tempA);
            tempB.mt = bmt;
            tempB.nt = bnt;
            tempB.sparsity = NULL;
            irr_tm_sparsity_init(&tempB);
            tempC.mt = cmt;
            tempC.nt = cnt;
            tempC.sparsity = NULL;
            irr_tm_sparsity_init(&tempC);

            Mtiling = (int*)malloc(MT * sizeof(int)); for(k = 0; k < MT; k++) Mtiling[k] = -1;
            Ntiling = (int*)malloc(NT * sizeof(int)); for(k = 0; k < NT; k++) Ntiling[k] = -1;
            Ktiling = (int*)malloc(KT * sizeof(int)); for(k = 0; k < KT; k++) Ktiling[k] = -1;

            M = N = K = 0;
            while(!feof(Aptr)) {
                fscanf(Aptr, "%u %u %u %u\n", &i, &j, &mb, &kb);
                assert(i < MT);
                assert(j < KT);
                assert(kb > 0);
                assert(mb > 0);
                Mtiling[i] = mb;
                Ktiling[j] = kb;
                irr_tm_sparse_block_set_full(&tempA, i, j);
            }
            fclose(Aptr);

            while(!feof(Bptr)) {
                fscanf(Bptr, "%u %u %u %u\n", &i, &j, &kb, &nb);
                assert(i < KT);
                assert(j < NT);
                if (Ktiling[i] != kb && Ktiling[i]!=-1) fprintf(stderr, "Bdist tile (%u;%u) has a mismatching kb = %u, previous value K[i] was %d ?!?\n", i, j, kb, Ktiling[i]);
                assert(kb > 0);
                assert(nb > 0);
                Ktiling[i] = kb;
                Ntiling[j] = nb;
                irr_tm_sparse_block_set_full(&tempB, i, j);
            }
            fclose(Bptr);

            while(!feof(Cptr)) {
                fscanf(Cptr, "%u %u %u %u\n", &i, &j, &mb, &nb);
                assert(i < MT);
                assert(j < NT);
                assert(kb > 0);
                assert(nb > 0);
                Mtiling[i] = mb;
                Ntiling[j] = nb;
                irr_tm_sparse_block_set_full(&tempC, i, j);
            }
            fclose(Cptr);

            for (k = 0; k < MT; ++k) { if(Mtiling[k] == -1) { fprintf(stderr, "Mtiling[%d] is not defined, setting to 0\n", k); Mtiling[k] = 0; } M += Mtiling[k]; }
            for (k = 0; k < NT; ++k) { if(Ntiling[k] == -1) { fprintf(stderr, "Ntiling[%d] is not defined, setting to 0\n", k); Ntiling[k] = 0; } N += Ntiling[k]; }
            for (k = 0; k < KT; ++k) { if(Ktiling[k] == -1) { fprintf(stderr, "Ktiling[%d] is not defined, setting to 0\n", k); Ktiling[k] = 0; } K += Ktiling[k]; }

            if (0 == rank) fprintf(stderr, "M:%d, K:%d, N:%d\n", M, K, N);

            MB = M/MT;
            NB = N/NT;
            KB = K/KT;

            Asparsity = tempA.sparsity;
            Bsparsity = tempB.sparsity;
            Csparsity = tempC.sparsity;
        }
    }

    if ( !tiledarraycase ) {
        Mtiling = (int*)malloc(MT*sizeof(int));
        Ktiling = (int*)malloc(KT*sizeof(int));
        Ntiling = (int*)malloc(NT*sizeof(int));

        KB = 1+(K-1)/KT;

        parsec_mca_param_reg_int_name("gemm", "random_tiling",
                                      "GEMM test will generate a random tiling based on MB, NB, KB",
                                      false, false,
                                      0, &mca_random_tiling);

        init_tiling(Mtiling, &Tseed, MT, M, mca_random_tiling);
        init_tiling(Ntiling, &Tseed, NT, N, mca_random_tiling);
        init_tiling(Ktiling, &Tseed, KT, K, mca_random_tiling);
    }

    if (rank == 0 && verbose) {
        int i;
        fprintf(stderr, "(MT = %d, mean({MB}) = %d) x (KT = %d, mean({KB}) = %d) x (NT = %d, mean({NB}) = %d)\n",
                MT, MB, KT, KB, NT, NB);
        if( MT < 80 && NT < 80 && KT < 80 ) {
            for (i = 0; i < MT; ++i)
                fprintf(stderr, "%s%d%s", (i == 0)?"M tiling: ":" ", Mtiling[i], (i == MT-1)?"\n":"");
            for (i = 0; i < KT; ++i)
                fprintf(stderr, "%s%d%s", (i == 0)?"K tiling: ":" ", Ktiling[i], (i == KT-1)?"\n":"");
            for (i = 0; i < NT; ++i)
                fprintf(stderr, "%s%d%s", (i == 0)?"N tiling: ":" ", Ntiling[i], (i == NT-1)?"\n":"");
        }
    }

    if (rank == 0 && verbose) {
        fprintf(stderr, "Initializing matrices structure\n");
    }
    /* initializing matrix structure */
    irr_bs_tm_t ddescA;
    irr_bs_tm_init(&ddescA, PARSEC_MATRIX_DOUBLE,
                   nodes, rank, M, K, MT, KT,
                   Mtiling, Ktiling,
                   0, 0, MT, KT, P, IRR_TM_STATIC, NULL);
    if(NULL == Asparsity)
        irregular_tiled_matrix_initialize_simple_random_sparsity(&ddescA, Adensity, &Aseed);
    else {
        ddescA.sparsity = Asparsity;
        irr_tm_sparse_block_finalize(&ddescA);
    }
    //irregular_tiled_matrix_initialize_simple_dense_sparsity(&ddescA);
    irr_bs_tm_t ddescB;
    irr_bs_tm_init(&ddescB, PARSEC_MATRIX_DOUBLE,
                   nodes, rank, K, N, KT, NT,
                   Ktiling, Ntiling,
                   0, 0, KT, NT, P, IRR_TM_GENERATED, NULL);
    if(NULL == Bsparsity)
        irregular_tiled_matrix_initialize_simple_random_sparsity(&ddescB, Bdensity, &Bseed);
    else {
        ddescB.sparsity = Bsparsity;
                irr_tm_sparse_block_finalize(&ddescB);
    }

    //irregular_tiled_matrix_initialize_simple_dense_sparsity(&ddescB);
    irr_bs_tm_t ddescC;
    irr_bs_tm_init(&ddescC, PARSEC_MATRIX_DOUBLE,
                   nodes, rank, M, N, MT, NT,
                   Mtiling, Ntiling,
                   0, 0, MT, NT, P, IRR_TM_CREATE_ON_DEMAND, NULL);
    if( NULL == Csparsity ) {
        irr_tm_sparsity_init(&ddescC);
        for(int im = 0; im < ddescA.sparsity->nb_nnz_row; im++) {
            int m = ddescA.sparsity->nnz_rows_index[im];
            for(int in = 0; in < ddescB.sparsity->nb_nnz_row; in++) {
                int n = ddescB.sparsity->nnz_rows_index[in];
                for(int ik = 0; ik < ddescA.sparsity->nb_nnz_cols_in_row[m]; ik++) {
                    int k = irr_tm_sparse_block_col_from_index(&ddescA, m, ik);
                    if( irr_tm_sparse_block_is_full(&ddescB, k, n) ) {
                        irr_tm_sparse_block_set_full(&ddescC, m, n);
                        break;
                    }
                }
            }
        }
    } else
        ddescC.sparsity = Csparsity;
    irr_tm_sparse_block_finalize(&ddescC);

    if (rank == 0 && verbose) {
        fprintf(stderr, "Allocating storage for A\n");
    }
    unsigned int max_tile = imax(ddescA.max_tile, imax(ddescB.max_tile, ddescC.max_tile));
    unsigned int max_mb = imax(ddescA.max_mb, imax(ddescB.max_mb, ddescC.max_mb));
    ddescA.max_tile = ddescB.max_tile = ddescC.max_tile = max_tile;
    ddescA.max_mb = ddescB.max_mb = ddescC.max_mb = max_mb;

    double **Astorage = (double**)calloc(MT*KT, sizeof(double*));

    /* matrix generation */
    if(rank ==0 && verbose) fprintf(stderr, "+++ Generate matrices metadata ... ");
    irr_sparse_set_data_distribution(&ddescA);

    int64_t flops = 0;
    int64_t nb_gemm = 0;
    int *my_ks = (int*)malloc(KT*sizeof(int));
    int *other_ks = (int*)malloc(KT*sizeof(int));
    int nb_my_k;
    int nb_other_k;
    parsec_hash_table_t gemm_per_mn;
    parsec_hash_table_init(&gemm_per_mn, offsetof(gemm_irr_sparse_plan_gemm_at_mn_t, ht_item), 12, default_hash_functions, NULL);

    for(int im = 0; im < ddescC.sparsity->nb_nnz_row; im++) {
        int m = ddescC.sparsity->nnz_rows_index[im];
        for(int in = 0; in < ddescC.sparsity->nb_nnz_cols_in_row[m]; in++) {
            int n = irr_tm_sparse_block_col_from_index(&ddescC, m, in);
            nb_my_k = 0;
            nb_other_k = 0;
            int64_t this_flops = 0;
            int this_gemm = 0;

            assert(irr_tm_sparse_block_is_full(&ddescC, m, n));
            for(int k = 0; k < ddescA.nt; k++) {
                if( irr_tm_sparse_block_is_full(&ddescA, m, k) &&
                    irr_tm_sparse_block_is_full(&ddescB, k, n) ) {
                    if( rank == ddescA.super.rank_of(&ddescA.super, m, k)) {
                        my_ks[nb_my_k] = k;
                        nb_my_k++;
                    } else {
                        other_ks[nb_other_k] = k;
                        nb_other_k++;
                    }
                    this_gemm += 1;
                    this_flops += (int64_t)2 * (int64_t)Mtiling[m] * (int64_t)Ntiling[n] * (int64_t)Ktiling[k];
                }
            }
            flops += this_flops;
            nb_gemm += this_gemm;
            if(nb_my_k + nb_other_k > 0) {
                parsec_key_t key = (parsec_key_t)m * (parsec_key_t)ddescB.nt + (parsec_key_t)n;
                gemm_irr_sparse_plan_gemm_at_mn_t *gemm = malloc(sizeof(gemm_irr_sparse_plan_gemm_at_mn_t) + (nb_my_k+nb_other_k)*sizeof(int));
                gemm->ht_item.key = key;
                gemm->m = m;
                gemm->n = n;
                gemm->nb_k = nb_my_k + nb_other_k;
                if(nb_my_k > 0) {
                    memcpy(gemm->k, my_ks, nb_my_k * sizeof(int));
                }
                if(nb_other_k > 0) {
                    memcpy(gemm->k+nb_my_k, other_ks, nb_other_k * sizeof(int));
                }
                parsec_hash_table_nolock_insert(&gemm_per_mn, &gemm->ht_item);
            }
        }
    }
    free(my_ks);
    free(other_ks);

    if(verbose && rank == 0) fprintf(stderr, "Will do %"PRId64" gemms representing %"PRId64" flops\n", nb_gemm, flops);

    if(rank ==0 && verbose) fprintf(stderr, "+++ Generate matrices local data ... ");
    init_irregular_tile_op_arg_t ddescA_init_arg = {
        .max_tile_size = max_tile,
        .zero = 0,
        .seed = Aseed,
        .storage = Astorage
    };
    parsec_irr_bs_tm_init_taskpool_t *ddescA_init_tp =
        parsec_irr_bs_tm_init_new(&ddescA, init_irregular_tile_op, &ddescA_init_arg);
    parsec_context_add_taskpool(parsec, (parsec_taskpool_t *)ddescA_init_tp);

    init_irregular_tile_op_arg_t ddescB_init_arg = {
        .max_tile_size = max_tile,
        .zero = 0,
        .seed = Bseed,
        .storage = NULL
    };
    ddescB.generate_tile = generate_b_tile_function;
    ddescB.generate_tile_arg = &ddescB_init_arg;

    parsec_context_start(parsec);
    parsec_context_wait(parsec);

    parsec_taskpool_free(&ddescA_init_tp->super);

    if(rank ==0 && verbose) fprintf(stderr, "Done\n");

    if( 0 == rank ) {
        char ikey[64];
        PROFILING_SAVE_dINFO("flops", (double)flops);
        PROFILING_SAVE_iINFO("MT", MT);
        PROFILING_SAVE_iINFO("NT", NT);
        PROFILING_SAVE_iINFO("KT", KT);
        PROFILING_SAVE_uint64INFO("A", (uint64_t)((uintptr_t) &ddescA));
        PROFILING_SAVE_uint64INFO("B", (uint64_t)((uintptr_t) &ddescB));
        PROFILING_SAVE_uint64INFO("C", (uint64_t)((uintptr_t) &ddescC));
        for(int m = 0; m < MT; m++) {
            snprintf(ikey, 64, "MB[%d]", m);
            PROFILING_SAVE_iINFO(ikey, (Mtiling[m]));
        }
        for(int k = 0; k < KT; k++) {
            snprintf(ikey, 64, "KB[%d]", k);
            PROFILING_SAVE_iINFO(ikey, (Ktiling[k]));
        }
        for(int n = 0; n < NT; n++) {
            snprintf(ikey, 64, "NB[%d]", n);
            PROFILING_SAVE_iINFO(ikey, (Ntiling[n]));
        }
    }

    free(Mtiling);
    free(Ntiling);
    free(Ktiling);

    if( verbose && nodes < 5 && MT < 100 && NT < 100 && KT < 100 && rank == 0) {
      fprintf(stderr, "Matrix A:\n");
      print_matrix_meta(&ddescA);
      fprintf(stderr, "Matrix B:\n");
      print_matrix_meta(&ddescB);
      fprintf(stderr, "Matrix C (has all the meta information, but 0 local tiles yet):\n");
      print_matrix_meta(&ddescC);
    }

    double gflops = -1.0;

    int nb_gpus = 0, dev;
    int nb_elt_gpu = 0;
    size_t gpu_elt_size = 0;
    for(dev = 0; dev < (int)parsec_nb_devices; dev++) {
        parsec_device_module_t *device = parsec_mca_device_get(dev);
        if( PARSEC_DEV_HIP == device->type || PARSEC_DEV_CUDA == device->type ) {
            nb_gpus++;
            if( (nb_elt_gpu == 0) || (((parsec_device_gpu_module_t*)device)->mem_nb_blocks < nb_elt_gpu) ) {
                nb_elt_gpu = ((parsec_device_gpu_module_t*)device)->mem_nb_blocks;
            }
            if( gpu_elt_size == 0 ) {
                gpu_elt_size = ((parsec_device_gpu_module_t*)device)->mem_block_size;
            } else {
                if( gpu_elt_size != ((parsec_device_gpu_module_t*)device)->mem_block_size ) {
                    parsec_warning("Current code works only if all GPUs use the same allocation block size...\n");
                    exit(1);
                }
            }
        }
    }
    nb_elt_gpu = 0.95 * nb_elt_gpu; // Keep 5% mem available for safety
    size_t gpu_mem = nb_elt_gpu * gpu_elt_size;
    if( gpu_mem < 10LL*1024LL*1024LL*1024LL ) {
        parsec_warning("Less than 10GB or RAM available on GPUs (%lld found) ?!?\n", gpu_mem);
    }

    int *dev_index = (int*)malloc(nb_gpus * sizeof(int));
    nb_gpus = 0;
    for(dev = 0; dev < (int)parsec_nb_devices; dev++) {
        parsec_device_module_t *device = parsec_mca_device_get(dev);
        if( PARSEC_DEV_HIP == device->type || PARSEC_DEV_CUDA == device->type ) {
            dev_index[nb_gpus++] = device->device_index;
        }
    }

    time_t now;
    time(&now);
    if(rank ==0 && verbose) fprintf(stderr, "Creating plan (starting at %s)... ", ctime(&now));
    struct timeval start, end, diff;
    gettimeofday(&start, NULL);
    assert(ddescA.super.nodes == P*Q);
    gemm_irr_sparse_plan_t *plan = gemm_irr_sparse_create_smart_plan(&ddescA, &ddescB, ddescA.super.myrank, P, P*Q, nb_gpus,
                                                                     gpu_mem, gpu_elt_size, bc_part, ddescA.mt/2,
                                                                     RL, CL, GL,
                                                                     &gemm_per_mn);
    gettimeofday(&end, NULL);
    timersub(&end, &start, &diff);
    if(rank ==0 && verbose) fprintf(stderr, "%d.%06d s\n", (int)diff.tv_sec, (int)diff.tv_usec);
    //gemm_irr_sparse_create_random_plan(A, B, ddescA.super.myrank, ddescA.super.nodes, nb);
    if(verbose && ddescA.super.nodes <= 4 && ddescA.mt < 100 && ddescA.nt < 100 && ddescB.nt < 100)
        gemm_irr_sparse_genB_describe_plan(plan, stderr);

    if(verbose > 0) {
        FILE *f;
        struct stat st;
        char fn[1024];
        snprintf(fn, 1024, "/gpfs/alpine/csc312/scratch/herault/GIBS-%d-%d-%d-%d.plan", ddescA.super.myrank, ddescA.mt, ddescB.nt, ddescA.nt);
        if( stat(fn, &st) != 0 ) {
            f = fopen(fn, "w");
            if(NULL != f) {
                fprintf(stderr, "Rank %d outputs plan into %s...", rank, fn);
                gemm_irr_sparse_genB_output_plan(plan, f);
                fclose(f);
                fprintf(stderr, " Done\n");
            } else {
                fprintf(stderr, "Could not create %s: %s\n", fn, strerror(errno));
            }
        }
    }

    /* Create Parsec taskpool */
    for(int run = 0; run < 5; run++) {
        struct timeval start, end, diff;

        if(verbose && rank == 0) fprintf(stderr, "Creating handle\n");

        parsec_taskpool_t* PARSEC_dgemm_isp = dgemm_irr_sparse_New(alpha, (irr_bs_tm_t*)&ddescA,
                                                                   (irr_bs_tm_t*)&ddescB, beta,
                                                                   (irr_bs_tm_t*)&ddescC,
                                                                   P, Q,
                                                                   plan, nb_gpus, dev_index,
                                                                   RL, CL, GL,
                                                                   show_progress);
        if(verbose && rank == 0) fprintf(stderr, "Submitting taskpool\n");

        parsec_context_add_taskpool(parsec, PARSEC_dgemm_isp);

        int timeout;
        if( (double)flops/1e9/(nb_gpus * 7e3) > 60 ) {
            timeout = 20 * (double)flops/1e9/(nb_gpus * 7e3);
        } else if( 60 * (double)flops/1e9/(nb_gpus * 7e3) >= 60 ) {
            timeout = 60 * (double)flops/1e9/(nb_gpus * 7e3);
        } else {
            timeout = 60;
        }
        if(debug == -1 && doalarm) {
            if(rank == 0) fprintf(stderr, "Running (%"PRId64" gflops to do: at least %g seconds on V100 GPUs -- set an alarm in %d seconds)\n", flops/(int64_t)1000000000, (double)flops/1e9/(nb_gpus * 7e3), timeout);
            alarm(timeout);
        } else {
            if(rank == 0) fprintf(stderr, "Running (%"PRId64" gflops to do: at least %g seconds at 100%% efficiency on V100 GPUs)\n", flops/(int64_t)1000000000, (double)flops/1e9/(nb_gpus * 7e3));
        }

        /* lets rock! */
#if defined(DISTRIBUTED)
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        time(&((parsec_dgemm_irr_sparse_genB_taskpool_t *)PARSEC_dgemm_isp)->_g_start_log);
        ((parsec_dgemm_irr_sparse_genB_taskpool_t *)PARSEC_dgemm_isp)->_g_last_log = ((parsec_dgemm_irr_sparse_genB_taskpool_t *)PARSEC_dgemm_isp)->_g_start_log;


        gettimeofday(&start, NULL);
        parsec_context_start(parsec);
        parsec_context_wait(parsec);
#if defined(DISTRIBUTED)
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        alarm(0);

        gettimeofday(&end, NULL);
        if( 0 == rank ) {
            char runstr[64];
            double time_elapsed;
            timersub(&end, &start, &diff);
            time_elapsed = (double)diff.tv_sec + (double)diff.tv_usec/1e6;
            fprintf(stderr, "DGEMM_IRR_SPARSE\t(dataflow, noprefetch)\tPxQxg= %3d %-3d %d average_NB= %4d M= %7d N= %7d K= %7d Tiling= %s A_density= %g B_density= %g bc_part= %g RL= %d CL= %d GL= %d gflops= %g seconds= %g gflops/s= %14f\n",
                   P, Q, nb_gpus, NB, M, N, K,
                   (tiledarraycase > 0) ? "IrregularTilingFromFile" : (mca_random_tiling ? "RandomIrregularTiling" : "RegularTiling"),
                    Adensity, Bdensity, bc_part, RL, CL, GL, (double)flops/1e9,
                    time_elapsed, gflops=((double)flops/1e9)/time_elapsed);
            snprintf(runstr, 64, "RUN%d_gflops", run);
            PROFILING_SAVE_dINFO(runstr, gflops);
        }
        dgemm_irr_sparse_Destruct( PARSEC_dgemm_isp );

        parsec_devices_reset_load(parsec);
        parsec_devices_release_memory();
    }

    gemm_irr_sparse_destroy_plan(plan);
    free(dev_index);

    parsec_info_id_t iid = parsec_info_lookup(&parsec_per_stream_infos, "GPU::HANDLES", NULL);
    parsec_info_unregister(&parsec_per_stream_infos, iid, NULL);

#if defined(DISTRIBUTED)
    MPI_Comm comm = MPI_COMM_WORLD;
    irr_bs_tm_consolidate_with_MPI(&ddescC, &comm);
#endif

    fini_matrix(Astorage, MT*KT);    free(Astorage);

    irr_bs_tm_destroy( (irr_bs_tm_t*)&ddescA);
    irr_bs_tm_destroy( (irr_bs_tm_t*)&ddescB);
    irr_bs_tm_destroy( (irr_bs_tm_t*)&ddescC);

    parsec_fini(&parsec);
#if defined(DISTRIBUTED)
    MPI_Finalize();
#endif

    return 0;
}
