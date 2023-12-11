/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2014-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#if defined(MPI_VERSION)
typedef enum { mode_single=MPI_THREAD_SINGLE, mode_funneled=MPI_THREAD_FUNNELED, mode_serialized=MPI_THREAD_SERIALIZED, mode_multiple=MPI_THREAD_MULTIPLE } threadmode_t;
#else
typedef enum { mode_single=0, mode_funneled=1, mode_serialized=2, mode_multiple=3 } threadmode_t;
#endif
const char* thread_level_str(int level);

struct opts_t {
  int verbose;
  int nsamples;
  int minmsgsize;
  int maxmsgsize;
  int minmsgnum;
  int maxmsgnum;
  int minclog;
  int maxclog;
  enum { mode_11=11, mode_1n=12, mode_n1=21 } multimode;
  int nthreads;
  threadmode_t initthread;
  int endpoints;
};
#define mode_is_11(opts) (mode_11==(opts).multimode)
#define mode_is_1n(opts) (mode_1n==(opts).multimode)
#define mode_is_n1(opts) (mode_n1==(opts).multimode)

void parse_opts( int argc, char* argv[], struct opts_t* opts );


#if !defined(USE_MPI_TIME)
#include <time.h>
static inline double wtime(void) { struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); return (ts.tv_sec+ts.tv_nsec*1e-9); }
#else
#define wtime MPI_Wtime
#endif

#ifdef ENABLE_VERBOSE
#define VERBOSE(opts, ...) do { if( opts.verbose ) printf( __VA_ARGS__ ); } while(0)
#else
#define VERBOSE(opts, ...) do {} while(0)
#endif

