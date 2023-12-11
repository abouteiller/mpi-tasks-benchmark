/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2014-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define USE_MPI_TIME 1
#include "common.h"

int main( int argc, char* argv[] ) {
    struct opts_t opts;
    int rank, size, nt, k, recvrole, peer;
    MPI_Request* preqs;
    MPI_Comm* tcomms, redcomm;
    double t0, *t, *tpp, tmax, trmax, tsmax, tppg=NAN;

    parse_opts( argc, argv, &opts );
    nt = opts.nthreads;
    if( nt > 1 && opts.initthread < MPI_THREAD_MULTIPLE ) {
        fprintf( stderr, "You requested more than one thread but forced MPI_Init_thread to not use MPI_THREAD_MULTIPLE\n", opts.multimode );
        MPI_Abort( MPI_COMM_WORLD, opts.multimode );
    }
    MPI_Init_thread( NULL, NULL, opts.initthread, &k );
    if( k < opts.initthread ) {
        fprintf( stderr, "MPI_THREADS=%s (wanted %s), I have %d threads\n", thread_level_str(k), thread_level_str(opts.initthread), nt);
        MPI_Abort( MPI_COMM_WORLD, k );
    }
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    if( 0 == rank ) {
        printf( "# %s\tmode: %2d\tthreads: %d\tinitthread: %s\tendpoints: %d\n", argv[0], opts.multimode, opts.nthreads, thread_level_str(k), opts.endpoints );
    }

    switch( opts.multimode ) {
    case mode_11:
        recvrole = rank%2;
        peer = recvrole? rank-1 : rank+1;
        break;
    case mode_n1:
        recvrole = 0==rank;
        peer = 0; /* for senders, peer at the recver set later */
        break;
    case mode_1n:
        recvrole = 0!=rank;
        peer = 0; /* for recvers, peer at the sender set later */
        break;
    }
    MPI_Comm_split( MPI_COMM_WORLD, recvrole, rank, &redcomm );

    tcomms = malloc( sizeof(MPI_Comm) * nt );
    for( k = 0; k < nt; k++ ) {
        if( opts.endpoints ) {
            MPI_Comm_dup( MPI_COMM_WORLD, &tcomms[k] );
        }
        else {
            tcomms[k] = MPI_COMM_WORLD;
        }
    }

    t  = malloc( sizeof(double) * nt );
    tpp= malloc( sizeof(double) * nt );
#define comm (tcomms[tid])

    preqs = malloc( opts.maxclog*sizeof(MPI_Request) );
    for( k = 0; k < opts.maxclog; k++ ) {
        preqs[k] = MPI_REQUEST_NULL;
    }

#pragma omp parallel num_threads(nt) private(t0, k) shared(t, tpp)
    {
        int tid = omp_get_thread_num();
        MPI_Status *statuses;
        MPI_Request* rreqs,* sreqs;
        rreqs = malloc( opts.maxmsgnum*sizeof(MPI_Request) );
        sreqs = malloc( opts.maxmsgnum*sizeof(MPI_Request) );
        char* sb, * rb;
        sb = malloc( opts.maxmsgnum * opts.maxmsgsize );
        rb = malloc( opts.maxmsgnum * opts.maxmsgsize );
#if 0
        /* pin the entire buffer in one go */
        if( rank % 2 ) MPI_Recv( rb, opts.maxmsgn * opts.maxmsgsize, MPI_CHAR, rank-1, 0, comm, MPI_STATUS_IGNORE );
        else MPI_Send( sb, opts.maxmsgn * opts.maxmsgsize, MPI_CHAR, rank+1, 0, comm );
#endif

        for( k = 0; k < opts.maxmsgnum; k++ ) {
            sreqs[k] = rreqs[k] = MPI_REQUEST_NULL;
        }
        statuses = (MPI_Status*)malloc(opts.maxmsgnum * sizeof(MPI_Status));

        for( int prepost = opts.minclog; prepost <= opts.maxclog; prepost? prepost *= 2: prepost++ )
            for( int r = 0; r < opts.nsamples; r++ )
                for( int msgsize = opts.minmsgsize; msgsize <= opts.maxmsgsize; msgsize *= 2 )
                    for( int nmsg = opts.minmsgnum; nmsg <= opts.maxmsgnum; nmsg *= 2 )
                        {
                            if( nmsg*msgsize>opts.maxmsgsize*opts.maxmsgnum ) continue;
#pragma omp master
                            MPI_Barrier( MPI_COMM_WORLD );
#pragma omp barrier

                            if( recvrole ) {

#pragma omp master
                                for( k = 0; k < prepost; k++ ) {
                                    MPI_Irecv( NULL, 0, MPI_CHAR, peer, 1000, comm, &preqs[k] );
                                }

                                for( k = 0; k < nmsg; k++ ) {
                                    statuses[k].MPI_TAG = -2000;
                                    statuses[k].MPI_SOURCE = -2000;
                                    peer = mode_is_n1(opts) ? (1 + k % (size-1)) : peer;
                                    VERBOSE( opts, "%d>%d irecv(%d:%d from %d)\n", nmsg, k+1, rank, tid, peer );
                                    MPI_Irecv( &(rb[k*opts.maxmsgsize]), msgsize, MPI_CHAR,
                                               peer, tid, comm, &rreqs[k] );
                                    VERBOSE( opts, "%d>%d irecv(%d:%d from %d)\n", nmsg, k+1, rank, tid, peer );
//                                    assert(MPI_REQUEST_NULL != rreqs[k]);
                                }
#pragma omp master
                                MPI_Barrier( MPI_COMM_WORLD );
#pragma omp barrier
                                t0=wtime();
                                MPI_Waitall( nmsg, rreqs, statuses );
                                t[tid]=wtime()-t0;
                                MPI_Send( NULL, 0, MPI_CHAR, peer, tid, comm );

                                for( k = 0; k < nmsg; k++ ) {
                                    peer = mode_is_n1(opts) ? (1 + k % (size-1)) : peer;
                                    VERBOSE( opts, "%d>%d irecv status(%d:%d from %d:%d)\n", nmsg, k+1, rank, tid, peer, statuses[k].MPI_TAG );
                                    if( statuses[k].MPI_TAG != tid ) {
                                        printf("tid %d got a message with a wrong tag (expected %d:%d got %d:%d)\n",
                                               tid, peer, tid, statuses[k].MPI_SOURCE, statuses[k].MPI_TAG);
                                    }
                                }

#pragma omp master
                                for( k = 0; k < prepost; k++ ) {
                                    MPI_Cancel( &preqs[k] ); MPI_Request_free( &preqs[k] );
                                }
                            }
                            else {
#pragma omp master
                                MPI_Barrier( MPI_COMM_WORLD );
#pragma omp barrier
                                t0=wtime();
                                for( k = 0;  k < nmsg; k++ ) {
                                    peer = mode_is_1n(opts)? 1+k%(size-1): peer;
                                    MPI_Isend( &(sb[k*opts.maxmsgsize]), msgsize, MPI_CHAR,
                                               peer, tid, comm, &sreqs[k] );
                                    VERBOSE( opts, "%d>%d isend(%d:%d to   %d)\n", nmsg, k+1, rank, tid, peer );
                                }
                                MPI_Waitall( nmsg, sreqs, MPI_STATUSES_IGNORE );
                                t[tid]=wtime()-t0;
                                MPI_Recv( NULL, 0, MPI_CHAR, peer, tid, comm, MPI_STATUS_IGNORE );
                                tpp[tid]=wtime()-t0;
                            }
#pragma omp barrier
#pragma omp master
                            {
                                for( k = 1; k < nt; k++ ) {
                                    if( t[0] < t[k] ) t[0] = t[k];
                                    if( tpp[0] < tpp[k] ) tpp[0] = tpp[k];
                                }
                                MPI_Reduce( &t[0], &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, redcomm );
                                MPI_Reduce( &tpp[0], &tppg, 1, MPI_DOUBLE, MPI_MAX, 0, redcomm );
                                if( 1 == rank ) {
                                    MPI_Send( &tmax, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD );
                                }
                                if( 0 == rank ) {
                                    if( recvrole ) {
                                        trmax = tmax;
                                        MPI_Recv( &tsmax, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
                                    }
                                    else {
                                        tsmax = tmax;
                                        MPI_Recv( &trmax, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
                                    }
                                    tmax = (trmax>tsmax)? trmax: tsmax;
                                    printf("s= %6d n= %6d c= %6d :  max= %13.7e tsnd= %13.7e trecv= %13.7e pingpong= %13.7e (s)  %13.6e (Mmsg/s)  %13.6e (Gbit/s)\n",
                                           msgsize, nmsg, prepost, tmax, tsmax, trmax, tppg, size/2*nt*nmsg/1e6/tppg, size/2*nt*msgsize*8.0*nmsg/(1024*1024*1024*tppg) );
                                }
                            } /* omp master */
                        }
        free(statuses);
    } /* omp parallel */
    MPI_Finalize();
    return 0;
}
