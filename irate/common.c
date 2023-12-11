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

#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <omp.h>
#include "common.h"

static void usage( char* progname, struct opts_t* opts ) {
    fprintf( stderr,
      "Usage: %s [options], where options are\n"
      "  [-v[N]]\t%d\t[-verbose[={0..2}]]\n"
      "  [-r N]\t%d\t[-nsamples={1..}]\n"
      "  [-s N]\t%d\t[-minmsgsize={1..maxmsgsize}]\n"
      "  [-S N]\t%d\t[-maxmsgsize={minmsgsize..}]\n"
      "  [-n N]\t%d\t[-minmsgnum={1..maxmsgnum}]\n"
      "  [-N N]\t%d\t[-maxmsgnum={minmsgnum..}]\n"
      "  [-c N]\t%d\t[-minclog={0..maxclog}]\n"
      "  [-C N]\t%d\t[-maxclog={minclog..}]\n"
      "  [-m N]\t%d\t[-multimode={11,12,21}]\n"
      "  [-t N]\t%d\t[-threads=N]\n"
      "  [-i N]\t%d\t[-initthread={0,1,2,3} (0: SINGLE, 1: FUNNELED, 2: SERIALIZED, 3: MULTIPLE)\n"
      "  [-E N]\t%d\t[-endpoints={0,1} (0: threads share the same endpoint, 1: thread exclusive endpoint)\n"
      "  [-h]  \t \t[-help]\n"
      "\n",
      progname,
      opts->verbose, opts->nsamples, opts->minmsgsize, opts->maxmsgsize,
      opts->minmsgnum, opts->maxmsgnum, opts->minclog, opts->maxclog,
      opts->multimode, opts->nthreads, opts->initthread, opts->endpoints
   );
   exit(-optind);
}

const char* thread_level_str(int level)
{
    switch(level) {
      case mode_single: return "SINGLE";
      case mode_serialized: return "SERIALIZED";
      case mode_funneled: return "FUNNELED";
      case mode_multiple: return "MULTIPLE";
      default: return "UNKNOWN (and that's a bad thing)";
    }
}



void parse_opts( int argc, char* argv[], struct opts_t* opts ) {
    struct option longopts[] = {
        { "verbose",    optional_argument,  NULL, 'v' },
        { "nsamples",   required_argument,  NULL, 'r' },
        { "minmsgsize", required_argument,  NULL, 's' },
        { "maxmsgsize", required_argument,  NULL, 'S' },
        { "minmsgnum",  required_argument,  NULL, 'n' },
        { "maxmsgnum",  required_argument,  NULL, 'N' },
        { "minclog",    required_argument,  NULL, 'c' },
        { "maxclog",    required_argument,  NULL, 'C' },
        { "multimode",  required_argument,  NULL, 'm' },
        { "initthread", required_argument,  NULL, 'i' },
        { "endpoints",  required_argument,  NULL, 'E' },
        { "threads",    required_argument,  NULL, 't' },
        { "help",       no_argument,        NULL, 'h' },
        { NULL, 0, NULL, 0 }
    };
    int o, help=0;

    /* default values */
    opts->verbose      = 0;
    opts->nsamples     = 200;
    opts->minmsgsize   = 1;
    opts->maxmsgsize   = (16*1024); // maxmsgsize*maxmsgnum <=PVOLUME
    opts->minmsgnum    = 1;
    opts->maxmsgnum    = (16*1024); // maxmsgsize*maxmsgnum <=PVOLUME
    opts->minclog      = 0;
    opts->maxclog      = 0*(16*1024);
    opts->multimode    = mode_11;
    opts->initthread   = mode_single;
    opts->endpoints    = 0;
#pragma omp parallel
    opts->nthreads      = omp_get_num_threads();

    /* read command line */
    while( -1 != (o = getopt_long_only( argc, argv, "v::r:s:S:n:N:c:C:m:i:E:t:h", longopts, NULL )) ) {
        if( NULL != optarg && '=' == optarg[0] ) optarg++;
        switch( o ) {
        case 'v': {
            opts->verbose = 1;
            if( NULL != optarg ) {
                opts->verbose = atoi( optarg );
            }
            break;
        }
        case 'r': {
            opts->nsamples = atoi( optarg );
            break;
        }
        case 's': {
            opts->minmsgsize = atoi( optarg );
            break;
        }
        case 'S': {
            opts->maxmsgsize = atoi( optarg );
            break;
        }
        case 'n': {
            opts->minmsgnum = atoi( optarg );
            break;
        }
        case 'N': {
            opts->maxmsgnum = atoi( optarg );
            break;
        }
        case 'c': {
            opts->minclog = atoi( optarg );
            break;
        }
        case 'C': {
            opts->maxclog = atoi( optarg );
            break;
        }
        case 'm': {
            opts->multimode = atoi( optarg );
            break;
        }
        case 'i': {
            opts->initthread = atoi( optarg );
            break;
        }
        case 'E': {
            opts->endpoints = atoi( optarg );
            break;
        }
        case 't': {
            opts->nthreads = atoi( optarg );
            break;
        }
        case 'h': {
            help=1;
            break;
        }
        case 0: break;
        default:
            fprintf( stderr, "%s: unrecognized option '%s'\n",
                     argv[0], argv[optind] );
            usage( argv[0], opts );
        }
    }
    if( opts->nsamples < 1 ) usage( argv[0], opts );
    if( opts->minmsgsize < 1 ) usage( argv[0], opts );
    if( opts->minmsgnum < 1 ) usage( argv[0], opts );
    if( opts->minclog < 0 ) usage( argv[0], opts );
    if( opts->maxmsgsize < opts->minmsgsize ) usage( argv[0], opts );
    if( opts->maxmsgnum < opts->minmsgnum ) usage( argv[0], opts );
    if( opts->maxclog < opts->minclog ) usage( argv[0], opts );
    switch( opts->multimode ) {
    case 11: case 12: case 21: break;
    default: usage( argv[0], opts );
    }
    if( opts->initthread < mode_single || opts->initthread > mode_multiple ) usage( argv[0], opts );
    if( opts->nthreads > 1 && opts->initthread != mode_multiple ) {
//        fprintf( stderr, "initthread set to %s, but %d threads are present, switching to %s\n", thread_level_str(opts->initthread), opts->nthreads, thread_level_str(mode_multiple) );
        opts->initthread = mode_multiple;
    }
    if( opts->endpoints < 0 || opts->endpoints > 1 ) usage( argv[0], opts ); 
    if(help) usage( argv[0], opts );
}
