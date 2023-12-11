OpenSHMEM Multithreaded Benchmarks
==================================

The OpenSHMEM Multithreaded Benchmarks are a set of tests that stress the capacity of the OpenSHMEM library to handle multiple threads communicating simultaneously.

The Injection Rate Benchmark
----------------------------
The injection rate benchmark (irate) measures the time to absorb multiple `putmem()` pings, followed by a synchronization handshake. Multiple threads are created using OpenMP, each thread runs a copy of the benchmark, peering with a thread on another process.

### Usage
In addition to the number of threads, which is set with OpenMP (i.e. `export OMP_NUM_THREADS=x`), the benchmarks takes the following command line arguments:
```
Usage: oshrate [options], where options are
  [-v[N]]       0       [-verbose[={0..2}]]
  [-r N]        200     [-nsamples={1..}]
  [-s N]        1       [-minmsgsize={1..maxmsgsize}]
  [-S N]        16384   [-maxmsgsize={minmsgsize..}]
  [-n N]        1       [-minmsgnum={1..maxmsgnum}]
  [-N N]        16384   [-maxmsgnum={minmsgnum..}]
  [-c N]        0       [-minclog={0..maxclog}]
  [-C N]        0       [-maxclog={minclog..}]
  [-m N]        11      [-multimode={11,12,21}]
  [-h]                  [-help]
```

#### msgnum
The parameters min/max _msgnum_ control the number of `putmem()` issued in a single bulk. After msgnum messages have been issued, the synchronization handshake will be established (from minmsgsize to maxmsgsize, in a geometric progression of scale 2).

#### msgsize
The parameters min/max _msgsize_ control the size the payload for each `putmem()`. The benchmark will issue a bulk of `putmem()` of _msgsize_ bytes (from minm to max, in a geometric progression of scale 2).

#### multimode
The parameter _multimode_ controls the pairing of processes. In _multimode 11_, 1 process puts to 1 process pairwise. In _multimode 21_, multiple processes issue put operations to the same target. In _multimode 12_, a single process spreads put oprations to multiple targets. 

Thread pairing is independent of process pairing: an origin process thread pairs with the same thread id at the target process; this mapping is immutable. 

_multimode_ permits comparing the injection rate of multiple threads using the resources with that of multiple processes using these same resources. 

#### clog
The parameters min/max _clog_ control the number of pending messages in the two-sided MPI benchmark. The MPI benchmark emulates the same behavior as the 1-sided OpenSHMEM benchmark. The clog exhibits the effect of the matching queue length on two sided injection rate. _clog_ has no effect in the OpenSHMEM benchmark.

### Authors
* Aurélien Bouteiller, The University of Tennessee (Knoxville, TN, USA)
* George Bosilca, The University of Tennessee (Knoxville, TN, USA)

