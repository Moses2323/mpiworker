# MPIWorker 

[![Platform](https://img.shields.io/badge/platform-Linux,%20OS%20X,%20Windows-green.svg?style=flat)](https://github.com/nikolskydn)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg?style=flat)](https://opensource.org/licenses/mit-license.php)


## About

Wrappers for collective operations  [MPI](https://www.open-mpi.org/)(Scaterv, Gatherv and other).

## Example

For an object `w` of class `MPIWorker`:

```
mpiworker::MPIWorker w;
```

```
w.setMode(0)
```

```
w.setNElems(1000)
```

collective operation MPI::Comm::Scatterv takes the form:

```
w.scatterv<float>(x,xPerNode,MPI::FLOAT); 
```

## Documentation

[here](http://nikolskydn.github.io/mpiworker/doc/ru/html/index.html)

