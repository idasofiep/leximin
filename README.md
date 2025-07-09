# leximin
Master thesis project repository

Overview
========
This repository containts implementation of different leximin algorithms. The intention is to analyse and compare different leximin methods.

The main program and methods can be found in the implementations.py file.

There are two types of problems that can be solved: allocations and stratification.
Also, stratification problems in the citizens' assembly format can be solved using the methods in citizensassembly.py.  

Running the program
===================
Some example data files are provided to run the different algorithms.

Running the ordered outcomes method using the large allocation example:
```
$ python solver.py allocations oo large

```

Running the ordered values method with the large allocation example:
```
$ python solver.py allocations ov large

```

Running the ordered outcome method for uniform lotteries using the large stratification example:
```
$ python solver.py stratifications oo_uniform large

```

Running the ordered values uniform method using the large stratification example:
```
$ python solver.py stratifications ov_uniform large

```

Run unit tests:
```
$ python -m unittest

```

Run analysis program:
```
$ python analysis.py

```

Run analysis for citizens assembly program:
```
$ python citizensassemblies.py

```
