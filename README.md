# cK-Means
(Final) Lab assignment for Parallel Computing course (at) University of Florence.
In this project we have an implementation of the K-Means algorithm over a 2D points space implemented in OpenMP and CUDA.
### Features

- It generates a random set of points.
- The user can select the number of cores to use (OpenMP).
- The user can select the TPB (CUDA).
- Detailed report and presentation inside with informations and tutorial.
- Both versions are deeply tested, plots are included.

### Implementation details (OpenMP version)
The main idea is to parallelize everything that can be parallelized from the 
sequential version.
This process involves the introduction of some critical sections that are resolved
using the #atomic construct.

### Implementation details (CUDA version)
Here the applied solution is to divide the work as it follows:
every CUDA thread represent a point, each point will find his best fitting cluster on his own.
There will be parallelization also in other parts.

### Screenshot
![alt text](https://i.ibb.co/N9HfbFL/Screen-per-readme.png)
nb: gnuplot http://www.gnuplot.info/ has been used for plotting.

### License
Licensed under the term of MIT License.
