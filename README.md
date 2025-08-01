# Onika

![](doc/logo.png)

Onika (Object Network Interface for Knit Applications), is a component based HPC software platform to build numerical simulation codes.

## Documentation

Onika is the foundation for the exaNBody particle simulation platform but is not bound to N-Body problems nor other domain specific simulation code.
Existing applications based on its building blocks include Molecular Dynamics, particle based fluid simulations using methods such as Smooth Particle Hydrodynamics (SPH) or rigid body simulations using methods such as Discrete Element Method (DEM).
It uses industry grade standards and widely adopted technologies such as CMake and C++20 for development and build, YAML for user input files, MPI and OpenMP for parallel programming, Cuda and HIP for GPU acceleration.

The documentation is available here: https://collab4exanbody.github.io/doc_onika/

## Citation

To cite `onika`, please use:

@inproceedings{carrard2023exanbody,
  title={ExaNBody: a HPC framework for N-Body applications},
  author={Carrard, Thierry and Prat, Rapha{\"e}l and Latu, Guillaume and Babilotte, Killian and Lafourcade, Paul and Amarsid, Lhassan and Soulard, Laurent},
  booktitle={European Conference on Parallel Processing},
  pages={342-354},
  year = {2024},
  isbn = {978-3-031-50683-3},
  publisher = {Springer-Verlag},
  address = {Berlin, Heidelberg},
  url = {https://doi.org/10.1007/978-3-031-50684-0_27},
  doi = {10.1007/978-3-031-50684-0_27}
}
