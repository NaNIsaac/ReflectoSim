# Changelog

## Unreleased

- The forward physics lives here: Parratt kernel, structures, units, materials, the resolution
  integral and the differentiable torch forward moved from ReflectoFit (source commit
  69c3207) with their tests, golden files and reference-engine comparisons. ReflectoFit now depends
  on this package. Dependencies are numpy, scipy, numba and periodictable; extras `hdf5` and
  `torch`.
- Dataset metadata records the package version as the physics version; files written with the
  earlier `reflectofit` version key still read.
- Priors over layer stacks (layer count, thickness, roughness tied to thickness, SLD with a minimum
  contrast, absorption, an explicitly placed native oxide, substrate) and instruments (resolution,
  2θ offset, I0, background, Kα2, footprint, counts).
- Simulation on the forward model with inert padding for mixed layer counts and one
  resolution node count per dataset; Poisson counts with optional dark counts and read noise.
- Dataset files (npz, HDF5) with the priors, seed and versions in the metadata; command line `sim`,
  `describe`.
