# Changelog

## Unreleased

- Priors over layer stacks (layer count, thickness, roughness tied to thickness, SLD with a minimum
  contrast, absorption, an explicitly placed native oxide, substrate) and instruments (resolution,
  2θ offset, I0, background, Kα2, footprint, counts).
- Simulation on the ReflectoFit forward model with inert padding for mixed layer counts and one
  resolution node count per dataset; Poisson counts with optional dark counts and read noise.
- Dataset files (npz, HDF5) with the priors, seed and versions in the metadata; command line `sim`,
  `describe`.
