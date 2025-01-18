# Installing Kokkos core and kernels

It is not straight forward (at least on macos):

- Homebrew removed Kokkos so `brew install` won't do.
- Macports has Kokkos but not Kokkos Kernels which does not make it a viable option.

Trying Micromamba:
```sh
brew install micromamba
micromamba install -c https://repo.prefix.dev/conda-forge kokkos-kernels
```