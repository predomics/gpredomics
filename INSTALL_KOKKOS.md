# Installing Kokkos core and kernels

It is not straight forward (at least on macos):

- Homebrew removed Kokkos so `brew install` won't do.
- Macports has Kokkos but not Kokkos Kernels which does not make it a viable option.
- Micromamba Kokkos port has issue with an old libomp version that seems compiled from macos 11...

So the current procedure is:

- install libomp and cmake by Homebrew: `brew install libomp cmake llvm`
- add in your building shell:
```sh
echo 'export PATH="/opt/homebrew/opt/llvm/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"
export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"

export CMAKE_PREFIX_PATH="/opt/homebrew/opt/libomp"
```
- in a `/usr/local/src` folder do:
```sh
git clone https://github.com/kokkos/kokkos.git
git clone https://github.com/kokkos/kokkos-kernels.git
```
- for KokkosCore: 
```sh
cd kokkos
mkdir build && cd build
cmake .. \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="${CPPFLAGS}" \
  -DCMAKE_EXE_LINKER_FLAGS="${LDFLAGS}"
make
sudo make install
cd ../..
```

- for KokkosKernel: 
```sh
cd kokkos-kernel
mkdir build && cd build
cmake .. \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DKokkos_ROOT=/path/to/install/kokkos \
  -DKokkosKernels_ENABLE_TESTS=OFF \
  -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DCMAKE_BUILD_TYPE=Release
make -j 10
sudo make install
cd ../..
```
