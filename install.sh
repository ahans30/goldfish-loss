# Complete, reproducible script to build and prepare environment
LITGPT_REPO=$(pwd)

# modify the installation path and env name if you want
INSTALLDIR=${HOME}
ENV_NAME="goldfish_loss"

cd ${INSTALLDIR}

# Base the installation on conda from module load
source deactivate > /dev/null 2>&1 # discard potentially preloaded conda environments
module load miniforge3
echo "Conda Version:" $(which conda) 


# Create conda environment, and print whether it is loaded correctly
conda create --prefix ${INSTALLDIR}/$ENV_NAME python=3.11 --yes -c defaults
source activate ${INSTALLDIR}/$ENV_NAME
echo "Pip Version:" $(which pip)  # should be from the new environment!

# Conda packages:
conda install -c conda-forge conda-pack --yes # install here, for the unpack


# Load module family
module load PrgEnv-cray # also loads cray-mpich and related stuff, will be loaded by default
module load amd-mixed/5.6.0 # will need to match if updating pytorch version
module load craype-accel-amd-gfx90a
module load libfabric
module load libtool # careful with LD_Library paths with this loaded, see RCCL notes below
# module load cce/16.0.1 # doesnt fix flash-attention with C++20 headers

######### COMPILE PIP PACKAGES ########################

# MPI
MPICC="cc -shared" pip install --no-cache-dir --no-binary=mpi4py mpi4py

# pytorch and core reqs
cd "${LITGPT_REPO}"
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/rocm5.6
pip install .
cd ${INSTALLDIR}

# flash attention
pip install packaging ninja numpy
git clone https://github.com/ROCmSoftwarePlatform/flash-attention
cd flash-attention
sed -i 's/c++20/c++17/g' setup.py # Annoying patch for now, there used to be a particular module config that loads a more modern cc version
PYTORCH_ROCM_ARCH='gfx90a' GPU_ARCHS='gfx90a' pip install .
cd ${INSTALLDIR}

# interconnects
mkdir -p ${INSTALLDIR}/tiny_plugins_rccl
git clone https://github.com/ROCmSoftwarePlatform/aws-ofi-rccl
cd aws-ofi-rccl
./autogen.sh
CC=cc CXX=CC ./configure --with-libfabric=/opt/cray/libfabric/1.15.0.0 --with-hip=/opt/rocm-5.6.0/ \
                         --with-rccl=${CONDA_PREFIX}/lib/python3.11/site-packages/torch/lib/ \
                         --prefix=${INSTALLDIR}/tiny_plugins_rccl
CC=cc CXX=CC make -j install
cd ${INSTALLDIR}

# Finally axonn
# pip install axonn
git clone https://github.com/axonn-ai/axonn
cd axonn
git checkout 3a3c5386c48a889e4ae1f81acfd51ea1bc7f6f98
pip install .
cd ${INSTALLDIR}


# Clean-up
cd ${INSTALLDIR}
rm -rf axonn
rm -rf flash-attention
rm -rf aws-ofi-rccl

######### PACK A STATIC COPY OF THE ENVIRONMENT ########################
# This step needs to be repeated if the env is changed

# Pack up the entire thing
cd ${INSTALLDIR}
rm -f ${ENV_NAME}_env_packed.tar.gz
conda pack -p ${INSTALLDIR}/$ENV_NAME -o ${ENV_NAME}_env_packed.tar.gz --compress-level=1