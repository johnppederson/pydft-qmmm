#!/usr/bin/bash
#SBATCH -J vasp_qmmm
#SBATCH --account=gts-jmcdaniel43-chemx
#SBATCH -N 1 -n 1
#SBATCH --cpus-per-task=8
#SBATCH -p gpu-h200
#SBATCH -q inferno
#SBATCH --gres=gpu:h200:1
#SBATCH -t 1:00:00
#SBATCH --mem-per-gpu=24G

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load nvhpc-openmpi3/24.1
module load intel-oneapi-mkl/2023.1.0

PYTHON_ENV=/storage/home/hcoda1/1/cshao48/.conda/envs/vasp_qmmm
VASP=/storage/project/r-jmcdaniel43-0/cshao48/install/vasp/vasp.6.6.1_nvhpc24/bin/vasp_std

export OMP_NUM_THREADS=1
export OPENMM_CPU_THREADS=$SLURM_CPUS_PER_TASK
export OMP_STACKSIZE=512m
ulimit -s unlimited

export LD_LIBRARY_PATH="${MKLROOT:+$MKLROOT/lib/intel64:}${LD_LIBRARY_PATH:-}:$PYTHON_ENV/lib"
export PYTHONHOME="$PYTHON_ENV"
export PYTHONPATH="$(realpath ../..):${PYTHONPATH:-}"
export PYDFT_QMMM_VASP_NUCLEAR_FIELD=spline
export PATH="$PYTHON_ENV/bin:$PATH"

export VASP_PP_PATH=/storage/project/r-jmcdaniel43-0/cshao48/install/vasp/potpaw_PBE.64
export PYDFT_QMMM_VASP_COMMAND="mpirun -np 1 $VASP"

python run.py
