#!/bin/bash
# =============================================================================
# run_final.slurm  --  upper-limit runs on the cluster (folder ~/version_final)
# =============================================================================
# One job = one node, 32 cores, ProcessPoolExecutor (no MPI). The (mass, pulsar)
# pairs are dealt to the N_JOBS jobs by stride. Idempotent: resubmitting the
# same sbatch (same COUPLING/JOB_ID/N_JOBS) skips what is done.
#
#   sbatch --export=ALL,COUPLING=alpha,JOB_ID=0,N_JOBS=2 run_final.slurm
#
# Output: ~/version_final/out_<COUPLING>/hist/m_<mass>__<pulsar>__s0.npz
# The raw chains go to the node's local disk and are deleted one by one.
# =============================================================================
#SBATCH --job-name=uldm_final
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/%x_%j.out

module load Anaconda3/5.1.0
source /opt/ohpc/pub/apps/software/Anaconda3/5.1.0/etc/profile.d/conda.sh
conda activate pulsares 2>/dev/null || true
PY="$HOME/.conda/envs/pulsares/bin/python"
"$PY" -c "import sys,scipy,enterprise,PTMCMCSampler; print('USING',sys.executable,'| scipy',scipy.__version__)"

cd "$HOME/version_final"
mkdir -p logs

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export OMPI_MCA_mca_base_component_show_load_errors=0 PMIX_MCA_mca_base_component_show_load_errors=0

: "${COUPLING:?COUPLING is not set (alpha|beta|vec)}"
export NANOGRAV_ROOT="${NANOGRAV_ROOT:-$HOME/vec2}"
export OUT_ROOT="$HOME/version_final/out_${COUPLING}"
export CHAIN_TMP="${TMPDIR:-/tmp}/uldm_final_${SLURM_JOB_ID:-local}"
export FROZEN_PARAMS="${FROZEN_PARAMS:-PBDOT}"
export N_ITER="${N_ITER:-200000}"
export JOB_ID="${JOB_ID:-0}" N_JOBS="${N_JOBS:-1}"
mkdir -p "$OUT_ROOT" "$CHAIN_TMP"

echo "host=$(hostname)  cpus=$SLURM_CPUS_PER_TASK  COUPLING=$COUPLING job $JOB_ID/$N_JOBS  start $(date)"
"$PY" -u worker_final.py
rm -rf "$CHAIN_TMP"
echo "END $(date)"
