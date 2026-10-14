#!/bin/bash
# =============================================================================
# run_final.slurm  --  corridas finales (cotas y Bayes factor), carpeta ~/version_final
# =============================================================================
# Un job = 1 nodo, 32 cores, ProcessPoolExecutor (sin MPI). Reparte los pares
# (masa, pulsar) por stride entre N_JOBS. Idempotente: relanzar el MISMO sbatch
# (mismos ACOPLE/MODO/JOB_ID/N_JOBS) saltea lo ya hecho.
#
#   sbatch --export=ALL,ACOPLE=alpha,MODO=cotas,JOB_ID=0,N_JOBS=2 run_final.slurm
#
# Salida: ~/version_final/out_<ACOPLE>_<MODO>/hist/m_<masa>__<pulsar>__s0.npz
# Las cadenas crudas van al disco local del nodo y se borran al terminar cada una.
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

: "${ACOPLE:?falta ACOPLE (alpha|beta|vec)}"
: "${MODO:?falta MODO (cotas|bf)}"
export NANOGRAV_ROOT="${NANOGRAV_ROOT:-$HOME/vec2}"
export OUT_ROOT="$HOME/version_final/out_${ACOPLE}_${MODO}"
export CHAIN_TMP="${TMPDIR:-/tmp}/uldm_final_${SLURM_JOB_ID:-local}"
export CONGELAR_SECULARES="${CONGELAR_SECULARES:-PBDOT}"
export N_ITER="${N_ITER:-200000}"
export JOB_ID="${JOB_ID:-0}" N_JOBS="${N_JOBS:-1}"
mkdir -p "$OUT_ROOT" "$CHAIN_TMP"

echo "host=$(hostname)  cpus=$SLURM_CPUS_PER_TASK  ACOPLE=$ACOPLE MODO=$MODO job $JOB_ID/$N_JOBS  inicio $(date)"
"$PY" -u worker_final.py
rm -rf "$CHAIN_TMP"
echo "FIN $(date)"
