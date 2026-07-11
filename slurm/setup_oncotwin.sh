module load miniforge/25.11.0-0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate oncotwin

echo "Using Python in Slurm job:"
which python
python --version
which python3
python3 --version

python3 - <<'PY'
import pydicom, numpy
print("pydicom", pydicom.__version__)
print("numpy", numpy.__version__)
PY
