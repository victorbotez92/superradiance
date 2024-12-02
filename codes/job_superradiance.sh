#!/bin/bash 
#SBATCH --job-name=superr
#SBATCH -o ./JobLogs/%x.o
#SBATCH -e ./JobLogs/test.err
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --partition=cpu_short
#SBATCH --time=1:00:00


cd ${SLURM_SUBMIT_DIR}


date
module purge
module load anaconda3/2022.10/gcc-11.2.0
module load openmpi/3.1.6/gcc-11.2.0

#superradiance_env
#source /gpfs/users/botezv/.venv/pod/bin/activate

#python POD_fourier_parallel.py 


set -x

date

data_directory="my_directory"

srun python /gpfs/workdir/botezv/parallel_supper/codes/main_parallel.py "$data_directory"
echo 'running superradiance'

wait

date