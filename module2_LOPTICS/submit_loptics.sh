#!/bin/bash
#SBATCH --job-name="LRMO_optics"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your@email.com  # The default value is the submitting user.
#SBATCH --partition=epyc96
#SBATCH -N 1       # Minimum of 2 nodes
#SBATCH -n 96     # 24 MPI processes per node, 48 tasks in total, appropriate for xeon24 nodes
#SBATCH --time=50:00:00
#SBATCH --output=LRMO_optics.log



# Running on epyc96
 module use /home/modules/energy/modules/all
 module load VASP/6.4.2-foss-2023a
 module load Python/3.11.3-GCCcore-12.3.0
 module load ASE/3.22.1-foss-2023a
# export VASP_COMMAND="mpirun vasp_gam"

# Running on all other partitions 
module use /home/modules/energy/modules/all

#module load VASP/6.5.1-intel-2025b
##module load Python/3.13.5-GCCcore-14.3.0
#module load ASE/3.23.0-iimkl-2023a

#export VASP_COMMAND="mpirun vasp_gam"

python module2_loptics.py


