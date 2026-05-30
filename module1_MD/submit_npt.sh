#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your@email.com
#SBATCH --partition=sm3090el8
#SBATCH -N 1-1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --time=50:00:00
#SBATCH --begin=now+0hour
#SBATCH --job-name=test
#SBATCH --error=test.err
#SBATCH --output=test.out


module load Python/3.13.5-GCCcore-14.3.0
module load ASE/3.26.0-iimkl-2025b
#source /home/energy/jmcro/miniforge3/etc/profile.d/conda.sh


source /home/energy/madil/mace/bin/activate
#export ASE_AIMS_COMMAND="/home/energy/jmcro/vibes/run_aims.sh"

#source /home/energy/jmcro/mace_phonopy/bin/activate

python  module1_npt.py
