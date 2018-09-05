#!/bin/bash

# Usage:
# sbatch PyTorch/script.sh Images/styles/starrynight.jpg Images/content/ace_pablo.jpg

# Confuguration for multiple GPU on a single Machine

#SBATCH -n 1                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -t 0-00:30          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu_requeue      # Partition to submit to

# Testing mode - quickly allocation of resources
#SBATCH --mem=30000          # Memory pool for all cores (see also --mem-per-cpu)

# Uncomment to full power
#SBATCH --gres=gpu:2        # Activate n GPU (let's say 8)

#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

module load Anaconda3/5.0.1-fasrc02

if [ $# -lt 2 ]
then
  echo [Error]: Missing arguments
  echo
  echo 2 arguments required:
  echo
  echo [1] Style Image PAth
  echo
  echo [2] Content Image Path
  exit
fi

echo Activating environment torch37...
module load cuda/9.0-fasrc02 cudnn/7.0_cuda9.0-fasrc01
CONDA="torch37"

# Output information
manage_env () {
  source activate $CONDA
  which python
  python -V
}

# Launch notebook
manage_env
echo Loading Script...

python PyTorch/style_transfer.py -s $1 -c $2

exit
