#!/bin/sh

#SBATCH --output=calmip-mgan-sn-%j.out
#SBATCH --error=calmip-mgan-sn-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:1
#SBATCH --mem=20000
#SBATCH --time=24:00:00

# use WPH env to run this code
module purge
module load conda/4.9.2
conda activate pt171env

export KYMATIO_BACKEND=skcuda

cd ~/wavelet-texture-synthesis/TextureNets_implementation

##### april 14 ######
python train_g2d_periodic_modelC_altgda.py -gpu -data tur2a -etag 1e-5 -etad 1e-3 -tau 20 -its 5000 -ch 16 -rand 0 -init normalstdbarx  -runid 1001

