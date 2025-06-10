#!/bin/sh

#SBATCH --output=calmip-tnets-%j.out
#SBATCH --error=calmip-tnets-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:1
#SBATCH --mem=20000
#SBATCH --time=24:00:00

#module purge
#module load pytorch/1.7.1

# use WPH env to run this code
module purge
#module load conda
#conda activate pt201env
module load conda/4.9.2
conda activate pt171env

#export KYMATIO_BACKEND=skcuda

cd ~/wavelet-texture-synthesis/TextureNets_implementation

##### june 10 ######
python train_linIdwt2d_periodic_modelA_altgda.py -gpu -init normalstdonly -J 5 -L 4 -dn 1 -wave db7 -data fbmB7 -etag 0.01 -etad 0.1 -tau 5 -bs 16 -ebs 100 -its 5000 -spite 100 -runid 1001

##### april 14 ######
#python train_g2d_periodic_modelC_altgda.py -gpu -data tur2a -etag 1e-5 -etad 1e-3 -tau 20 -its 5000 -ch 16 -rand 0 -init normalstdbarx  -runid 1001

