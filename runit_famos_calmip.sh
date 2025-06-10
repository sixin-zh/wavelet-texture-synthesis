#!/bin/sh

#SBATCH --output=calmip-famos-%j.out
#SBATCH --error=calmip-famos-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:1
#SBATCH --mem=20000
#SBATCH --time=30:00:00

module purge
module load pytorch/1.7.1

cd ~/wavelet-texture-synthesis/famos/

########## Sep 1 ##########
python train_psgan.py -gpu -lr 5e-5  -its 100 -gzloc 20 -gzgl 0 -runid 2001


########## Mai 28 ##########
#python train_psgan_wgan.py -gpu -gzloc 30 -gzgl 0 -lrD 5e-5 -lrG 5e-5 -la 10 -its 25 -runid 2008


########## Mai 27 ##########
#python train_psgan_wgan.py -gpu -lrD 5e-5 -lrG 5e-5 -la 10 -its 25 -runid 2007

#python train_psgan_wgan.py -gpu -lrD 0.0001 -lrG 0.0001 -la 10 -its 25 -runid 2006

#python train_psgan_wgan.py -gpu -lrD 0.0002 -lrG 0.0002 -la 10  -its 25 -runid 2005

#python train_psgan_wgan.py -gpu -lrD 0.0002 -lrG 0.0002 -la 0.1  -its 25 -runid 2004

#python train_psgan_wgan.py -gpu -lrD 0.0002 -lrG 0.0002 -la 1  -its 25 -runid 2003

#LOAD="ckpt/samples/turbulence/ns_randn4_train_N256_wgan_gp/nDep=5&nDepD=5&ngf=120&zLoc=10&zGL=20&ndf=120&bs=16&ims=160&lrD=0.0002&lrG=0.0002&crits=5&beta1=0.5&epoch=25&la=1.0&runid=2001"
#python train_psgan_wgan.py -gpu -lrD 0.0001 -lrG 0.0002 -la 1 -its 1 -load $LOAD -runid 2002

#########################

########## OLD ##########
#python train_psgan_wgan.py -gpu -lrD 0.0002 -lrG 0.0002 -la 1  -its 25 -runid 2001

#python test_psgan_tur2a.py
#########################
