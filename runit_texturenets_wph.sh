# use WPH env to run this code
# install kymatio_wph3 using python setup.py install
. ~/activate_wph.sh
export KYMATIO_BACKEND=skcuda

cd TextureNets_implementation

##### april 13 ######
python train_g2d_periodic_modelC_altgda.py -gpu -data tur2a -etag 1e-5 -etad 0.01 -tau 20 -its 5000 -ch 16 -rand 0  -init normalstdbarx  -runid 2

##### april 11 ######

#python train_g2d_periodic_modelC.py -gpu -data tur2a -init normalstdbarx -ch 16 -lr 0.001 -resample 1 -bs 1 -rand 0 -its 10000 -factr 10 -runid 1

#python train_g2d_periodic_modelC.py -gpu -data tur2a -init normalstdbarx -ch 16 -lr 0.001 -resample 0 -bs 1 -rand 0 -its 10000 -factr 10 -runid 1

#LOAD="./ckpt/tur2a_g2d_modelC/J=5&L=8&dn=2&dj=1&dk=0&dl=4&ch=8&lr=0.01&its=2000&bs=1&resample=0&factr=10.0&runid=2&init=normalstdbarx&spite=10&gpu=True&loaddir=0/"

#python train_g2d_periodic_modelC.py -gpu -data tur2a -init normalstdbarx -lr 0.0001 -resample 0 -bs 1 -rand 0 -its 2000 -factr 10 -runid 2 -load $LOAD 

#python train_g2d_periodic_modelC.py -gpu -data tur2a -init normalstdbarx -lr 0.01 -resample 0 -bs 1 -rand 0 -its 2000 -factr 10 -runid 2

#python train_g2d_periodic_modelC.py -gpu -data tur2a -init normalstdbarx -lr 0.01 -resample 0 -bs 1 -rand 0 -its 1000 -factr 10 -runid 2

#python train_g2d_periodic_modelC.py -gpu -data tur2a -init normal -lr 0.01 -resample 0 -bs 1 -rand 0 -its 1000 -factr 10 -ch 16

#python train_g2d_periodic_modelC_altgda.py -gpu -data tur2a -rand 1 -etag 0.01 -etad 0.1 -tau 50 -its 50000 -spite 100

# LOAD="./ckpt/tur2a_g2d_modelC_altgda/J=5&L=8&dn=2&dj=1&dk=0&dl=4&ch=8&rand=1&its=5000&lrD=0.1&lrG=0.005&tau=5&bs=1&factr=10.0&spite=10&runid=1&init=normal&gpu=True&loaddir=0"

#python train_g2d_periodic_modelC_altgda.py -gpu -data tur2a -etag 0.005 -etad 0.1 -tau 50 -its 5000 -rand 1 -load $LOAD

#python train_g2d_periodic_modelC_altgda.py -gpu -data tur2a -etag 0.005 -etad 0.2 -tau 20 -its 5000 -rand 1 -load $LOAD

# python train_g2d_periodic_modelC_altAdamGA.py -gpu -data tur2a -etag 0.00001 -etad 0.1 -tau 5 -its 5000 -rand 1 -load $LOAD


##### april 10 ######

# python train_g2d_periodic_modelC_altgda.py -gpu -data tur2a -etag 0.005 -etad 0.1 -tau 5 -its 5000 -rand 1

#python train_g2d_periodic_modelC_altgda.py -gpu -data tur2a -etag 0.02 -etad 0.1 -tau 5 -its 5000 -rand 1

#python train_g2d_periodic_modelC_altgda.py -gpu -data tur2a -etag 0.01 -etad 0.1 -tau 5 -its 5000

#python train_g2d_periodic_modelA_altgda.py -gpu -data tur2a -etag 0.01 -etad 0.1 -tau 5 -its 5000


##### april 9 ######

# python train_linIdwt2d_periodic_modelA_altgda.py -gpu  -fs 64 -wave db7 -data tur2a -etag 0.1 -etad 0.1 -tau 5 -bs 16 -its 5000

#python train_linIdwt2d_periodic_modelA_altgda.py -gpu  -factr 10 -fs 64 -wave db7 -data tur2a -etag 0.1 -etad 0.1 -tau 5 -its 5000


#python train_linIdwt2d_periodic_modelA_rgda.py -gpu  -factr 10 -fs 64 -wave db7 -data tur2a -ga 0.01 -its 5000

#python train_linIdwt2d_periodic_modelA_rgda.py -gpu  -factr 10 -fs 64 -wave db7 -data tur2a

##### april 8 ######

#python train_linIdwt2d_periodic_modelA.py -gpu  -factr 10 -fs 64 -wave db13 -data tur2a

#python train_linIdwt2d_periodic_modelA.py -gpu  -factr 10 -fs 64 -wave db11 -data tur2a

#python train_linIdwt2d_periodic_modelA.py -gpu  -factr 10 -fs 64 -wave db9 -data tur2a

#python train_linIdwt2d_periodic_modelA.py -gpu  -factr 10 -fs 64 -wave db7 -data tur2a

#python train_linIdwt2d_periodic_modelA_adam.py -gpu -its 1000  -data tur2a 

#python train_linIdwt2d_periodic_modelA.py -gpu -factr 10 -fs 64 -data tur2a -runid 2

#python train_linIdwt2d_periodic_modelA.py -gpu -factr 100 -fs 256 -data tur2a

#python train_linIdwt2d_periodic_modelA_djl.py -factr 100 -fs 64 -data tur2a -gpu -bs 10

# python train_linIdwt2d_periodic_modelA.py -gpu -factr 100 -fs 64 -data tur2a -dn 4

# python train_linIdwt2d_periodic_modelA.py -gpu -factr 100 -fs 128 -data tur2a

#python train_linIdwt2d_periodic_modelA.py -gpu -factr 100 -fs 64 -data tur2a -runid 2

# python train_linIdwt2d_periodic_modelA.py -gpu -factr 100 -fs 64 -data tur2a

##### april 28 ######

#python train_g2d_lin_periodic_modelA.py -data tur2a -init normal -gpu -lr 0.005 -its 2000

##### april 27 ######

#python train_g2d_periodic_modelA.py -data tur2a -init normal -gpu

#python train_g2d_periodic_modelC.py -data tur2a -init normal -gpu -lr 0.001
#LOAD="ckpt/tur2a_g2d_modelC/J=5&L=8&dn=2&dj=1&dk=0&dl=4&ch=8&lr=0.001&its=500&bs=16&factr=1.0&runid=1&init=normal&spite=10&gpu=True/trained_gen_model.pt"
#python train_g2d_periodic_modelC.py -data tur2a -init normal -gpu -lr 0.0001 -load $LOAD


##### april 25 ######
#python train_g2d_periodic_modelC.py -data tur2a -init normal -gpu
#python test_tur2a_periodic_modelC.py

###### april 18 ######
#python train_g2d_periodic_modelC.py -data tur2a -gpu -factr 10 -lr 0.001 -its 1000

###### april 16 ######
#python train_g2d_periodic_modelC.py -data tur2a -gpu -factr 10 -lr 0.001
#python train_g2d_periodic_modelC.py -data tur2a -gpu -factr 10 

###### april 15 ######
#python train_fbm_periodic_modelA.py -data tur2a -fs 81 -gpu
#python train_g2d_periodic_modelA.py -data tur2a -gpu 

#python train_g2d_periodic_modelA.py -data tur2a -gpu -runid 2
# TODO ./ckpt/tur2a_g2d/J=5&L=8&fs=101&dn=2&lr=0.01&its=500&bs=16&factr=1.0&runid=2&init=normalstdbarx&spite=10&gpu=True/training_500

#python train_g2d_periodic_modelC.py -data tur2a -gpu 


#python train_fbm_periodic_modelA.py
