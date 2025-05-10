# use WPH env to run this code
# install kymatio_wph3 using python setup.py install
. ~/activate_wph.sh
export KYMATIO_BACKEND=skcuda

cd TextureNets_implementation

##### april 9 ######

python train_linIdwt2d_periodic_modelA_altgda.py -gpu  -fs 64 -wave db7 -data tur2a -etag 0.1 -etad 0.1 -tau 5 -bs 16 -its 5000


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
