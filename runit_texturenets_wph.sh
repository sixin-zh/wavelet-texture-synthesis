# use WPH env to run this code
# install kymatio_wph3 using python setup.py install
. ~/activate_wph.sh
export KYMATIO_BACKEND=skcuda

cd TextureNets_implementation

##### april 28 ######

python train_g2d_lin_periodic_modelA.py -data tur2a -init normal -gpu -lr 0.005 -its 2000

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
