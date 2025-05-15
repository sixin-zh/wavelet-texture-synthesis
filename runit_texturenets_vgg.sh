# use WPH env to run this code
. ~/activate_wph.sh
cd TextureNets_implementation

##### mai 15 ######
#python train_g2d_periodic_vgg_gray.py -gpu -layers 3 -lr 0.1
python train_g2d_periodic_vgg_gray.py -gpu -layers 3 -lr 0.01

##### old ######

#python train_g2d_periodic_vgg_gray.py -gpu -lr 0.1
#python train_g2d_periodic_vgg_gray.py -gpu -lr 0.01
#python train_g2d_periodic_vgg_gray.py -gpu -lr 0.001
#python train_g2d_periodic_vgg_gray.py -gpu -lr 0.0001

#python train_g2d_periodic_vgg_color.py
#python train_g2d_periodic_noGN.py
