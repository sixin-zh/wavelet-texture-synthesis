# use WPH env to run this code
. ~/activate_wph.sh


cd TextureNets_implementation

##### mai 16 ######
python synthesis_vgg_model_lbfgs_gray.py -gpu -layers 3 -its 30 -Ns 1024


##### old  ######

#python synthesis_vgg_model_lbfgs.py
#python synthesis_vgg_model_lbfgsB.py



#cd vgg_color_caffe
#python example.py

