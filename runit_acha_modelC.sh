# use WPH env to run this code
# install kymatio_wph3 using python setup.py install

. ~/activate_wph.sh
export KYMATIO_BACKEND=skcuda

cd TextureNets_implementation

#############
# 12 Mai
#############

python synthesis_modelC_micro.py -data tur2a -gpu -dl 4 -bs 1 -init normalstdbarx -adam -lr 1e-1

#python synthesis_modelC_micro.py -data tur2a -gpu -dl 4 -bs 1 -init normal -adam -lr 1e-1

#python synthesis_modelC_micro.py -data tur2a -gpu -dl 4 -bs 1 -init normal -adam -lr 1e-1

#python synthesis_modelC_micro.py -data tur2a -gpu -dl 4 -bs 1 -init normalstdbarx

## TODO fix modelC dl 1 -> 4
#python synthesis_modelC_micro.py -data tur2a -gpu -dl 4 -bs 1


#python test_tur2a_periodic_modelC.py
#python synthesis_modelC_micro.py -data tur2a -gpu
# OPT fini avec: 0.28438571095466614 500


#cd kymatio_wph3/projetLip1
#python modelC_lbfgs2.py


