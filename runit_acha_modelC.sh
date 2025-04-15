# use WPH env to run this code
# install kymatio_wph3 using python setup.py install

. ~/activate_wph.sh
export KYMATIO_BACKEND=skcuda

#cd kymatio_wph3/projetLip1
#python modelC_lbfgs2.py

cd TextureNets_implementation
python test_tur2a_periodic_modelC.py

