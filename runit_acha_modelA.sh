# use WPH env to run this code
# install kymatio_wph3 using python setup.py install

#cd kymatio_wph3/projetLip1
#matlab -nodesktop -nosplash -r maxent_modelA

. ~/activate_wph.sh
export KYMATIO_BACKEND=skcuda

cd TextureNets_implementation
python synthesis_modelA_micro.py -data tur2a -gpu


