# use WPH env to run this code
# install kymatio_wph3 using python setup.py install

. ~/activate_wph.sh
cd kymatio_wph3/projetLip1
export KYMATIO_BACKEND=skcuda

python modelC_lbfgs2.py

