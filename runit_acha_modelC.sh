# use WPH env to run this code
. ~/activate_wph.sh
cd kymatio_wph3/projetLip1
export KYMATIO_BACKEND=skcuda

python modelC_lbfgs2.py

