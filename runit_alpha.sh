# use GAN env to run this code

# use DAN env to generate filters: python build-filters.py

#python synthesis/gray.py -i tur2d_N512_sample -fmt mat --N 512 --J 6 --L 4
python synthesis/gray.py -i fig26_input_512_gray -fmt jpg --N 512 --J 6 --L 4

#python synthesis/color.py --image fig26_input_512 --nb_chunks 32 --N 512 --J 5 --L 4
