seed=$1

CUDA_VISIBLE_DEVICES=1 python3 ./main.py --ver 5.0 --cudaDevice 0 --alr 1e-5 --clr 1e-4 --seed $seed