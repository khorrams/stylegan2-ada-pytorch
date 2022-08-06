#!/bin/bash
#SBATCH -J gncvd
#SBATCH -A eecs
#SBATCH -p dgx2
#SBATCH -o logs/out_mn.out
#SBATCH -e logs/out_mn.err
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 6-23:59:59
#SBATCH --gres=gpu:4
source ../../gancvd/stylegan2-ada-pytorch/env/bin/activate
python train.py --outdir=training_runs_mnist --data=datasets/mnist --gpus=4
#python dataset_tool.py --source ../data/inpainted/train/ --dest data_ --transform center-crop --width 256 --height 256
