#!/bin/bash

#SBATCH -p gpu20
#SBATCH --gres gpu:1
#SBATCH -o /BS/contact-human-pose/work/monodepth2/tmp/slurm-%A_%a.out
#SBATCH -t 1:30:00

cmd="python adaptation.py --model_name kitti2dgp --dataset dgp --load_weights_folder ./exp_logs/kitti_sup/models/weights_19 --models_to_load encoder depth --reg_path ./exp_logs/kitti_unsup/models/weights_19 --thres 0.4 --learning_rate 1e-5 --num_workers 0"

echo $(date)
echo $cmd

$cmd
