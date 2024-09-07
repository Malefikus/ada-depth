#!/bin/bash

#SBATCH -p gpu20
#SBATCH --gres gpu:2
#SBATCH -o /BS/contact-human-pose/work/monodepth2/tmp/slurm-%A_%a.out
#SBATCH -t 3-0

cmd="python train.py --model_name dgp_sup --dataset dgp --data_path /BS/contact-human-pose2/static00/ddad_train_val/ddad.json --width 640 --height 384 --batch_size 4"

echo $(date)
echo $cmd

$cmd
