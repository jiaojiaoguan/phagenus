#!/bin/bash
#SBATCH -o test.o
#SBATCH -e test.e
#SBATCH -J test
#SBATCH -p team3
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=20G
#SBATCH --export=all
#SBATCH --gres=gpu:2

python phagenus.py --contigs test_contigs.fasta --midfolder test_phagenus --sim high
