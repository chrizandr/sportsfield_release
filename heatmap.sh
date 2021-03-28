#!/bin/bash
#SBATCH -A chrizandr
#SBATCH --nodelist=gnode03
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH -n 8
#SBATCH --time=2-00:00:00

module load gflags/2.2.1 &&

module load glog/0.3.5 &&

module load cuda/10.2 && module load cudnn/7.6.5-cuda-10.2 &&

module load openmpi/4.0.0 &&

module load python/3.7.4


source ~/mmd/bin/activate


echo "Copying data files"
if [ -d "/ssd_scratch/cvit/chrizandr" ]
then
  echo "Clearing existing files on node";
  rm -r /ssd_scratch/cvit/chrizandr
fi

images="england_vs_croatia"
prefix="envscr"
annot="/home/chrizandr/annot/envscr_out.xml"

mkdir -p /ssd_scratch/cvit/chrizandr/
rsync -az chrizandr@ada:/share3/chrizandr/sports/dataset/$images/images/ /ssd_scratch/cvit/chrizandr/images

echo "Done copying data files"


python heatmap.py --savefile heatmap_`echo $prefix`.pkl --data /ssd_scratch/cvit/chrizandr/images --annot $annot

# frvscr 51
# frvsbe 50
