![icon](https://github.com/jiaojiaoguan/phagenus/assets/43172888/16438222-6fe5-4f32-8eaa-ec4d4c6cb182)

## Overview

PhaGenus is a learning-based model, which conducts genus-level taxonomic classification for phage contigs. It utilizes a powerful transformer model to learn the association between protein clusters and support the classification of up to 508 genera.

The input of the program should be fasta files and the output will be a csv file showing the predictions. Since it is a deep learning model, if you have GPU units on your PC, we recommand you to use them to save your time.

If you have any trouble installing or using PhaGenus, please let us know by emailing us (jiaojguan2-c@my.cityu.edu.hk).

## Quick install
Note: we suggest you to install all the package using conda (both miniconda and Anaconda are ok).

After cloning this respository, you can use anaconda to install the phaGenus.yaml. This will install all packages you need with gpu mode (make sure you have installed cuda on your system to use the gpu version). We use multiple GPU to accelerate the training. Therefore, you need to use more than one GPU to run PhaGenus.

### Prepare the database and environment

Due to the limited size of the GitHub, we zip the database. You can download the database and model from Google Drive or Baidu Netdisk(百度网盘). 
You can follow steps bellow to install Phagenu.

### Step1: Download the code.

       wget https://github.com/jiaojiaoguan/phagenus/archive/refs/heads/main.zip
       unzip main.zip

### Step2: Install the conda environment.

       cd phagenus-main/
       conda env create -f phagenus.yaml -n phagenus
       conda activate phagenus

### Step3: Download the database and model.
       
      from the Google Drive:
      https://drive.google.com/file/d/1_nxQD6Q87tM2y1biAXKtZcrTEzINwH-w/view?usp=share_link

      from Baidu NetDisk(百度网盘):
      链接: https://pan.baidu.com/s/1J7CE119QacBSOuZ0h02ydQ?pwd=genu 提取码: genu
      
      Note: You need to put the "data" folder in "phagenus/".
      
### Step4: Run PhaGenus model.
       An example command is provided below in the "job.sh" file. You may use "sbatch job.sh" to execute it.
       
       python phagenus.py [--contigs INPUT_FA] [--out OUTPUT_CSV] [--midfolder DIR]

                     --contigs INPUT_FA   input fasta file
                     --len MINIMUM_LEN    predict only for sequence >= len bp (default 3000)  
                     --out OUTPUT_CSV     The output csv file (prediction_output.csv in midfolder)                
                     --midfolder DIR      Folder to store the intermediate files (default midfolder/)
                     --reject             The uncertainty threshold (default 0.035)
                     --sim                We prepare two database and model for different scenes. You can choose high-smilarity or low-similarity database and model. The paremeters are "high" or "low". (default high)

## Example

       python phagenus.py --contigs test_contigs.fasta --midfolder test_phagenus --sim high

       The prediction will be written in prediction_output.csv. The CSV file has three columns: contigs, prediction, and uncertainty score. 

### Contact
If you have any questions, please email us: jiaojguan2-c@my.cityu.edu.hk
