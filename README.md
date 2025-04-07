# Efficient and Scalable ResNet Training on ImageNet

Efficient and Scalable ResNet Training on ImageNet is a benchmarking project focused on training ResNet-50 efficiently across multiple A100 GPUs using PyTorch’s Distributed Data Parallel (DDP) on the SLURM-managed Kempner AI Cluster. The project demonstrates best practices for distributed training, including proper SLURM integration, environment setup, and optimized data loading. It includes a fast, multithreaded extraction utility for ImageNet preprocessing and evaluates scalability by measuring wall-clock time and validation accuracy at different GPU scales. The goal is to achieve strong performance and near-linear scaling while preserving model accuracy, providing reproducible results for both research and operational use.


## Dataset

In this project, we use the ImageNet-1k dataset, which consists of 1.2 million training images and 50,000 validation images. The images are divided into 1,000 classes. The dataset is available for download from the [ImageNet](https://image-net.org/) website.

### Dowloading the dataset

To download the ImageNet data, you need to have an account on the ImageNet website. 

Create an account here: [Imagenet Website](https://image-net.org/index.php)

After creating an account, click on the Download tab. You will see that there are many ImageNet datasets available for download. In this benchmarking project, we focus on the ILSVRC2012 dataset. There are three tasks available for download: Classification, Detection, and Localization. We focus on the Classification task. So we download the Training images (Task 1 & 2) and the Validation images (Task 1 & 2) for the ILSVRC2012 dataset. Here are the MD5 checksums for the downloaded files:

 | File Name                 |  MD5 Checksum                     | Size   |
 | ------------------------- | --------------------------------- | ------ | 
 | ILSVRC2012_img_train.tar: |`1d675b47d978889d74fa0da5fadfb00e` | 138 GB | 
 | ILSVRC2012_img_val.tar:   |`29b22e2961454d5413ddabcf34fc5622` | 6.3 GB |

Use the following commands to download the files (be sure to use your own username and password): 


> [!WARNING]
> Please start an interactive session (or submit a batch job) on the cluster to run and preprocess any data. 
> For more details read the [Interactive Jobs](https://handbook.eng.kempnerinstitute.harvard.edu/s1_high_performance_computing/kempner_cluster/accessing_gpu_by_fasrc_users.html#interactive-jobs) from the [handbook](https://handbook.eng.kempnerinstitute.harvard.edu/intro.html). 
> Processing or downloading data on the login node is not allowed and can lead to your account being suspended. 

```bash
wget --user=your_username --password=your_password https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
wget --user=your_username --password=your_password https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
```

> [!TIP]
> Run the following commands to verify the integrity of the downloaded files. The MD5 checksums should be the same as the ones listed above. 
>
>  ```bash
>  md5sum ILSVRC2012_img_train.tar
>  md5sum ILSVRC2012_img_val.tar
>  ```

### Extracting the data

#### Option 1: Using the shell script (slow)

Pytorch provides a shell script for extracting the downloaded files, which can be found [here](https://github.com/pytorch/examples/tree/main/imagenet). A copy of the script is also included in the `scripts` directory.

```bash
scripts/extract_ILSVRC.sh
```

To run this script, ensure that the `ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar` files are in the same directory as the script. However, this script is not very efficient and can take a long time to extract the data. 

#### Option 2: Using the multithreaded Python script (fast)

We developed a Python version of the script which is significantly faster and utilizes multiple CPU cores. The script is located in the scripts directory.

- Extracting both training and validation data

```bash
python scripts/extract_ILSVRC_mp.py \
       --train_tar /path/to/ILSVRC2012_img_train.tar \
       --val_tar /path/to/ILSVRC2012_img_val.tar \
       --output_dir /path/to/output \
       --valprep_script /path/to/valprep.sh \
       --num_threads 8
       --run_type all
```

- Extracting only training data

```bash
python scripts/extract_ILSVRC_mp.py \
       --train_tar /path/to/ILSVRC2012_img_train.tar \
       --output_dir /path/to/output \
       --num_threads 8
       --run_type train
```

- Extracting only validation data

```bash
 python scripts/extract_ILSVRC_mp.py \
       --val_tar /path/to/ILSVRC2012_img_val.tar \
       --output_dir /path/to/output \
       --valprep_script /path/to/valprep.sh \
       --num_threads 8
       --run_type val
```


## Environment

In order to run the code on Kempner AI Cluster, you need to have the following environment set up.

- Load the modules

```bash
module load python/3.12.8-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01
```

- Create a conda environment

```bash
conda create --name resnet_pytorch_env python=3.12.8
conda activate resnet_pytorch_env
```

- Install the required packages

```bash
pip3 install torch torchvision torchaudio
```

Please not that `requirements.txt` and `environment.yml` files are provided for documentation purposes only. It is simpler to follow the instructions above to set up the environment. However, if you want to use the `requirements.txt` file or the `environment.yml` file, you can use the following command to install the required packages:

- Installing the required packages using `requirements.txt` (you need to create the conda environment first):

  ```bash
  pip3 install -r requirements.txt
  ```

- Installing the required packages using `environment.yml`:

  ```bash
  conda env create -f environment.yml
  ```

## Training the model

This project is based on the official PyTorch implementation of ResNet-50 using Distributed Data Parallel (DDP) training. for more details see the following link in the PyTorch Github repository:

- https://github.com/pytorch/examples/blob/main/imagenet/main.py

We modified the code to integrate seamlessly with SLURM schedular using `srun` for job execution. When training PyTorch models in distributed mode on SLURM clusters with `srun`, SLURM launches one process per GPU (or per task). Therefore:

- Using `torch.multiprocessing.spawn()` is redundant and problematic, as it spawns extra processes inside SLURM-launched tasks.
- GPU assignment, rank, and world size must be derived from SLURM environment variables (`SLURM_PROCID`, `WORLD_SIZE`, etc.).

The table below summarizes the results of training ResNet-50 on A100 GPUs using the PyTorch implementation. Submission scripts can be find in [run_results/A100](run_results/A100) directory and the run logs can be find in [run_results/A100/logs](run_results/A100/logs) directory. For all experiments, we use 16 CPUs per task for data loading, enabled `pin_memory` option, and used the default `prefetch_factor` of `2`. Each GPU was assigned a batch of 256 samples. 

We compute the scaling efficiency as the ratio of the time taken to train the model on 1 GPU to the time taken to train the model on N GPUs, where N is the number of GPUs used in the run. The scaling efficiency is a measure of how well the model scales with the number of GPUs used.

$$
Scaling\ Efficiency = \frac{T_{1\ GPU}}{T_{N\ GPUs} \times N} \times 100
$$

where $T_{1\ GPU}$ is the time taken to train the model on 1 GPU and $T_{N\ GPUs}$ is the time taken to train the model on $N GPUs$.

| GPU | Batch Size | #N  | #GPUs | LR  | WCT (s) | WCT (h:m:s) | Val Acc@1 | Val Acc@5 | Scaling Efficiency |
| --- | ---------- | --- | ----- | --- | ------- | ----------- | --------- | --------- | ------------------ |
| A100 | 256       | 1   | 1     | 0.1 | 129686  | 36:01:26    | 75.392    | 92.470    |       100%         |
| A100 | 512       | 1   | 2     | 0.1 | 65752   | 18:15:52    | 75.356    | 92.600    |       98.62%       |
| A100 | 1024      | 1   | 4     | 0.1 | 33216   | 09:13:36    | 74.672    | 92.054    |       97.61%       |
| A100 | 2048      | 2   | 8     | 0.1 | 17104   | 04:45:04    | 73.308    | 91.204    |       94.78%       |
| A100 | 4096      | 4   | 16    | 1.6 | 8914    | 02:28:34    | 70.342    | 89.346    |       90.93%       |
| A100 | 8192      | 8   | 32    | 3.2 | 4765    | 01:19:25    | 72.550    | 90.910    |       85.05%       |
| A100 | 16384     | 16  | 64    | 6.4 | 2640    | 00:44:00    | 64.208    | 85.644    |       76.76%       | 
| A100 | 32768     | 32  | 128   | 6.4 | TBD     | TBD         | TBD       | TBD       |       TBD          |


Note that using a large batch size can affect the performance of the optimizer. There is a substantial body of research focused on selecting appropriate learning rates for large batch training. Please open an issue if you're interested in learning more about this topic.


