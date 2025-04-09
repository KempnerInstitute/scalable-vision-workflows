# Efficient and Scalable ResNet Training on ImageNet

Efficient and Scalable ResNet Training on ImageNet is a benchmarking project focused on training ResNet-50 efficiently across multiple A100 GPUs using PyTorch’s Distributed Data Parallel (DDP) on the SLURM-managed Kempner AI Cluster. The project demonstrates best practices for distributed training, including proper SLURM integration, environment setup, and optimized data loading. It includes a fast, multithreaded extraction utility for ImageNet preprocessing and evaluates scalability by measuring wall-clock time and validation accuracy at different GPU scales. The goal is to achieve strong performance and near-linear scaling while preserving model accuracy, providing reproducible results for both research and operational use.

## Available Workflows

| Workflow                                   | Model     | Dataset     |  Max Tested GPUs  |         Tags                 |
| ------------------------------------------ | --------- | ----------- | ----------------- | ---------------------------- |
| [imagenet1k_resnet50](imagenet1k_resnet50) | ResNet-50 | ImageNet-1k |       64          | `A100`, `DDP`                |

