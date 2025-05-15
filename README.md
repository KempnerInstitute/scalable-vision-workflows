# Efficient and Scalable Vision Model Training

Efficient and Scalable Vision Model Training is a benchmarking and workflow-oriented project designed to help users train deep vision models efficiently across multiple A100 or H100 GPUs using PyTorch’s Distributed Data-Parallel (DDP) on SLURM-managed clusters. Rather than being tied to a specific dataset or architecture, the project allows users to plug in available or their own model and dataset configurations. It demonstrates best practices for scalable training, including SLURM-native job management, environment setup, and optimized data loading strategies. The goal is to achieve strong performance and near-linear scaling while maintaining flexibility and reproducibility across a variety of vision workloads on the Kempner AI cluster.

## Available Workflows

| Workflow                                   | Model     | Dataset     |  Max Tested GPUs  |         Tags                 |
| ------------------------------------------ | --------- | ----------- | ----------------- | ---------------------------- |
| [imagenet1k_resnet50](imagenet1k_resnet50) | ResNet-50 | ImageNet-1k |       64          | `A100`, `DDP`                |
| [imagenet1k_alexnet](imagenet1k_alexnet)   | AlexNet   | ImageNet-1k |       4          | `A100`, `DDP`                |

