# Training AlexNet on ImageNet-1K

This repository provides code for training AlexNet on ImageNet-1K using PyTorch, with support for Distributed Data Parallel (DDP) training.  Please refer to the [ResNet benchmarking section](https://github.com/KempnerInstitute/scalable-vision-workflows/imagenet1k_resnet50)
for detailed information on software setup and data preparation.  This section highlights the key differences involved in running the AlexNet model, as well as its execution time, compared to ResNet.

To run AlexNet using the provided Slurm job script, set the following environment variable:

```
ARCHITECTURE=alexnet
```


| GPU | Batch Size | #N  | #GPUs | LR  | WCT (s) | WCT (h:m:s) | Val Acc@1 | Val Acc@5 | Scaling Efficiency |
| --- | ---------- | --- | ----- | --- | ------- | ----------- | --------- | --------- | ------------------ |
| A100 | 1024      | 1   | 4     | 0.1 | 14607   | 04:03:27    |  53.864     | 76.980  |       TBD        |



