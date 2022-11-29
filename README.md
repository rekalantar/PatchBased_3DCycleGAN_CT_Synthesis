# 3D Cycle-GAN TensorFlow

Cycle-consistent generative adversarial network (Cycle-GAN) is an unsupervised approach for image synthesis. In medical imaging, it promises to provide a tool for intricate data augmentation. Although some clinical studies report that GANs may generate unrealistic features in some cases, they are still promising tools for image generation with high visual fidelity. This repository provides the code for 3D patch-based synthesis of medical images using a Cycle-GAN network. 

![](https://github.com/rekalantar/CycleGAN3D_Tensorflow/blob/main/images/contrastremoval.gif)

## Usage
Create a virtual environment to install the prerequisites. If there are issues with tensorflow-gpu, cuda and cuDNN version mismatch, it is recommended to use Anaconda or conda-forge to install the requirements. The working versions of the Nvidia cuda driver, tensorflow-gpu, cudatoolkit and cudnn can be found [here](https://medium.com/@rekalantar/gpu-enabled-tensorflow-pytorch-setup-without-manually-installing-cuda-and-cudnn-conda-forge-52cf43b6ddd6). 

```bash
conda create -n tfgpu
conda activate tfgpu
conda install tensorflow-gpu -c conda-forge
```

The code benefits from the [Monai Library](https://monai.io/) that is a Torch-based medical imaging library for custom preprocessing and caching of medical images. The preprocessing operations are performed in dataloader.py with specifications defned in the config.py file. 

The expected directories are as follows:

```bash
|--directory
       |----train
              |----A
                   |----xxx.nii.gz, xxx.nii.gz, ...
              |----B
                   |----xxx.nii.gz, xxx.nii.gz, ...
       |----test
              |----A
                   |----xxx.nii.gz, xxx.nii.gz, ...
              |----B
                   |----xxx.nii.gz, xxx.nii.gz, ...
```

Train:
```bash
python main.py path/to/data/directory path/to/save/results
```
