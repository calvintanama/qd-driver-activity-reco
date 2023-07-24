# Quantized Distillation for Driver Activity Recognition

This is the official PyTorch implementation of our IROS 2023 paper:

Quantized Distillation: Optimizing Driver Activity Recognition Models for Resource-Constrained Environments


# Setup

Add folder called `model` in the same directory as above code to save trained checkpoints. Before training, the folder structure should look like this

```
├── [path to your cloned repository]
    ├── architecture
    ├── config
    ├── framework_activity_recognition
    ├── model    # add this
    ├── LICENSE
    ├── README.md
    └── requirement.txt
```
Do not forget to install the requirement stated in the folder.

### Dataset
[Drive&Act](https://driveandact.com/)

### Pretrained Model
[MobileNet3D](https://drive.google.com/drive/folders/1eggpkmy_zjb62Xra6kQviLa67vzP_FR8) | [RGB I3D](https://github.com/hassony2/kinetics_i3d_pytorch/tree/master/model)

### Training
To train baseline RGB I3D model on Drive&Act, use the following command
```
python -m framework_activity_recognition config/train/i3dbaseline.yaml
```
To train other baseline or using knowledge distillation and/or quantization on Drive&Act, replace the yaml file in the command to one of the following yaml file in config/train folder

```
├── ./config
    ├── /train
        ├── i3dbaseline.yaml           # RGB I3D baseline on Drive&Act
        ├── mobilenet_quant.yaml       # MobileNet3D with PyTorch quantization on Drive&Act
        ├── mobilenetbaseline.yaml     # MobileNet3D baseline on Drive&Act
        ├── studentteacher.yaml        # MobileNet3D on Drive&Act with knowledge distillation from teacher RGB I3D
        └── studentteacher_quant.yaml  # MobileNet3D with PyTorch quantization and knowledge distillation from teacher RGB I3D on Drive&Act
```

### Test
To test RGB I3D Model with test split of Drive&Act, use the following command
```
python -m framework_activity_recognition config/test/i3dtest.yaml
```
To test another model, replace the yaml file in the command with one of the following yaml file in config/test folder

```
├── ./config
    ├── /train
        ├── i3dtest.yaml             # RGB I3D test on Drive&Act test split
        ├── mobilenetquanttest.yaml  # MobileNet3D with PyTorch quantization test on Drive&Act test split
        └── mobilenettest.yaml       # MobileNet3D test on Drive&Act test split
```

