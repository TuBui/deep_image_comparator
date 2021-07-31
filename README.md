# Deep Image Comparator: Learning to Visualize Editorial Change

![Python 3.8](https://img.shields.io/badge/Python-3.8-green) ![Pytorch 1.7.0](https://img.shields.io/badge/Pytorch-1.7.0-green) ![MIT License](https://img.shields.io/badge/Licence-MIT-green)

This repo contains official code, datasets and model for the CVPR 2021 Workshop on Media Forensics paper ["Deep Image Comparator: Learning to Visualize Editorial Change"](https://openaccess.thecvf.com/content/CVPR2021W/WMF/html/Black_Deep_Image_Comparator_Learning_To_Visualize_Editorial_Change_CVPRW_2021_paper.html).

## Dependencies

In order to run training, the following main libraries are required:
```
pytorch == 1.7.0
torchvision == 0.8.0
imagenet-C (see below)
opencv-python >= 4.2.0
Pillow >= 8.0.1
slackclient >= 2.5.0
...

```
The full list of dependencies can be found at [requirements.txt](requirements.txt).

To install imagenet-C:
```
git clone https://github.com/hendrycks/robustness.git && cd robustness/ImageNet-C/imagenet_c/ && pip install -e .
```

We also provide a [Dockerfile](Dockerfile) so that you can build a docker image yourself. Alternatively you can download our pre-built docker image at:

```bash
docker pull tuvbui/image_comparator:latest

```

In order to run inference (extract image feature), only pytorch, torchvision and Pillow are required.

## Download data

```bash
./download_data.sh 

```
This script downloads the PSBattles dataset (15GB) to `./data` and the deepaugmix pretrained model (100MB) to `./pretrained` necessary for the training step below. Make sure you have enough disk space before running this command.


The deepaugmix pretrained model (ResNet50) is obtained from the [official repo](https://github.com/hendrycks/imagenet-r). This is used to initialize our model weight during training.


## Train

To train the retrieval model (phase 1 in the paper):

```python
python train_phase1.py -td ./data -vd ./data -tl data/train_pairs.csv  -vl data/test_pairs.csv -m ResnetModel -vp batch_size=10,nepochs=20,d_model=256,lr=0.001,optimizer=SGD,lr_steps=[0.6] -w pretrained/deepaugment_and_augmix.pth

```

We also release our trained model, which can be downloaded directly from [here](https://cvssp.org/data/Flickr25K/tubui/cvpr21wmf/image_comparator_phase1.pt).


## Inference

To extract the fingerprint of an image:


```python
python inference.py -i example.png -w ./weight/image_comparator_phase1.pt

```

This produces a 256-D float32 descriptor robust to both benign transformations (Imagenet-C and more) and manipulations (photoshop).


## Reference

```
@InProceedings{Black_2021_CVPR,
    author    = {Black, Alexander and Bui, Tu and Jin, Hailin and Swaminathan, Vishy and Collomosse, John},
    title     = {Deep Image Comparator: Learning To Visualize Editorial Change},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2021},
    pages     = {972-980}
}

```