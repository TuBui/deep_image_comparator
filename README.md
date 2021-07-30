# Deep Image Comparator: Learning to Visualize Editorial Change

This repo contains official code and model for the CVPR 2021 Workshop on Media Forensics paper "Deep Image Comparator: Learning to Visualize Editorial Change".

## Dependencies
This repo requires the following libraries:
```
pytorch == 1.7.0
torchvision == 0.8.0
imagenet-C
opencv-python
faiss-gpu == 1.6.3
...

```
The full list of dependencies can be found at [requirements.txt](requirements.txt).

We also provide a Dockerfile to build a docker image yourself. Alternatively you can download our docker image at:

```bash
docker pull tuvbui/image_comparator:latest

```


## Download data
```bash
./download_data.sh 

```
This script downloads the PSBattles dataset and the deepaugmix pretrained model necessary for the training step below.


The deepaugmix pretrained model (ResNet50) can be downloaded from the [official repo](https://github.com/hendrycks/imagenet-r). This is used to initialize the model weight for our training.


## Train
To train the retrieval model (phase 1 in the paper):

```python
python train.py -td data/psbattles -vd data/psbattles -tl data/psbattles/train_pairs.csv  -vl data/psbattles/test_pairs.csv -m ResnetModel -vp batch_size=10,nepochs=20,d_model=256,lr=0.001,optimizer=SGD,lr_steps=[0.6] -w pretrained/deepaugment_and_augmix.pth

```

The trained model can be downloaded directly from [here](https://cvssp.org/data/Flickr25K/tubui/cvpr21wmf/image_comparator_phase1.pt).


## Inference
To extract the fingerprint of an image:


```python
python inference.py -i example.png -w ./weight/image_comparator_phase1.pt

```

This produces a 256-D float32 descriptor.


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