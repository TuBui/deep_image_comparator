#!/bin/bash

echo "Downloading imagenet-C pretrained model"
mkdir ./pretrained
wget -P ./pretrained https://cvssp.org/data/Flickr25K/tubui/cvpr21wmf/deepaugment_and_augmix.pth

echo "downloading PSBattles dataset ..."
mkdir ./data
wget -P ./data https://cvssp.org/data/Flickr25K/tubui/cvpr21wmf/psbattles.tar 

tar -xvf ./data/psbattles.tar -C ./data