# semiDepth
Under construction



Tensorflow implementation of Semi-Supervised Monocular Depth Estimation with Left-Right Consistency Using Deep Neural Network

[comment]:<p align="center">
[comment]:  <img src="http://visual.cs.ucl.ac.uk/pubs/monoDepth/monodepth_teaser.gif" alt="monodepth">
[comment]:</p>

**Semi-Supervised Monocular Depth Estimation with Left-Right Consistency Using Deep Neural Network**  


## Requirements
This code was tested with Tensorflow 1.12, CUDA 9.0 and Ubuntu 16.04 and Gentoo.  
Training takes about 15 hours with the default parameters on the **kitti** split on a single Nvidia 1080 Ti machine.

[comment]:## I just want to try it on an image!
[comment]:There is a simple mode `monodepth_simple.py` which allows you to quickly run our model on a test image.  
[comment]:Make sure your first [download one of the pretrained models](#models) in this example we will use [comment]:`model_cityscapes`.
[comment]:```shell
[comment]:python monodepth_simple.py --image_path ~/my_image.jpg --checkpoint_path ~/models/model_cityscapes
[comment]:```
[comment]:**Please note that there is NO extension after the checkpoint name**  

## Data
This model requires rectified stereo pairs for training.  
There are two main datasets available: 
### [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php)
We used two different split of the data, **kitti** and **eigen**, amounting for respectively 29000 and 22600 training samples, you can find them in the [filenames](utils/filenames) folder.  
You can download the entire raw dataset by running:
```shell
wget -i utils/kitti_archives_to_download.txt -P ~/my/output/folder/
```
**Warning:** it weights about **175GB**, make sure you have enough space to unzip too!  

[comment]:### [Cityscapes](https://www.cityscapes-dataset.com)
[comment]:You will need to register in order to download the data, which already has a train/val/test set with 22973 training images.  
[comment]:We used `leftImg8bit_trainvaltest.zip`, `rightImg8bit_trainvaltest.zip`, `leftImg8bit_trainextra.zip` and `rightImg8bit_trainextra.zip` which weights **110GB**.

## Training

**Warning:** The input sizes need to be mutiples of 128 for `vgg` or 64 for `resnet50` . 

The model's dataloader expects a data folder path as well as a list of filenames (relative to the root data folder):  
```shell
python monodepth_main.py --mode train --model_name my_model --data_path ~/data/KITTI/ \
--filenames_file ~/code/monodepth/utils/filenames/kitti_train_files.txt --log_directory ~/tmp/
```
You can continue training by loading the last saved checkpoint using `--checkpoint_path` and pointing to it:  
```shell
python monodepth_main.py --mode train --model_name my_model --data_path ~/data/KITTI/ \
--filenames_file ~/code/monodepth/utils/filenames/kitti_train_files.txt --log_directory ~/tmp/ \
--checkpoint_path ~/tmp/my_model/model-50000
```
You can also fine-tune from a checkpoint using `--retrain`.  
You can monitor the learning process using `tensorboard` and pointing it to your chosen `log_directory`.  
By default the model only saves a reduced summary to save disk space, you can disable this using `--full_summary`.  
Please look at the [main file](monodepth_main.py) for all the available options.

## Testing  
To test change the `--mode` flag to `test`, the network will output the disparities in the model folder or in any other folder you specify wiht `--output_directory`.  
You will also need to load the checkpoint you want to test on, this can be done with `--checkpoint_path`:  
```shell
python monodepth_main.py --mode test --data_path ~/data/KITTI/ \
--filenames_file ~/code/monodepth/utils/filenames/kitti_stereo_2015_test_files.txt --log_directory ~/tmp/ \
--checkpoint_path ~/tmp/my_model/model-181250
```
**Please note that there is NO extension after the checkpoint name**  
If your test filenames contain two files per line the model will ignore the second one, unless you use the `--do_stereo` flag.
The network will output two files `disparities.npy` and `disparities_pp.npy`, respecively for raw and post-processed disparities.

## Evaluation on KITTI
To evaluate run:  
```shell
python utils/evaluate_kitti.py --split kitti --predicted_disp_path ~/tmp/my_model/disparities.npy \
--gt_path ~/data/KITTI/
```
The `--split` flag allows you to choose which dataset you want to test on.  
* `kitti` corresponds to the 200 official training set pairs from [KITTI stereo 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo).  
* `eigen` corresponds to the 697 test images used by [Eigen NIPS14](http://www.cs.nyu.edu/~deigen/depth/) and uses the raw LIDAR points.

**Warning**: The results on the Eigen split are usually cropped, which you can do by passing the `--garg_crop` flag.

## Models
You can download our pre-trained models to an existing directory by running:  

## Results


## Reference

## Video
[![Screenshot](https://img.youtube.com/vi/go3H2gU-Zck/0.jpg)](https://www.youtube.com/watch?v=go3H2gU-Zck)

## License
