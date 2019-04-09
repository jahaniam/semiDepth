# semiDepth

Tensorflow implementation of Semi-Supervised Monocular Depth Estimation with Left-Right Consistency Using Deep Neural Network.

#### disclaimer:
 Most of this code is based on [monodepth](https://github.com/mrharicot/monodepth). We extended their work and added the lidar supervision into our training. The authors take no credit from Monodepth, therefore the licenses and namings conventions of the files should remain intact. Please cite their work if you find them helpful.

[comment]:<p align="center">
[comment]:  <img src="http://visual.cs.ucl.ac.uk/pubs/monoDepth/monodepth_teaser.gif" alt="monodepth">
[comment]:</p>

**Semi-Supervised Monocular Depth Estimation with Left-Right Consistency Using Deep Neural Network**  


## Requirements
This code was tested with Tensorflow 1.12, CUDA 9.0 and Ubuntu 16.04 and Gentoo.  
Training takes about 15 hours with the default parameters on the **kitti** split on a single Nvidia 1080 Ti machine.


## Data
This model requires rectified stereo pairs for training.  
There are two main datasets available: 
### [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php) and [Cityscapes](https://www.cityscapes-dataset.com) 
please follow [monodepth](https://github.com/mrharicot/monodepth) download instruction (do not convert into jpg).

### [KITTI Depth Annotated](http://www.cvlibs.net/download.php?file=data_depth_annotated.zip)
You can download depth annotated data from this [website](http://www.cvlibs.net/download.php?file=data_depth_annotated.zip)
Please go to "utils/filenames/eigen_train_files_withGT_annotated.txt" and make sure your folder names and structure of your folders are correct.

"eigen_train_files_withGT_annotated.txt" is structured as bellow:
"left_image right_image left_annotated_depth right_annotated_depth"

## Training

Eigen split:  
```shell
python monodepth_main.py --mode train --model_name my_model --data_path ~/data/KITTI/ \
--filenames_file utils/filenames/eigen_train_files_withGT_annotated.txt --log_directory tmp/
```
You can continue training by loading the last saved checkpoint using `--checkpoint_path` and pointing to it:  
```shell
python monodepth_main.py --mode train --model_name my_model --data_path ~/data/KITTI/ \
--filenames_file utils/filenames/eigen_train_files_withGT_annotated.txt --log_directory ~/tmp/ \
--checkpoint_path tmp/my_model/model-5000
```
For fine-tune from a checkpoint you should use `--retrain`.  
For monitoring use `tensorboard` and point it to your `log_directory`.  
  
Please look at the [monodepth_main](monodepth_main.py) and [original monodepth github](https://github.com/mrharicot/monodepth) for all the available options.

## Testing  
To test change the `--mode` flag to `test` and provide the path of the checkpoint of your model by `--checkpoint_path`. You can visualized the result using `--save_visualized` .It should create a folder next to the model checkpoint folder containing results:  
```shell
python monodepth_main.py --mode test --data_path ~/data/KITTI/ \
--filenames_file utils/filenames/eigen_test_files.txt --log_directory tmp/ \
--checkpoint_path tmp/my_model/model-181250 --save_visualized
```
**Please note that there is NO extension after the checkpoint name**  

This will create a file named invDepth.npy containing result. 

## Evaluation on KITTI
To evaluate eigen, we used 652 annotated images:  
```shell
python utils/evaluate_kitti_depth.py --split eigen --predicted_disp_path tmp/my_model/invDepth.npy \
--gt_path ~/data/KITTI/ --garg_crop
```

## Models
You can download our best pre-trained model from links below:
 [eigen_finedTuned_cityscape_resnet50Forward](https://drive.google.com/drive/folders/1U7KmrbXjTfFvuffwxPZ2XqN5zLKNHlpf?usp=sharing)
  [cityscape_resnet50Forward](https://drive.google.com/drive/folders/1U7KmrbXjTfFvuffwxPZ2XqN5zLKNHlpf?usp=sharing)
 
## Results
You can download the npy file containing result for the 697 eigen test files from [here](https://drive.google.com/file/d/1yvZsO-ZMmlz0LK6vLnH513FPM70OS1wQ/view?usp=sharing). We are filtering the 45 images which there is no annotation depth map for it in our evaluation python code using the file "eigen692_652_shared_index.txt"

## Reference
Tobe at IROS2019
## Video
[![Screenshot](https://img.youtube.com/vi/7ldCPJ60abw/0.jpg)](https://www.youtube.com/watch?v=7ldCPJ60abw)

## License
