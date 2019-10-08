# semiDepth

Tensorflow implementation of [Semi-Supervised Monocular Depth Estimation with Left-Right Consistency Using Deep Neural Network](https://arxiv.org/pdf/1905.07542.pdf).

#### disclaimer:
 Most of this code is based on [monodepth](https://github.com/mrharicot/monodepth). We extended their work and added the lidar supervision into the training process. The authors take no credit from Monodepth, therefore namings conventions of the files are same and licenses should remain intact. Please cite their work if you find them helpful.

<p align="center">
<img src="https://github.com/a-jahani/semiDepth/blob/master/demo.gif" alt="semiDepth">
</p>
Link to full video: https://www.youtube.com/watch?v=7ldCPJ60abw

**Semi-Supervised Monocular Depth Estimation with Left-Right Consistency Using Deep Neural Network**  


## Requirements
This code was tested with Tensorflow 1.12, CUDA 9.0 and Ubuntu 16.04 and Gentoo.  
Please download kitti depth annotaion dataset and place it with the correct folder structure.

## Data
This model requires rectified stereo pairs for training and registered annotated depth map.  
There are two main datasets available: 
### [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php) and [Cityscapes](https://www.cityscapes-dataset.com) 
please follow [monodepth](https://github.com/mrharicot/monodepth) download instruction (we do not convert into jpg). We only used stereo images for training the cityscape model.

### [KITTI Depth Annotated](http://www.cvlibs.net/download.php?file=data_depth_annotated.zip)
You can download depth annotated data from this [link](http://www.cvlibs.net/download.php?file=data_depth_annotated.zip).
sudo pip3 install rospkg catkin_pkg
Please go to `utils/filenames/eigen_train_files_withGT_annotated.txt` and make sure your folder names and structure of your folders matches the file.

`eigen_train_files_withGT_annotated.txt` is structured as bellow:
`left_image right_image left_annotated_depth right_annotated_depth`

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
To apply hotfix for gradient smoothness loss bug add `--do_gradient_fix` (we used this flag for all of our experiments)
  
Please look at the [monodepth_main](monodepth_main.py) and [original monodepth github](https://github.com/mrharicot/monodepth) for all the available options.

## Testing  
To test change the `--mode` flag to `test` and provide the path of the checkpoint of your model by `--checkpoint_path`. You can visualized the result using `--save_visualized`. We save post processed output for visualization. It should create a folder next to the model checkpoint folder containing results:  
```shell
python monodepth_main.py --mode test --data_path ~/data/KITTI/ \
--filenames_file utils/filenames/eigen_test_files.txt --log_directory tmp/ \
--checkpoint_path tmp/my_model/model-181250 --save_visualized
```

## Testing on single image
To test the network on one image you can use the `monodepth_simple.py` it should svae the output file with the same input file name+'_disp' in the same directory
```shell
python monodepth_simple.py --image /path-to-image --checkpoint_path /path-to-model
```
**Please note that there is NO extension after the checkpoint name**  

This will create a file named invDepth.npy containing result. 

## Evaluation on KITTI
To evaluate eigen, we used 652 annotated images:  
```shell
python2 utils/evaluate_kitti_depth.py --split eigen --predicted_disp_path \
models/eigen_finedTuned_cityscape_resnet50Forward/invDepth.npy  \
--gt_path /home/datasets/ --garg_crop --invdepth_provided --test_file \
utils/filenames/eigen_test_files_withGT.txt \
--shared_index utils/filenames/eigen692_652_shared_index.txt
```

By running the code for `eigen_finedTuned_cityscape_resnet50Forward` and `invDepth.npy` you should get results below:
```  
abs_rel,     sq_rel,        rms,    log_rms,     d1_all,         a1,         a2,         a3
0.0784,     0.4174,      3.464,      0.126,      0.000,      0.923,      0.984,      0.995
```
## Models
You can download our pre-trained model from links below:

 [eigen_finedTuned_cityscape_resnet50Forward](https://drive.google.com/drive/folders/1U7KmrbXjTfFvuffwxPZ2XqN5zLKNHlpf?usp=sharing)
 
  [cityscape_resnet50Forward](https://drive.google.com/drive/folders/1U7KmrbXjTfFvuffwxPZ2XqN5zLKNHlpf?usp=sharing)
 
## Results
You can download the npy file containing result for the 697 eigen test files from [invDepth.npy](https://drive.google.com/file/d/1yvZsO-ZMmlz0LK6vLnH513FPM70OS1wQ/view?usp=sharing). We are filtering the 45 images which there is no annotation depth map for it in our evaluation python code using the file `eigen692_652_shared_index.txt`


## Video
[![Screenshot](https://img.youtube.com/vi/7ldCPJ60abw/0.jpg)](https://www.youtube.com/watch?v=7ldCPJ60abw)

## License
Please have a look at [original monodepth](https://github.com/mrharicot/monodepth)
