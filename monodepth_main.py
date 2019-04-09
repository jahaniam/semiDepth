# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim

from monodepth_model import *
from monodepth_dataloader import *
from average_gradients import *
import cv2
import png
from datetime import datetime

parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')

parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--model_name',                type=str,   help='model name', default='monodepth')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, vgg or resnet50', default='resnet50-forward')
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti, or cityscapes or make3D', default='kitti')
parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
parser.add_argument('--input_height',              type=int,   help='input height', default=256)
parser.add_argument('--input_width',               type=int,   help='input width', default=512)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=8)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=25)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--lr_loss_weight',            type=float, help='left-right consistency weight', default=1.0)
parser.add_argument('--alpha_image_loss',          type=float, help='weight between SSIM and L1 in the image loss', default=0.85)
parser.add_argument('--disp_gradient_loss_weight', type=float, help='disparity smoothness weigth', default=0.1)
parser.add_argument('--do_stereo',                             help='if set, will train the stereo model', action='store_true')
parser.add_argument('--wrap_mode',                 type=str,   help='bilinear sampler wrap mode, edge or border', default='border')
parser.add_argument('--use_deconv',                            help='if set, will use transposed convolutions', action='store_true')
parser.add_argument('--num_gpus',                  type=int,   help='number of GPUs to use for training', default=1)
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=8)
parser.add_argument('--output_directory',          type=str,   help='output directory for test disparities, if empty outputs to checkpoint folder', default='')
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default='')
parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--full_summary',                          help='if set, will keep more data for each summary. Warning: the file can become very large', action='store_true')
parser.add_argument('--lidar_weight', type=float, help='weight of the Lidar loss', default=15.0)
parser.add_argument('--save_visualized', help='save visualized results- automatically finds the min and max value! it is not good for comparison! for comparison use the visualize in evaluation python file',  action='store_true')
parser.add_argument('--save_official', help='save visualized results- for benchmark submission',  action='store_true')

parser.add_argument('--do_gradient_fix', help='apply hotfix for gradient bug',  action='store_true', default='True')
args = parser.parse_args()



def visualize_colormap(mat, colormap=cv2.COLORMAP_JET):
    min_val = np.amin(mat)
    max_val = np.amax(mat)
    min_val= 1/80.
    max_val= 1./5.
    mat[mat<min_val]=min_val
    mat[mat>max_val]=max_val
    mat_view = (mat - min_val) / (max_val - min_val)
    mat_view *= 255
    mat_view = mat_view.astype(np.uint8)
    mat_view = cv2.applyColorMap(mat_view, colormap)

    return mat_view


def save_visualized_results(disparities_pp,img, width, height,step):

    img_dir_vis=args.checkpoint_path + '/output_vis'
    if not os.path.exists(img_dir_vis):
        os.makedirs(img_dir_vis)
    print('saving ',img_dir_vis+'/'+str(step).zfill(10)+'.png')
    cv2.imwrite(img_dir_vis+'/'+str(step).zfill(10)+'.png',img)

    resized_disparity = cv2.resize(disparities_pp, (width, height), interpolation=cv2.INTER_LINEAR)
    im_view = visualize_colormap(resized_disparity)
    cv2.imwrite(img_dir_vis+'/'+str(step).zfill(10)+'_disp.png',im_view)
    return img_dir_vis
    # cv2.imshow('I', im_view) # * width)
    # cv2.waitKey(1)

def save_official(invDepth,width,height,img_dir,img_name):
    with open(img_dir+'/' + img_name, 'wb') as f:
        pred_depths = (1.0 / invDepth).astype(np.float32)

        pred_depths = cv2.resize(pred_depths, (width, height), interpolation=cv2.INTER_LINEAR)
        pred_depths[np.isinf(pred_depths)] = 80.
        pred_depths[pred_depths > 80.0] = 80.
        pred_depths[pred_depths < 0.5] = 0.5

        pred_depths *= 256

        # pypng is used because cv2 cannot save uint16 format images
        writer = png.Writer(width=width,
                            height=height,
                            bitdepth=16,
                            greyscale=True)
        writer.write(f, pred_depths.astype(np.uint16))


def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def count_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)

def train(params):
    """Training loop."""

    with tf.Graph().as_default(), tf.device('/cpu:0'):

        global_step = tf.Variable(0, trainable=False)

        # OPTIMIZER
        num_training_samples = count_text_lines(args.filenames_file)

        steps_per_epoch = np.ceil(num_training_samples / params.batch_size).astype(np.int32)
        num_total_steps = params.num_epochs * steps_per_epoch
        start_learning_rate = args.learning_rate

        boundaries = [np.int32((3/5) * num_total_steps), np.int32((4/5) * num_total_steps)]
        values = [args.learning_rate, args.learning_rate / 2, args.learning_rate / 4]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

        opt_step = tf.train.AdamOptimizer(learning_rate)

        print("total number of samples: {}".format(num_training_samples))
        print("total number of steps: {}".format(num_total_steps))

        dataloader = MonodepthDataloader(args.data_path, args.filenames_file, params, args.dataset, args.mode)
        left  = dataloader.left_image_batch
        right = dataloader.right_image_batch
        left_depth= dataloader.left_depth_batch
        right_depth= dataloader.right_depth_batch
        focal_length= dataloader.focal_length_batch

        # split for each gpu
        left_splits  = tf.split(left,  args.num_gpus, 0)
        right_splits = tf.split(right, args.num_gpus, 0)
        left_depth_splits = tf.split(left_depth, args.num_gpus, 0)
        right_depth_splits = tf.split(right_depth, args.num_gpus, 0)
        focal_length_splits = tf.split(focal_length, args.num_gpus, 0)

        tower_grads  = []
        tower_losses = []
        reuse_variables = None
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(args.num_gpus):
                with tf.device('/gpu:%d' % i):

                    model = MonodepthModel(params, args.mode, left_splits[i], right_splits[i], left_depth_splits[i], right_depth_splits[i], focal_length_splits[i], reuse_variables, i)

                    loss = model.total_loss
                    tower_losses.append(loss)

                    reuse_variables = True

                    grads = opt_step.compute_gradients(loss)

                    tower_grads.append(grads)

        grads = average_gradients(tower_grads)

        apply_gradient_op = opt_step.apply_gradients(grads, global_step=global_step)

        total_loss = tf.reduce_mean(tower_losses)

        tf.summary.scalar('learning_rate', learning_rate, ['model_0'])
        tf.summary.scalar('total_loss', total_loss, ['model_0'])
        summary_op = tf.summary.merge_all('model_0')

        # SESSION
        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)

        # SAVER
        summary_writer = tf.summary.FileWriter(args.log_directory + '/' + args.model_name, sess.graph)
        train_saver = tf.train.Saver()

        # COUNT PARAMS
        total_num_parameters = 0
        for variable in tf.trainable_variables():
            total_num_parameters += np.array(variable.get_shape().as_list()).prod()
        print("number of trainable parameters: {}".format(total_num_parameters))

        # INIT
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        # LOAD CHECKPOINT IF SET
        if args.checkpoint_path != '':
            train_saver.restore(sess, args.checkpoint_path.split(".")[0])

            if args.retrain:
                sess.run(global_step.assign(0))

        # GO!
        start_step = global_step.eval(session=sess)
        start_time = time.time()
        for step in range(start_step, num_total_steps):
            before_op_time = time.time()
            _, loss_value = sess.run([apply_gradient_op, total_loss])
            duration = time.time() - before_op_time
            if step and step % 100 == 0:
                examples_per_sec = params.batch_size / duration
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / step - 1.0) * time_sofar
                print_string = 'batch {:>6} | examples/s: {:4.2f} | loss: {:.5f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(step, examples_per_sec, loss_value, time_sofar, training_time_left))
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, global_step=step)
            if step and step % 10000 == 0:
                train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=step)

        train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=num_total_steps)

def test(params):
    """Test function."""

    dataloader = MonodepthDataloader(args.data_path, args.filenames_file, params, args.dataset, args.mode)
    left  = dataloader.left_image_batch
    right = dataloader.right_image_batch

    model = MonodepthModel(params, args.mode, left, right, None, None, None)

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    if args.checkpoint_path == '':
        restore_path = tf.train.latest_checkpoint(args.log_directory + '/' + args.model_name)
    else:
        restore_path = args.checkpoint_path.split(".")[0]
    train_saver.restore(sess, restore_path)

    num_test_samples = count_text_lines(args.filenames_file)

    print('now testing {} files'.format(num_test_samples))
    disparities    = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    disparities_pp = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)

    img_paths = open(args.filenames_file, "r").read().split('\n')
    ####img
    if args.save_official:
        img_dir=args.checkpoint_path + '/output'+ datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
    
    for step in range(num_test_samples):
        disp = sess.run(model.invDepth_left_est[0])
        disparities[step] = disp[0].squeeze()
        disparities_pp[step] = post_process_disparity(disp.squeeze())

        if args.save_visualized:
        #getting shape of the image
            img_path = os.path.join(args.data_path, img_paths[step].split(' ')[0])
            img = cv2.imread(img_path)
            img_name = img_path.split('/')[-1]
            # Change to png
            img_name = img_name[:-3] + 'png'
            height, width, channel = img.shape

            if args.dataset=='make3D':
                half_crop_height = width//5
                img  =  img[height//2 - half_crop_height :height//2 + half_crop_height,:,:]
            
            img_dir_vis=save_visualized_results(disparities_pp[step],img, img.shape[1], img.shape[0],step)
            if args.save_official:
                save_official(disparities_pp[step],width,height,img_dir,img_name)


    print('done.')
    #os.system("ffmpeg -f image2 -r 20 -i "+img_dir_vis+"/%10d_disp.png -vcodec libx264 -crf 22 "+img_dir_vis+"/video.mp4")

    print('writing inverse depths.')
    if args.output_directory == '':
        output_directory = os.path.dirname(args.checkpoint_path)
    else:
        output_directory = args.output_directory
    np.save(output_directory + '/invDepth.npy',    disparities)

    print('done.')


def main(_):

    params = monodepth_parameters(
        encoder=args.encoder,
        height=args.input_height,
        width=args.input_width,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        num_epochs=args.num_epochs,
        do_stereo=args.do_stereo,
        wrap_mode=args.wrap_mode,
        use_deconv=args.use_deconv,
        alpha_image_loss=args.alpha_image_loss,
        disp_gradient_loss_weight=args.disp_gradient_loss_weight,
        lr_loss_weight=args.lr_loss_weight,
        full_summary=args.full_summary,
        lidar_weight=args.lidar_weight,
        do_gradient_fix=args.do_gradient_fix)

    if args.mode == 'train':
        train(params)
    elif args.mode == 'test':
        test(params)

if __name__ == '__main__':
    tf.app.run()
