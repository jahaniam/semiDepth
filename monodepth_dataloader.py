# Copyright UCL Business plc 2017. Patent Pending. All rights reserved. 
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence, 
# please contact info@uclb.com

"""Monodepth data loader.
"""

from __future__ import absolute_import, division, print_function
import tensorflow as tf

def string_length_tf(t):
  return tf.py_func(len, [t], [tf.int64])

class MonodepthDataloader(object):
    """monodepth dataloader"""

    def __init__(self, data_path, filenames_file, params, dataset, mode):
        self.data_path = data_path
        self.params = params
        self.dataset = dataset
        self.mode = mode

        self.left_image_batch  = None
        self.right_image_batch = None
        focal_length = None


        input_queue = tf.train.string_input_producer([filenames_file], shuffle=False)
        line_reader = tf.TextLineReader()
        _, line = line_reader.read(input_queue)

        split_line = tf.string_split([line]).values

        # we load only one image for test, except if we trained a stereo model
        if mode == 'test' and not self.params.do_stereo:
            left_image_path  = tf.string_join([self.data_path, split_line[0]])
            left_image_o,junk  = self.read_image(left_image_path)
        else:
            left_image_path  = tf.string_join([self.data_path, split_line[0]])
            right_image_path = tf.string_join([self.data_path, split_line[1]])
            left_depth_path  = tf.string_join([self.data_path, split_line[2]])
            right_depth_path = tf.string_join([self.data_path, split_line[3]])


            left_image_o, focal_length  = self.read_image(left_image_path)
            right_image_o,junk = self.read_image(right_image_path)

            left_depth_o  = self.read_depth(left_depth_path)
            right_depth_o = self.read_depth(right_depth_path)


        if mode == 'train':
            # randomly flip images
            do_flip = tf.random_uniform([], 0, 1)
            left_image  = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(right_image_o), lambda: left_image_o)
            right_image = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(left_image_o),  lambda: right_image_o)
            left_depth  = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(right_depth_o), lambda: left_depth_o)
            right_depth = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(left_depth_o),  lambda: right_depth_o)

            # randomly augment images
            do_augment  = tf.random_uniform([], 0, 1)
            left_image, right_image = tf.cond(do_augment > 0.5, lambda: self.augment_image_pair(left_image, right_image), lambda: (left_image, right_image))

            left_image.set_shape( [None, None, 3])
            right_image.set_shape([None, None, 3])
            left_depth.set_shape([self.params.height, self.params.width, 1])
            right_depth.set_shape([self.params.height, self.params.width, 1])

            # capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
            min_after_dequeue = 2048
            capacity = min_after_dequeue + 4 * params.batch_size
            self.left_image_batch, self.right_image_batch,self.left_depth_batch,self.right_depth_batch,self.focal_length_batch = tf.train.shuffle_batch([left_image, right_image,left_depth, right_depth, focal_length],
                        params.batch_size, capacity, min_after_dequeue, params.num_threads)


        elif mode == 'test':
            self.left_image_batch = tf.stack([left_image_o,  tf.image.flip_left_right(left_image_o)],  0)
            self.left_image_batch.set_shape( [2, None, None, 3])

            if self.params.do_stereo:
                self.right_image_batch = tf.stack([right_image_o,  tf.image.flip_left_right(right_image_o)],  0)
                self.right_image_batch.set_shape( [2, None, None, 3])

    def augment_image_pair(self, left_image, right_image):
        # randomly shift gamma
        random_gamma = tf.random_uniform([], 0.8, 1.2)
        left_image_aug  = left_image  ** random_gamma
        right_image_aug = right_image ** random_gamma

        # randomly shift brightness
        random_brightness = tf.random_uniform([], 0.5, 2.0)
        left_image_aug  =  left_image_aug * random_brightness
        right_image_aug = right_image_aug * random_brightness

        # randomly shift color
        random_colors = tf.random_uniform([3], 0.8, 1.2)
        white = tf.ones([tf.shape(left_image)[0], tf.shape(left_image)[1]])
        color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
        left_image_aug  *= color_image
        right_image_aug *= color_image

        # saturate
        left_image_aug  = tf.clip_by_value(left_image_aug,  0, 1)
        right_image_aug = tf.clip_by_value(right_image_aug, 0, 1)

        return left_image_aug, right_image_aug

    def read_image(self, image_path):
        # tf.decode_image does not return the image size, this is an ugly workaround to handle both jpeg and png
        path_length = string_length_tf(image_path)[0]
        file_extension = tf.substr(image_path, path_length - 3, 3)
        file_cond = tf.equal(file_extension, 'jpg')
        
        image  = tf.cond(file_cond, lambda: tf.image.decode_jpeg(tf.read_file(image_path)), lambda: tf.image.decode_png(tf.read_file(image_path)))
        w = tf.shape(image)[1]
        # if the dataset is cityscapes, we crop the last fifth to remove the car hood
        if self.dataset == 'cityscapes':
            o_height    = tf.shape(image)[0]
            crop_height = (o_height * 4) // 5
            image  =  image[:crop_height,:,:]

        

        elif self.dataset == 'make3D':
            o_height    = tf.shape(image)[0]
            o_width   = tf.shape(image)[1]
            half_crop_height = o_width//5
            image  =  image[o_height//2 - half_crop_height :o_height//2 + half_crop_height,:,:]

            # image = tf.image.crop_to_bounding_box(image,o_height//2 - half_crop_height ,0,2*half_crop_height,o_width)



        ###################





        ################### getting focal length
        def f1():
            return tf.constant(721.5377 * self.params.width / 1242.)

        def f2():
            return tf.constant(718.856 * self.params.width / 1241.)

        def f3():
            return tf.constant(707.0493 * self.params.width / 1224.)

        def f4():
            return tf.constant(707.0493 * self.params.width / 1226.)

        def f5():
            return tf.constant(718.3351 * self.params.width / 1238.)    

        focal_length = tf.case({tf.equal(w, 1242): f1, tf.equal(w, 1241): f2, tf.equal(w, 1224): f3,
                                tf.equal(w, 1226): f4, tf.equal(w, 1238): f5}, default=f1, exclusive=True)

        #############

        image  = tf.image.convert_image_dtype(image,  tf.float32)
        image  = tf.image.resize_images(image,  [self.params.height, self.params.width], tf.image.ResizeMethod.AREA)




        return image,focal_length

    def read_depth(self, depth_path):
        depth = tf.image.decode_png(tf.read_file(depth_path), dtype=tf.uint16, channels=1)
        if self.dataset=='virtualKitti':
            depth = tf.cast(depth, tf.float32) / 100.
        else:
            depth = tf.cast(depth, tf.float32) / 256.
        depth = tf.image.resize_images(depth, [self.params.height, self.params.width], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        if_zero = tf.fill([self.params.height, self.params.width, 1], -1.0)
        depth = tf.where(tf.equal(depth, 0.0), if_zero, 1. / depth)

        return depth    
