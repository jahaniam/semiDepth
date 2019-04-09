import numpy as np
import cv2
import argparse
from evaluation_utils import *
import os


width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351
width_to_focal[1238] = 718.3351
width_to_focal[1226] = 707.0493

parser = argparse.ArgumentParser(description='Evaluation on the KITTI dataset')
parser.add_argument('--split',               type=str,   help='data split, kitti or eigen',         required=True)

parser.add_argument('--predicted_disp_path', type=str,   help='path to estimated disparities',      required=True)
parser.add_argument('--gt_path',             type=str,   help='path to ground truth disparities',   required=True)
parser.add_argument('--min_depth',           type=float, help='minimum depth for evaluation',        default=1.)
parser.add_argument('--max_depth',           type=float, help='maximum depth for evaluation',        default=80.)
parser.add_argument('--eigen_crop',                      help='if set, crops according to Eigen NIPS14',   action='store_true')
parser.add_argument('--garg_crop',                       help='if set, crops according to Garg  ECCV16',   action='store_true')
parser.add_argument('--save_visualized',                       help='if set it will save the visualized data',   action='store_true')


parser.add_argument('--depth_provided',                       help='if set, depth is provided instead of disparity',   action='store_true')
parser.add_argument('--invdepth_provided',                       help='if set, depth is provided instead of disparity',   action='store_true')


parser.add_argument('--test_file',       type=str,                help='test file contains gt',   required=True)
parser.add_argument('--shared_index',       type=str,                help='test file contains gt',  required=False)


args = parser.parse_args()



def visualize_colormap(mat, colormap=cv2.COLORMAP_JET,print_if=False):

    mat[mat > 80.] = 80.
    # mat=np.reciprocal(mat)
    #mat[np.nonzero(mat)]=1.0/mat    
    min_val = 1.0#np.amin(mat)
    max_val = 80.#np.amax(mat)

    if print_if:
        print('min',np.amin(mat),'max',np.amax(mat))
    # print('min',min_val,'max',max_val)
    mat_view = (mat - min_val) / (max_val - min_val)
    mat_view *= 255
    mat_view = mat_view.astype(np.uint8)
    mat_view = cv2.applyColorMap(mat_view, colormap)

    return mat_view

def filter_prediction_to_652(shared_index,pred_disparities):
    pred_disparities_filtered = np.zeros((652, pred_disparities.shape[1], pred_disparities.shape[2]), dtype=np.float32)

    for i in range(len(shared_index)):
        pred_disparities_filtered[i]=pred_disparities[np.int(shared_index[i])]

    return pred_disparities_filtered




if __name__ == '__main__':

    pred_disparities = np.load(args.predicted_disp_path)

    if args.split == 'kitti':
        num_samples = 200
        
        gt_disparities = load_gt_disp_kitti(args.gt_path)
        gt_depths, pred_depths, pred_disparities_resized = convert_disps_to_depths_kitti(gt_disparities, pred_disparities)

    elif args.split == 'eigen':
        if args.shared_index != None:
            idx_numbers = read_text_lines(os.path.join(args.shared_index))
            pred_disparities=filter_prediction_to_652(idx_numbers,pred_disparities)


        print('*****************************',pred_disparities.shape)
        test_files = read_text_lines(os.path.join(args.test_file))
        num_samples = len(test_files)
        gt_files,im_sizes=read_depth_data(test_files,args.gt_path)

        gt_depths = []
        pred_depths = []
        for t_id in range(num_samples):
            
            # camera_id = cams[t_id]  # 2 is left, 3 is right

            gt_depth=read_ground_truth_depth(gt_files[t_id])

            # depth = generate_depth_map(gt_calib[t_id], gt_files[t_id], im_sizes[t_id], camera_id, False, False)
            gt_depths.append(gt_depth.astype(np.float32))

            if args.invdepth_provided:
                pred_depth = 1./cv2.resize(pred_disparities[t_id], (im_sizes[t_id][1], im_sizes[t_id][0]), interpolation=cv2.INTER_LINEAR)
            elif args.depth_provided:
                pred_depth = cv2.resize(pred_disparities[t_id], (im_sizes[t_id][1], im_sizes[t_id][0]), interpolation=cv2.INTER_LINEAR)

            else:
                disp_pred = cv2.resize(pred_disparities[t_id], (im_sizes[t_id][1], im_sizes[t_id][0]), interpolation=cv2.INTER_LINEAR)
                disp_pred = disp_pred * disp_pred.shape[1]

                # need to convert from disparity to depth
                # focal_length, baseline = get_focal_length_baseline(gt_calib[t_id], camera_id)
                baseline=0.54
                focal_length=width_to_focal[im_sizes[t_id][1]]
                pred_depth = (baseline * focal_length) / disp_pred
            pred_depth[np.isinf(pred_depth)] = 80.


            pred_depths.append(pred_depth)

            #save visualization
            if args.save_visualized:
                im_view = visualize_colormap(pred_depth)
                cv2.imwrite('tmp/result/'+str(t_id).zfill(10)+'.png',im_view)
                im_view2 = visualize_colormap(gt_depth)
                cv2.imwrite('tmp/result/'+str(t_id).zfill(10)+'_GT.png',im_view2)

    rms     = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel  = np.zeros(num_samples, np.float32)
    d1_all  = np.zeros(num_samples, np.float32)
    a1      = np.zeros(num_samples, np.float32)
    a2      = np.zeros(num_samples, np.float32)
    a3      = np.zeros(num_samples, np.float32)
    
    for i in range(num_samples):
        
        gt_depth = gt_depths[i]
        pred_depth = pred_depths[i]


        if args.split == 'eigen':
            mask = np.logical_and(gt_depth > args.min_depth, gt_depth < args.max_depth)

            
            if args.garg_crop or args.eigen_crop:
                gt_height, gt_width = gt_depth.shape

                # crop used by Garg ECCV16
                # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
                if args.garg_crop:
                    crop = np.array([0.40810811 * gt_height,  0.99189189 * gt_height,   
                                     0.03594771 * gt_width,   0.96405229 * gt_width]).astype(np.int32)
                # crop we found by trial and error to reproduce Eigen NIPS14 results
                elif args.eigen_crop:
                    crop = np.array([0.3324324 * gt_height,  0.91351351 * gt_height,   
                                     0.0359477 * gt_width,   0.96405229 * gt_width]).astype(np.int32)

                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)

        if args.split == 'kitti':
            gt_disp = gt_disparities[i]
            mask = gt_disp > 0
            pred_disp = pred_disparities_resized[i]

            disp_diff = np.abs(gt_disp[mask] - pred_disp[mask])
            bad_pixels = np.logical_and(disp_diff >= 3, (disp_diff / gt_disp[mask]) >= 0.05)
            d1_all[i] = 100.0 * bad_pixels.sum() / mask.sum()

        pred_depth[pred_depth < args.min_depth] = args.min_depth
        pred_depth[pred_depth > args.max_depth] = args.max_depth
        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[mask], pred_depth[mask])

    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3'))
    print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean(), a2.mean(), a3.mean()))
