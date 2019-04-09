import numpy as np
import cv2
import argparse
from evaluation_utils import *
from scipy.interpolate import LinearNDInterpolator
parser = argparse.ArgumentParser(description='Evaluation on the KITTI dataset')
parser.add_argument('--split',               type=str,   help='data split, kitti , eigen or make3D',         required=True)
parser.add_argument('--predicted_disp_path', type=str,   help='path to estimated disparities',      required=True)
parser.add_argument('--gt_path',             type=str,   help='path to ground truth disparities',   required=True)
parser.add_argument('--min_depth',           type=float, help='minimum depth for evaluation',        default=1)
parser.add_argument('--max_depth',           type=float, help='maximum depth for evaluation',        default=80)
parser.add_argument('--eigen_crop',                      help='if set, crops according to Eigen NIPS14',   action='store_true')
parser.add_argument('--garg_crop',                       help='if set, crops according to Garg  ECCV16',   action='store_true')
parser.add_argument('--save_visualized',                       help='if set, crops according to Garg  ECCV16',   action='store_true')

parser.add_argument('--depth_provided',                       help='if set, depth is provided instead of disparity',   action='store_true')
parser.add_argument('--use_official',                       help='if set, gt is official depth',   action='store_true')

args = parser.parse_args()




def visualize_colormap(mat, colormap=cv2.COLORMAP_JET,print_if=False):
    # mat=1.0/mat
    mat=np.reciprocal(mat)
    # mat[mat == inf] = 0
    #mat[np.nonzero(mat)]=1.0/mat    
    min_val = 0.01#np.amin(mat)
    max_val = 0.2#np.amax(mat)
 
    min_val = np.amin(mat)
    max_val = np.amax(mat)
    if print_if:
        print('min',np.amin(mat),'max',np.amax(mat))
    # print('min',min_val,'max',max_val)
    mat_view = (mat - min_val) / (max_val - min_val)
    mat_view *= 255
    mat_view = mat_view.astype(np.uint8)
    mat_view = cv2.applyColorMap(mat_view, colormap)

    return mat_view





if __name__ == '__main__':

    pred_disparities = np.load(args.predicted_disp_path)
    

    num_samples = pred_disparities.shape[0]
    print("num_samples=",num_samples)
    gt_depths = load_gt_depth_make3D(args.gt_path)
    # pred_disparities=np.ones(pred_disparities.shape)
    print('[pred 0]',pred_disparities[0])
    pred_depths=[]


    for t_id in range(num_samples):
        if args.depth_provided:
            depth_pred = 1./cv2.resize(pred_disparities[t_id], (gt_depths[t_id].shape[1], gt_depths[t_id].shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            disp_pred = cv2.resize(pred_disparities[t_id], (gt_depths[t_id].shape[1], gt_depths[t_id].shape[0]), interpolation=cv2.INTER_LINEAR)
            disp_pred = disp_pred * disp_pred.shape[1]

            # need to convert from disparity to depth
            # focal_length, baseline = 2262*disp_pred.shape[1]/2048 ,0.22
            focal_length, baseline = 741.*disp_pred.shape[1]/1242. ,0.54
            depth_pred = (baseline * focal_length) / disp_pred
        depth_pred[np.isinf(depth_pred)] = 0

        pred_depths.append(depth_pred)


    print (np.max(gt_depths[0]))
    print (np.max(pred_depths[0]))
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

        pred_depth[pred_depth < args.min_depth] = args.min_depth
        pred_depth[pred_depth > args.max_depth] = args.max_depth

        if args.save_visualized:

            im_view = visualize_colormap(pred_depth)
            
            cv2.imwrite('tmp/result/'+str(i).zfill(10)+'.png',im_view)
            im_view2 = visualize_colormap(gt_depth)
            cv2.imwrite('tmp/result/'+str(i).zfill(10)+'_GT.png',im_view2)


        if args.split == 'kitti' :
            gt_disp = gt_disparities[i]
            mask = gt_disp > 0
            pred_disp = pred_disparities_resized[i]

            disp_diff = np.abs(gt_disp[mask] - pred_disp[mask])
            bad_pixels = np.logical_and(disp_diff >= 3, (disp_diff / gt_disp[mask]) >= 0.05)
            d1_all[i] = 100.0 * bad_pixels.sum() / mask.sum()

        if args.split=='make3D':
            mask = (gt_depth < 70) & (gt_depth > 0)

        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[mask], pred_depth[mask])

    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3'))
    print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean(), a2.mean(), a3.mean()))
