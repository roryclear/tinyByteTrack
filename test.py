from tinygrad.nn import Conv2d, BatchNorm2d
from tinygrad.tensor import Tensor
from tinygrad.device import is_dtype_supported
from tinygrad import dtypes
import numpy as np
from itertools import chain
from pathlib import Path
import cv2
from collections import defaultdict
import time, sys
from tinygrad.helpers import fetch
from tinygrad.nn.state import safe_load, load_state_dict
import json
from tinygrad import TinyJit
import lap
from collections import OrderedDict
import pickle


class KalmanFilter(object):
    def __init__(self):
        ndim, dt = 4, 1.  
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._motion_mat_tg = Tensor(self._motion_mat,dtype=dtypes.float32)
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._update_mat_tg = Tensor.eye(ndim, 2 * ndim)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance


    def initiate_batch(self, mean_pos_tg):
        means_tg = Tensor.cat(mean_pos_tg,Tensor.zeros_like(mean_pos_tg),dim=1)
        h_tg = mean_pos_tg[:, 3]       
        h_tg = h_tg.reshape(-1, 1)
        std_tg = 2 * self._std_weight_position * h_tg
        std_tg = std_tg.cat(2 * self._std_weight_position * h_tg, dim=1)
        std_tg = std_tg.cat(Tensor.full_like(h_tg, 1e-2), dim=1)
        std_tg = std_tg.cat(2 * self._std_weight_position * h_tg, dim=1)
        std_tg = std_tg.cat(10 * self._std_weight_velocity * h_tg, dim=1)
        std_tg = std_tg.cat(10 * self._std_weight_velocity * h_tg, dim=1)
        std_tg = std_tg.cat(Tensor.full_like(h_tg, 1e-5), dim=1)
        std_tg = std_tg.cat(10 * self._std_weight_velocity * h_tg, dim=1)
        covariances_tg = Tensor.einsum('ij,jk->ijk', std_tg**2, Tensor.eye(std_tg.shape[1]))
        return means_tg, covariances_tg

    def project(self, mean, covariance):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def project_batch(self, mean_tg, covariances_tg):
        
        sp = (mean_tg[:,3]*self._std_weight_position).cat(mean_tg[:,3]*self._std_weight_position)
        sp = sp.cat(1e-1 * Tensor.ones(mean_tg.shape[0]))
        sp = sp.cat(mean_tg[:,3]*self._std_weight_position)
        std_pos_tg = sp.reshape(4, mean_tg.shape[0]).T
        squared_stds = std_pos_tg.square()
        eye = Tensor.eye(4, dtype=dtypes.float32).reshape(1, 4, 4).expand(mean_tg.shape[0], 4, 4)
        innovation_covs_tg = eye * squared_stds.reshape(-1, 1, 4)
        projected_means_tg = mean_tg @ self._update_mat_tg.T
        projected_covariances_tg = Tensor.einsum('ij,njk,kl->nil', self._update_mat_tg, covariances_tg, self._update_mat_tg.T)
        projected_covariances_tg += innovation_covs_tg
        return projected_means_tg, projected_covariances_tg

    def multi_predict(self, mean_tg, covariance_tg):
        sp = (mean_tg[:,3]*self._std_weight_position).cat(mean_tg[:,3]*self._std_weight_position)
        sp = sp.cat(1e-2 * Tensor.ones(mean_tg.shape[0]))
        sp = sp.cat(mean_tg[:,3]*self._std_weight_position)
        std_pos_tg = sp.reshape(4,int(sp.shape[0]/4))

        sv = (mean_tg[:,3]*self._std_weight_velocity).cat(mean_tg[:,3]*self._std_weight_velocity)
        sv = sv.cat(1e-5 * Tensor.ones(mean_tg.shape[0]))
        sv = sv.cat(mean_tg[:,3]*self._std_weight_velocity)
        std_vel_tg = sv.reshape(4,int(sv.shape[0]/4))
        
        r = std_pos_tg.cat(std_vel_tg)
        sqr = Tensor.square(r).T
        batch_size = sqr.shape[0]
        dim = sqr.shape[1]
        motion_cov_tg = Tensor.eye(dim).reshape(1, dim, dim) * sqr.reshape(batch_size, dim, 1)
        mean_tg = Tensor.dot(mean_tg, self._motion_mat_tg.T)
        left_tg = Tensor.dot(self._motion_mat_tg,covariance_tg)
        covariance_tg = Tensor.dot(left_tg, self._motion_mat_tg.T) + motion_cov_tg
        return mean_tg, covariance_tg
        
    def cholesky(self,A):
        L = Tensor.zeros_like(A).contiguous()
        L[:, 0, 0] = Tensor.sqrt(A[:, 0, 0])
        L[:, 1, 0] = A[:, 1, 0] / L[:, 0, 0]
        L[:, 1, 1] = Tensor.sqrt(A[:, 1, 1] - L[:, 1, 0]**2)
        L[:, 2, 0] = A[:, 2, 0] / L[:, 0, 0]
        L[:, 2, 1] = (A[:, 2, 1] - L[:, 2, 0]*L[:, 1, 0]) / L[:, 1, 1]
        L[:, 2, 2] = Tensor.sqrt(A[:, 2, 2] - L[:, 2, 0]**2 - L[:, 2, 1]**2)
        L[:, 3, 0] = A[:, 3, 0] / L[:, 0, 0]
        L[:, 3, 1] = (A[:, 3, 1] - L[:, 3, 0]*L[:, 1, 0]) / L[:, 1, 1]
        L[:, 3, 2] = (A[:, 3, 2] - L[:, 3, 0]*L[:, 2, 0] - L[:, 3, 1]*L[:, 2, 1]) / L[:, 2, 2]
        L[:, 3, 3] = Tensor.sqrt(A[:, 3, 3] - L[:, 3, 0]**2 - L[:, 3, 1]**2 - L[:, 3, 2]**2)
        return L
    
    def solve_triangular(self, L, b):
        d = L.shape[0]
        if b.ndim == 1:
            b = b[:, None]
        diag_inv = 1.0 / np.diag(L)[:, None]
        L_offdiag = np.tril(L, k=-1)
        x = (np.eye(d) - np.tril(L_offdiag * diag_inv, k=-1)) @ (b * diag_inv)
        return x.squeeze()
    
    def solve_all_triangular(self,chol_factors_tg, R_tg):
        N, d, _ = chol_factors_tg.shape
        if R_tg.ndim == 2:
            R_tg = R_tg[..., None]
        eye = Tensor.eye(d)
        eye = eye.reshape(1, d, d)
        mask = eye._broadcast_to((N, d, d))
        diag_only = chol_factors_tg * mask
        diag_tg = diag_only.sum(axis=2)
        diag_tg = 1.0 / diag_tg
        diag_tg = diag_tg[..., None]
        L_offdiag_tg = Tensor.tril(chol_factors_tg, diagonal=-1)
        I_tg = Tensor.eye(d)[None, :, :]
        diag_inv_matrix_tg = Tensor._broadcast_to(diag_tg, (N, d, d))
        L_term_tg = Tensor.tril(L_offdiag_tg * diag_inv_matrix_tg, -1)
        T_tg = I_tg - L_term_tg
        RHS_tg = R_tg * diag_tg
        x = Tensor.matmul(T_tg, RHS_tg)
        return x

    def update_batch(self, means_tg, covariances_tg, measurements_tg):
        if means_tg.shape[0] == 0: return means_tg, covariances_tg
        projected_means_tg, projected_covs_tg = self.project_batch(means_tg, covariances_tg)
        chol_factors_tg = self.cholesky(projected_covs_tg)
        update_mat_T_tg = self._update_mat_tg.T
        R_tg = Tensor.einsum('ijk,kl->ilj', covariances_tg, update_mat_T_tg)
        y_tg = self.solve_all_triangular(chol_factors_tg, R_tg)
        chol_factors_T_tg = Tensor.permute(chol_factors_tg, (0, 2, 1))
        kalman_gain_tg = self.solve_all_triangular(chol_factors_T_tg, y_tg)
        innovation_tg = measurements_tg - projected_means_tg
        new_means_tg = means_tg + Tensor.einsum('ij,ijk->ik', innovation_tg, kalman_gain_tg)
        new_covariances_tg = covariances_tg - Tensor.einsum('nji,njk,nkl->nil', kalman_gain_tg, projected_covs_tg, kalman_gain_tg)
        return new_means_tg, new_covariances_tg

class TrackState(object):
    New = 1
    Tracked = 2
    Lost = 3
    Removed = 4
    Replaced = 5

def tlbr_np(values, mean):
    """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
    `(top left, bottom right)`."""
    if mean is None:
        ret = values[:4].copy()
    else:
        ret = mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
    ret[2:] += ret[:2]
    return ret

def tlbr_np_batch3(tracked_stracks_values):
    ret = tracked_stracks_values[:, :4]
    ret_0_1 = ret[:, :2]
    ret_2_3 = ret[:, 2:]
    ret_2_3_new = ret_2_3 + ret_0_1
    ret = ret_0_1.cat(ret_2_3_new, dim=1)
    return ret

def tlbr_np_batch2(means):
    if means.shape[0] == 0: return Tensor.empty((0, 4))
    ret = means[:, :4]
    ret2 = ret[:, 2]
    ret3 = ret[:, 3]
    ret_new2 = ret2 * ret3
    ret = ret[:, :2].cat(ret_new2.unsqueeze(1), dim=1).cat(ret3.unsqueeze(1),dim=1)
    ret[:, :2] -= ret[:, 2:] / 2
    ret[:, 2:] += ret[:, :2]
    return ret

def tlwh_to_xyah_batch(tlwh):
    tlwh = tlwh.contiguous()
    tlwh[:, :2] += tlwh[:, 2:] / 2
    tlwh[:, 2] /= tlwh[:, 3]
    return tlwh

def bbox_ious(boxes, query_boxes):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    
    boxes_area = (
        (boxes[:, 2] - boxes[:, 0] + 1) *
        (boxes[:, 3] - boxes[:, 1] + 1)
    ).reshape(N, 1)
    
    query_areas = (
        (query_boxes[:, 2] - query_boxes[:, 0] + 1) *
        (query_boxes[:, 3] - query_boxes[:, 1] + 1)
    ).reshape(1, K)
    
    ixmin = Tensor.maximum(boxes[:, 0].reshape(N, 1), query_boxes[:, 0].reshape(1, K))
    iymin = Tensor.maximum(boxes[:, 1].reshape(N, 1), query_boxes[:, 1].reshape(1, K))
    ixmax = Tensor.minimum(boxes[:, 2].reshape(N, 1), query_boxes[:, 2].reshape(1, K))
    iymax = Tensor.minimum(boxes[:, 3].reshape(N, 1), query_boxes[:, 3].reshape(1, K))
    
    iw = Tensor.maximum(ixmax - ixmin + 1, 0)
    ih = Tensor.maximum(iymax - iymin + 1, 0)
    intersection = iw * ih
    
    union = boxes_area + query_areas - intersection
    
    overlaps = Tensor.where(
        (iw > 0) & (ih > 0),
        intersection / union,
        Tensor.zeros_like(intersection)
    ) 
    return overlaps

class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        self._count = 0

        self.lost_stracks_ids_tg = Tensor.empty((0))
        self.lost_stracks_fids_tg = Tensor.empty((0))
        self.lost_stracks_startframes_tg = Tensor.empty()
        self.lost_stracks_states_tg = Tensor.empty()
        self.lost_stracks_bools_tg = Tensor.empty((0,),dtype=dtypes.bool)
        self.lost_stracks_values_tg = Tensor.empty((0,6))
        self.lost_stracks_means_tg = Tensor.empty((0,8),dtype=dtypes.float32)
        self.lost_stracks_covs_tg = Tensor.empty(dtype=dtypes.float32)

        self.tracked_stracks_values = []
        self.lost_stracks_values = []

        self.tracked_stracks_means_tg = Tensor.empty((0,8),dtype=dtypes.float32)
        self.tracked_stracks_bools_tg = Tensor.empty((0,))
        self.tracked_stracks_covs_tg = Tensor.empty((0,8,8),dtype=dtypes.float32)
        self.tracked_stracks_ids = []
        self.tracked_stracks_fids_tg = Tensor((0,),dtype=dtypes.int)
        self.tracked_stracks_startframes_tg = Tensor((0,),dtype=dtypes.int)
        self.tracked_stracks_states = []
        self.tracked_stracks_states_tg = Tensor.empty((0,),dtype=dtypes.int)
        self.tracked_stracks_values_tg = Tensor.empty((0,6),dtype=dtypes.float32)
        self.lost_stracks_means = []
        self.lost_stracks_bools = []
        self.lost_stracks_covs = []
        self.lost_stracks_ids = []
        self.lost_stracks_fids = []
        self.lost_stracks_startframes = []
        self.lost_stracks_states = []

        self.frame_id = 0
        self.args = args
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, output_results, img_info, img_size):
        self.frame_id += 1
        activated_stracks_values_tg = Tensor.empty((0,6),dtype=dtypes.float)
        activated_stracks_means_tg = Tensor.empty((0,6),dtype=dtypes.float)
        activated_stracks_bools_tg = Tensor.empty((0,),dtype=dtypes.bool)
        activated_stracks_covs_tg = Tensor.empty((0,8,8),dtype=dtypes.float)
        activated_stracks_ids_tg = Tensor.empty((0,),dtype=dtypes.int)
        activated_stracks_fids_tg = Tensor.empty((0,),dtype=dtypes.int)
        activated_stracks_states_tg = Tensor.empty((0,),dtype=dtypes.int)
        refind_stracks_means = []
        refind_stracks_bools_tg = Tensor.empty()
        refind_stracks_bools2_tg = Tensor.empty()
        refind_stracks_states_tg = Tensor.empty()
        refind_stracks_values_tg = Tensor.empty((0,6),dtype=dtypes.float32)
        refind_stracks_covs = []
        refind_stracks_ids = []
        removed_stracks_ids = []

        classes = output_results[:, 5]
        scores = output_results[:, 4]
        bboxes = output_results[:, :4]
        
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes = bboxes / scale

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh
        inds_second = inds_low & inds_high

        dets = bboxes * remain_inds.unsqueeze(1)
        dets_second = bboxes * inds_second.unsqueeze(1)
        dets[:, 2:] -= dets[:, :2] #tlbr to tlwh
        dets_second[:, 2:] -= dets_second[:, :2]
        dets_score_classes_tg = dets.cat(scores.reshape(-1,1), dim=1).cat(classes.reshape(-1,1), dim=1)
        dets_score_classes_second_tg = dets_second.cat(scores.reshape(-1,1), dim=1).cat(classes.reshape(-1,1), dim=1)    

        tracked_stracks_bools_tg = self.tracked_stracks_bools_tg 

        original_indices_tg = nonzero_indices_1d(self.tracked_stracks_bools_tg).cast(dtype=dtypes.int)

        tracked_stracks_bools_tg = tracked_stracks_bools_tg.cast(dtype=dtypes.bool)
        self.tracked_stracks_ids_tg = Tensor(self.tracked_stracks_ids, dtype=dtypes.int)
        tracked_stracks_ids_tg = Tensor(self.tracked_stracks_ids, dtype=dtypes.int)
        unconfirmed_ids_tg = tracked_stracks_ids_tg * ~tracked_stracks_bools_tg
        id_mask_tg = nonzero_indices_1d(tracked_stracks_bools_tg != True).cast(dtype=dtypes.int)
        unconfirmed_ids_tg = tracked_stracks_ids_tg[id_mask_tg]
        tracked_stracks_ids_tg = tracked_stracks_ids_tg * tracked_stracks_bools_tg
        
        unconfirmed_values_tg = self.tracked_stracks_values_tg[id_mask_tg]
        unconfirmed_covs_tg = self.tracked_stracks_covs_tg[id_mask_tg]
        unconfirmed_means_tg = self.tracked_stracks_means_tg[id_mask_tg]
        unconfirmed_startframes_tg = self.tracked_stracks_startframes_tg[id_mask_tg]

        tracked_stracks_states_tg = self.tracked_stracks_states_tg
        tracked_stracks_values_tg = self.tracked_stracks_values_tg

        tracked_stracks_fids_tg = self.tracked_stracks_fids_tg

        if len(self.lost_stracks_values_tg.shape) > 1 and self.lost_stracks_values_tg.shape[0] > 0:
            tracked_stracks_ids_tg = tracked_stracks_ids_tg.cat(self.lost_stracks_ids_tg)
            tracked_stracks_fids_tg = tracked_stracks_fids_tg.cat(self.lost_stracks_fids_tg)
            tracked_stracks_bools_tg = tracked_stracks_bools_tg.cat(self.lost_stracks_bools_tg)
            self.tracked_stracks_startframes_tg = self.tracked_stracks_startframes_tg.cat(self.lost_stracks_startframes_tg)
            tracked_stracks_states_tg = tracked_stracks_states_tg.cat(self.lost_stracks_states_tg)
            tracked_stracks_values_tg = tracked_stracks_values_tg.cat(self.lost_stracks_values_tg)

        
        id_mask_tg = nonzero_indices_1d(tracked_stracks_ids_tg != 0).cast(dtype=dtypes.int)
        tracked_stracks_ids_tg = tracked_stracks_ids_tg[id_mask_tg]

        tracked_stracks_fids_tg = tracked_stracks_fids_tg[id_mask_tg]

        tracked_stracks_bools_tg = tracked_stracks_bools_tg[id_mask_tg]
        self.tracked_stracks_startframes_tg = self.tracked_stracks_startframes_tg[id_mask_tg]
        tracked_stracks_states_tg = tracked_stracks_states_tg[id_mask_tg]
        tracked_stracks_values_tg = tracked_stracks_values_tg[id_mask_tg]

        tracked_stracks_states = tracked_stracks_states_tg.numpy().tolist()
        tracked_stracks_startframes = self.tracked_stracks_startframes_tg.numpy().tolist()
        tracked_stracks_ids = tracked_stracks_ids_tg.numpy().tolist()

        if self.tracked_stracks_means_tg.shape[0] > 0:
            self.tracked_stracks_means_tg, self.tracked_stracks_covs_tg = self.kalman_filter.multi_predict(self.tracked_stracks_means_tg, self.tracked_stracks_covs_tg)
        if self.lost_stracks_means_tg.shape[0] > 0:
            self.lost_stracks_means_tg, self.lost_stracks_covs_tg = self.kalman_filter.multi_predict(self.lost_stracks_means_tg, self.lost_stracks_covs_tg)

        tracked_stracks_means_tg = self.tracked_stracks_means_tg
        tracked_stracks_covs_tg = self.tracked_stracks_covs_tg

        means_in_tg = tracked_stracks_means_tg[original_indices_tg]
        covs_in_tg = tracked_stracks_covs_tg[original_indices_tg]
        
        if len(self.lost_stracks_means_tg.shape) > 0 and self.lost_stracks_means_tg.shape[0] > 0:
            means_in_tg = means_in_tg.cat(self.lost_stracks_means_tg)
            covs_in_tg = covs_in_tg.cat(self.lost_stracks_covs_tg)

        atlbrs_tg = tlbr_np_batch2(means_in_tg)
        btlbrs_tg = tlbr_np_batch3(dets_score_classes_tg)
        dists_tg = iou_distance(atlbrs_tg, btlbrs_tg)
        dists_tg = fuse_score(dists_tg, dets_score_classes_tg)
        dists = dists_tg.numpy()
        matches_tg, u_track_tg, u_detection_tg = linear_assignment(dists, thresh=self.args.match_thresh)


        det_values_arr_tg = dets_score_classes_tg[matches_tg[:,1]]
        activated_stracks_startframes_tg = Tensor.empty((0,))
        tracked_stracks_startframes_tg = self.tracked_stracks_startframes_tg
        activated_stracks_bools_tg = Tensor.empty((0,),dtype=dtypes.bool)
        refind_stracks_ids_tg = Tensor.empty((0,),dtype=dtypes.int)
        refind_stracks_means_tg = Tensor.empty((0,8),dtype=dtypes.float32)
        refind_stracks_covs_tg = Tensor.empty((0,8,8),dtype=dtypes.float32)
        if matches_tg.shape[0] > 0:
            tlwh_tg = det_values_arr_tg[:, :4]
            xyahs_tg = tlwh_to_xyah_batch(tlwh_tg)
            means_tg = means_in_tg[matches_tg[:,0]]
            covs_tg = covs_in_tg[matches_tg[:,0]]
            updated_means_tg, updated_covs_tg = self.kalman_filter.update_batch(means_tg, covs_tg, xyahs_tg)

            itracked_tg = matches_tg[:,0]
            itracked = itracked_tg.numpy()
            tracked_mask_tg = nonzero_indices_1d(itracked_tg < original_indices_tg.shape[0])
            lost_mask_tg = nonzero_indices_1d(itracked_tg >= original_indices_tg.shape[0])

            valid_tracked_indices_tg = original_indices_tg[itracked_tg[tracked_mask_tg]]
            self.tracked_stracks_means_tg[valid_tracked_indices_tg] = updated_means_tg[tracked_mask_tg]
            self.tracked_stracks_covs_tg[valid_tracked_indices_tg] = updated_covs_tg[tracked_mask_tg]
      
            tracked_stracks_fids_tg[itracked_tg] = self.frame_id
            self.tracked_stracks_fids_tg[valid_tracked_indices_tg] = self.frame_id

            if lost_mask_tg.shape[0] > 0:
                valid_lost_indices_tg = itracked_tg[lost_mask_tg] - original_indices_tg.shape[0]
                self.lost_stracks_means_tg[valid_lost_indices_tg] = updated_means_tg[lost_mask_tg]
                self.lost_stracks_covs_tg[valid_lost_indices_tg] = updated_covs_tg[lost_mask_tg]
            itracked_tracked_tg = nonzero_indices_1d(tracked_stracks_states_tg[itracked_tg] == TrackState.Tracked).cast(dtype=dtypes.int)

            activated_stracks_means_tg = tracked_stracks_means_tg[original_indices_tg][itracked_tg[itracked_tracked_tg]]
            activated_stracks_values_tg = tracked_stracks_values_tg[itracked_tg][itracked_tracked_tg]
            activated_stracks_states_tg = tracked_stracks_states_tg[itracked_tg][itracked_tracked_tg]

            itracked_untracked = np.array(tracked_stracks_states)[itracked] != TrackState.Tracked
            activated_stracks_bools_tg = tracked_stracks_bools_tg[itracked_tg][itracked_tracked_tg]
            activated_stracks_ids_tg = tracked_stracks_ids_tg[itracked_tg][itracked_tracked_tg]
            activated_stracks_fids_tg = tracked_stracks_fids_tg[itracked_tg][itracked_tracked_tg]
            activated_stracks_startframes_tg = tracked_stracks_startframes_tg[itracked_tg][itracked_tracked_tg]
            activated_stracks_covs_tg = self.tracked_stracks_covs_tg[original_indices_tg[itracked_tg[itracked_tracked_tg]]]
            

            refind_stracks_startframes_tg = tracked_stracks_startframes_tg
            itracked_untracked_tg = Tensor(itracked_untracked,dtype=dtypes.bool)

            refind_stracks_ids = np.array(tracked_stracks_ids)[itracked][itracked_untracked].tolist()
            refind_stracks_ids_tg = Tensor(refind_stracks_ids,dtype=dtypes.int)

            refind_stracks_fids_tg = tracked_stracks_fids_tg[itracked_tg]
            refind_stracks_values_tg = tracked_stracks_values_tg[itracked_tg]
            x = itracked_untracked.sum()
            refind_stracks_ids2_tg = tracked_stracks_ids_tg[itracked_tg] * itracked_untracked_tg
            refind_stracks_bools_tg = Tensor(True).repeat(int(x))
            refind_stracks_bools2_tg = Tensor(itracked_untracked)
            x = itracked.shape[0]
            refind_stracks_states_tg = Tensor(TrackState.Tracked).repeat(int(x))
            refind_stracks_startframes_tg = refind_stracks_startframes_tg[itracked_tg]

            if self.frame_id == 2:
                refind_stracks_means_tg = self.tracked_stracks_means_tg[original_indices_tg]
                refind_stracks_covs_tg = self.tracked_stracks_covs_tg[original_indices_tg]
            
            big_tg = itracked_tg[(nonzero_indices_1d(itracked_tg >= original_indices_tg.shape[0])).cast(dtype=dtypes.int)] - original_indices_tg.shape[0]

            refind_stracks_means_tg = refind_stracks_means_tg.cat(self.lost_stracks_means_tg[big_tg])
            refind_stracks_covs_tg = refind_stracks_covs_tg.cat(self.lost_stracks_covs_tg[big_tg])

            refind_stracks_means = refind_stracks_means_tg.numpy().tolist()
            refind_stracks_covs = refind_stracks_covs_tg.numpy().tolist()

        tracked_indices_tg = u_track_tg[nonzero_indices_1d(tracked_stracks_states_tg[u_track_tg] == TrackState.Tracked).cast(dtypes.int)]
        means_tg = tracked_stracks_means_tg[original_indices_tg[tracked_indices_tg]]
        atlbrs_tg = Tensor.empty((means_tg.shape[0]),dtype=dtypes.float32)
        if tracked_indices_tg.shape[0] > 0:
            atlbrs_tg = means_tg[:, :4].contiguous()
            atlbrs_tg[:, 2] *= atlbrs_tg[:, 3]
            atlbrs_tg[:, :2] -= atlbrs_tg[:, 2:] / 2
            atlbrs_tg[:, 2:] += atlbrs_tg[:, :2]
        btlbrs_tg = dets_score_classes_second_tg[:, :4].contiguous()
        btlbrs_tg[:, 2:] += btlbrs_tg[:, :2]
        dists_tg = iou_distance(atlbrs_tg, btlbrs_tg)
        dists = dists_tg.numpy()

        matches_tg, u_track2_tg, _ = linear_assignment(dists, thresh=0.5)
        matches = matches_tg.numpy()
        self.tracked_stracks_states_tg = self.tracked_stracks_states_tg.contiguous()
        self.tracked_stracks_states_tg[original_indices_tg[u_track_tg[u_track2_tg]]] = TrackState.Lost

        # Build inputs for batch update
        tlwh_tg = dets_score_classes_second_tg[matches_tg[:, 1]][:, :4]
        xyahs_tg = tlwh_to_xyah_batch(tlwh_tg)

        means_tg = tracked_stracks_means_tg[original_indices_tg[u_track_tg[matches_tg[:,0]]]]
        covs_tg = tracked_stracks_covs_tg[original_indices_tg[u_track_tg[matches_tg[:,0]]]]

        updated_means_tg, updated_covs_tg = self.kalman_filter.update_batch(means_tg, covs_tg, xyahs_tg)
        
        if len(matches) > 0:
            self.tracked_stracks_means_tg[original_indices_tg[u_track_tg[matches_tg[:,0]]]] = updated_means_tg
            self.tracked_stracks_covs_tg[original_indices_tg[u_track_tg[matches_tg[:,0]]]] = updated_covs_tg
            self.tracked_stracks_fids_tg[original_indices_tg[u_track_tg[matches_tg[:,0]]]] = self.frame_id
            self.tracked_stracks_states_tg[original_indices_tg[u_track_tg[matches_tg[:,0]]]] = TrackState.Tracked
            self.tracked_stracks_values_tg[original_indices_tg[u_track_tg[matches_tg[:, 0]]]] = dets_score_classes_second_tg[matches_tg[:, 1]]

            activated_stracks_means_tg = activated_stracks_means_tg.cat(self.tracked_stracks_means_tg[original_indices_tg[u_track_tg[matches_tg[:, 0]]]])
            
            activated_stracks_values_tg = activated_stracks_values_tg.cat(tracked_stracks_values_tg[original_indices_tg[u_track_tg[matches_tg[:, 0]]]])
            activated_stracks_states_tg = activated_stracks_states_tg.cat(self.tracked_stracks_states_tg[u_track_tg[matches_tg[:, 0]]])
            activated_stracks_ids_tg = activated_stracks_ids_tg.cat(tracked_stracks_ids_tg[u_track_tg[matches_tg[:, 0]]])

            activated_stracks_bools_tg = activated_stracks_bools_tg.cat(self.tracked_stracks_bools_tg[original_indices_tg[u_track_tg[matches_tg[:, 0]]]])
            activated_stracks_covs_tg = activated_stracks_covs_tg.cat(self.tracked_stracks_covs_tg[original_indices_tg[u_track_tg[matches_tg[:, 0]]]])
            activated_stracks_startframes_tg = activated_stracks_startframes_tg.cat(self.tracked_stracks_startframes_tg[u_track_tg[matches_tg[:, 0]]])
            activated_stracks_fids_tg = activated_stracks_fids_tg.cat(Tensor(self.frame_id).repeat(len(matches)))
        
        u_track3_tg = u_track_tg[u_track2_tg]

        lost_stracks_values_tg = self.tracked_stracks_values_tg[original_indices_tg][u_track3_tg]
        lost_stracks_means_tg = self.tracked_stracks_means_tg[original_indices_tg][u_track3_tg]
        lost_stracks_covs_tg = self.tracked_stracks_covs_tg[original_indices_tg][u_track3_tg]
        lost_stracks_bools_tg = self.tracked_stracks_bools_tg[original_indices_tg][u_track3_tg]
        lost_stracks_fids_tg = self.tracked_stracks_fids_tg[original_indices_tg][u_track3_tg]
        lost_stracks_ids_tg = self.tracked_stracks_ids_tg[original_indices_tg][u_track3_tg]
        lost_stracks_states_tg = self.tracked_stracks_states_tg[original_indices_tg][u_track3_tg]
        lost_stracks_startframes_tg = self.tracked_stracks_startframes_tg[original_indices_tg][u_track3_tg]
                
        dets_score_classes_second_tg = dets_score_classes_tg[u_detection_tg]
      

        atlbrs_tg = tlbr_np_batch2(unconfirmed_means_tg)
        btlbrs_tg = tlbr_np_batch3(dets_score_classes_second_tg)

        dists_tg = iou_distance(atlbrs_tg, btlbrs_tg)
        dists_tg = fuse_score(dists_tg, dets_score_classes_second_tg)
        dists = dists_tg.numpy()
        matches_tg, u_unconfirmed_tg, u_detection_tg = linear_assignment(dists, thresh=0.7)
        u_unconfirmed = u_unconfirmed_tg.numpy()
        tracks_values_tg = Tensor.empty((0,6),dtype=dtypes.float32)
        if matches_tg.shape[0] > 0:
            itracked_arr_tg = matches_tg[:, 0]
            itracked_arr = itracked_arr_tg.numpy()
            tracks_values_tg = unconfirmed_values_tg[itracked_arr_tg]
            scores_tg = dets_score_classes_second_tg[matches_tg[:, 1]][:, 4]
            scores = scores_tg.numpy()
            tlwh_tg = dets_score_classes_second_tg[matches_tg[:, 1]][:, :4]
            xyahs_tg = tlwh_to_xyah_batch(tlwh_tg)
            means_tg = unconfirmed_means_tg[itracked_arr_tg]
            covs_tg = unconfirmed_covs_tg[itracked_arr_tg]
            updated_means_tg, updated_covs_tg = self.kalman_filter.update_batch(means_tg, covs_tg, xyahs_tg)
            updated_means, updated_covs = updated_means_tg.numpy(), updated_covs_tg.numpy()
            activated_stracks_means_tg = activated_stracks_means_tg.cat(updated_means_tg)
            activated_stracks_covs_tg = activated_stracks_covs_tg.cat(updated_covs_tg)
            activated_stracks_fids_tg = activated_stracks_fids_tg.cat(Tensor(self.frame_id).repeat(len(itracked_arr)))
            activated_stracks_states_tg = activated_stracks_states_tg.cat(Tensor(TrackState.Tracked).repeat(itracked_arr_tg.shape[0]))
            activated_stracks_ids_tg = activated_stracks_ids_tg.cat(unconfirmed_ids_tg[itracked_arr_tg])
            activated_stracks_startframes_tg = activated_stracks_startframes_tg.cat(unconfirmed_startframes_tg[itracked_arr_tg])

            tracks_values_tg[:itracked_arr_tg.shape[0], 4] = scores_tg
            
            unconfirmed_ids_arr = unconfirmed_ids_tg.numpy()
            tracked_ids_arr = np.array(self.tracked_stracks_ids)
            _, tracked_indices = np.where(unconfirmed_ids_arr[itracked_arr][:, None] == tracked_ids_arr)
            tracked_indices_tg = Tensor(tracked_indices,dtype=dtypes.int)
            self.tracked_stracks_bools_tg[tracked_indices_tg] = True
            self.tracked_stracks_states_tg[tracked_indices_tg] = TrackState.Tracked

        if activated_stracks_values_tg.shape[0] > 0:
            activated_stracks_values_tg = activated_stracks_values_tg.cat(tracks_values_tg)
        else:
           activated_stracks_values_tg = tracks_values_tg

        u_unconfirmed_np = np.asarray(u_unconfirmed)
        unconfirmed_ids = unconfirmed_ids_tg.numpy()
        ids = unconfirmed_ids[u_unconfirmed_np]
        if ids.size > 0:
            removed_stracks_ids.extend(ids.tolist())
        
        bools_tg = dets_score_classes_second_tg[u_detection_tg][:,4] >= self.det_thresh
        idx_tg = nonzero_indices_1d(bools_tg).cast(dtype=dtypes.int)
        idx_tg = u_detection_tg[idx_tg]

        tlwh_tg = dets_score_classes_second_tg[idx_tg][:,:4]
        xyahs_tg = tlwh_to_xyah_batch(tlwh_tg)
        new_ids = idx_tg.shape[0]
        activated_stracks_ids_tg = activated_stracks_ids_tg.cat(Tensor.arange(self._count+1,self._count+new_ids+1))
        self._count += new_ids
        x_tg, y_tg = self.kalman_filter.initiate_batch(xyahs_tg)
        activated_stracks_fids_tg = activated_stracks_fids_tg.cat(Tensor(self.frame_id).repeat(new_ids))


        if self.frame_id == 1:
            activated_stracks_bools_tg = Tensor(True).repeat(idx_tg.shape[0])
        else:
            activated_stracks_bools_tg = activated_stracks_bools_tg.cat(Tensor(False).repeat(matches_tg.shape[0]))

        if activated_stracks_means_tg.shape[0] > 0:
            activated_stracks_means_tg = activated_stracks_means_tg.cat(x_tg)
        else:
            activated_stracks_means_tg = x_tg
        
        if activated_stracks_covs_tg.shape[0] > 0:
            activated_stracks_covs_tg = activated_stracks_covs_tg.cat(y_tg)
        else:
            activated_stracks_covs_tg = y_tg
        valid_indices_tg = idx_tg
        if activated_stracks_values_tg.shape[0] > 0:
            activated_stracks_values_tg = activated_stracks_values_tg.cat(dets_score_classes_second_tg[valid_indices_tg])
        else:
           activated_stracks_values_tg = dets_score_classes_second_tg[valid_indices_tg]

        remove_mask_tg = (self.frame_id - self.lost_stracks_fids_tg) > self.max_time_lost
        self.lost_stracks_ids_tg *= ~remove_mask_tg

        mask_tg = self.tracked_stracks_states_tg == TrackState.Tracked
        self.tracked_stracks_ids_tg *= mask_tg
        mask_tg = self.tracked_stracks_ids_tg != 0

        self.tracked_stracks_ids2_tg = self.tracked_stracks_ids_tg * mask_tg
        a_exp = activated_stracks_ids_tg.reshape(-1, 1)
        b_exp = self.tracked_stracks_ids2_tg.reshape(1, -1)
        matches = (a_exp == b_exp)
        match_counts = matches.sum(axis=1)
        not_in_mask = (match_counts == 0)
        idxs = Tensor.arange(activated_stracks_ids_tg.shape[0])
        masked_idxs = (idxs * not_in_mask)
        sorted_idxs = masked_idxs.sort(descending=True)[1]
        sorted_mask = not_in_mask[masked_idxs]
        count = int(sorted_mask.sum().item())
        keep_activated_tg = sorted_idxs[:count][::-1]

        self.tracked_stracks_ids_tg = self.tracked_stracks_ids_tg.cat(activated_stracks_ids_tg[keep_activated_tg])
        self.tracked_stracks_fids_tg = self.tracked_stracks_fids_tg.cat(activated_stracks_fids_tg[keep_activated_tg])
        self.tracked_stracks_states_tg = self.tracked_stracks_states_tg.cat(activated_stracks_states_tg[keep_activated_tg])
        self.tracked_stracks_startframes_tg = self.tracked_stracks_startframes_tg.cat(activated_stracks_startframes_tg[keep_activated_tg])
        self.tracked_stracks_bools_tg = self.tracked_stracks_bools_tg.cat(activated_stracks_bools_tg[keep_activated_tg])
        if self.tracked_stracks_values_tg.shape[0] > 0:
            self.tracked_stracks_values_tg = self.tracked_stracks_values_tg.cat(activated_stracks_values_tg[keep_activated_tg])
            self.tracked_stracks_means_tg = self.tracked_stracks_means_tg.cat(activated_stracks_means_tg[keep_activated_tg])
            self.tracked_stracks_covs_tg = self.tracked_stracks_covs_tg.cat(activated_stracks_covs_tg[keep_activated_tg])
        else:
           self.tracked_stracks_values_tg = activated_stracks_values_tg[keep_activated_tg]
           self.tracked_stracks_means_tg = activated_stracks_means_tg[keep_activated_tg]
           self.tracked_stracks_covs_tg = activated_stracks_covs_tg[keep_activated_tg]

        refind_stracks_ids_tg = Tensor(refind_stracks_ids,dtype=dtypes.int)
        refind_stracks_means_tg = Tensor(refind_stracks_means)
        refind_stracks_covs_tg = Tensor(refind_stracks_covs)

        self.tracked_stracks_means2_tg = self.tracked_stracks_means_tg
        self.tracked_stracks_ids2_tg = self.tracked_stracks_ids_tg
        self.tracked_stracks_bools2_tg = self.tracked_stracks_bools_tg

        self.tracked_stracks_ids_tg = self.tracked_stracks_ids_tg.cat(refind_stracks_ids_tg)
        if len(refind_stracks_bools_tg.shape) > 0:
            self.tracked_stracks_bools_tg = self.tracked_stracks_bools_tg.cat(refind_stracks_bools_tg)
            self.tracked_stracks_bools2_tg = self.tracked_stracks_bools2_tg.cat(refind_stracks_bools2_tg)
            self.tracked_stracks_startframes_tg = self.tracked_stracks_startframes_tg.cat(refind_stracks_startframes_tg)
        if refind_stracks_means_tg.shape[0] > 0:
            self.tracked_stracks_means_tg = self.tracked_stracks_means_tg.cat(refind_stracks_means_tg)
            self.tracked_stracks_covs_tg = self.tracked_stracks_covs_tg.cat(refind_stracks_covs_tg)
        if refind_stracks_values_tg.shape[0] > 0:
            self.tracked_stracks_values_tg = self.tracked_stracks_values_tg.cat(refind_stracks_values_tg)
            self.tracked_stracks_ids2_tg = self.tracked_stracks_ids2_tg.cat(refind_stracks_ids2_tg)
            self.tracked_stracks_fids_tg = self.tracked_stracks_fids_tg.cat(refind_stracks_fids_tg)
            self.tracked_stracks_states_tg = self.tracked_stracks_states_tg.cat(refind_stracks_states_tg)

        
        a_exp = self.lost_stracks_ids_tg.reshape(-1, 1)
        b_exp = self.tracked_stracks_ids_tg.reshape(1, -1)
        matches = (a_exp == b_exp).float()
        match_counts = matches.sum(axis=1)
        mask_tg = (match_counts == 0)

        self.lost_stracks_ids_tg *= mask_tg

        if self.lost_stracks_means_tg.shape[0] > 0: self.lost_stracks_means_tg[:,7] = 0
        
        if self.lost_stracks_values_tg.shape[0] == 0:
            self.lost_stracks_values_tg = lost_stracks_values_tg
            self.lost_stracks_means_tg = lost_stracks_means_tg
            self.lost_stracks_bools_tg = lost_stracks_bools_tg
            self.lost_stracks_covs_tg = lost_stracks_covs_tg
            self.lost_stracks_ids_tg = lost_stracks_ids_tg
            self.lost_stracks_fids_tg = lost_stracks_fids_tg
            self.lost_stracks_startframes_tg = lost_stracks_startframes_tg
            self.lost_stracks_states_tg = lost_stracks_states_tg
            self.lost_stracks_states_tg = lost_stracks_states_tg
        elif lost_stracks_values_tg.shape[0] != 0:
            self.lost_stracks_values_tg = self.lost_stracks_values_tg.cat(lost_stracks_values_tg)
            self.lost_stracks_means_tg = self.lost_stracks_means_tg.cat(lost_stracks_means_tg)
            self.lost_stracks_bools_tg = self.lost_stracks_bools_tg.cat(lost_stracks_bools_tg)
            self.lost_stracks_covs_tg = self.lost_stracks_covs_tg.cat(lost_stracks_covs_tg)
            self.lost_stracks_ids_tg = self.lost_stracks_ids_tg.cat(lost_stracks_ids_tg)
            self.lost_stracks_fids_tg = self.lost_stracks_fids_tg.cat(lost_stracks_fids_tg)
            self.lost_stracks_startframes_tg = self.lost_stracks_startframes_tg.cat(lost_stracks_startframes_tg)
            self.lost_stracks_states_tg = self.lost_stracks_states_tg.cat(lost_stracks_states_tg)

        self.tracked_stracks_ids = self.tracked_stracks_ids_tg.numpy()

        self.tracked_stracks_ids2 = self.tracked_stracks_ids2_tg.numpy()
        self.tracked_stracks_states = self.tracked_stracks_states_tg.numpy()

        output_stracks_values_tg = self.tracked_stracks_values_tg * self.tracked_stracks_bools2_tg.unsqueeze(-1)
        output_stracks_ids2_tg = self.tracked_stracks_ids2_tg * self.tracked_stracks_bools2_tg

        output_stracks_ids2 = output_stracks_ids2_tg.numpy()[:self.tracked_stracks_bools_tg.shape[0]]
        output_stracks_values = output_stracks_values_tg.numpy()[:self.tracked_stracks_bools_tg.shape[0]]
        
        zeros = self.tracked_stracks_ids != 0
        zeros_tg = nonzero_indices_1d(Tensor(zeros) == True)

        self.tracked_stracks_bools_tg = self.tracked_stracks_bools_tg[zeros_tg]
        self.tracked_stracks_means_tg = self.tracked_stracks_means_tg[zeros_tg]
        self.tracked_stracks_covs_tg = self.tracked_stracks_covs_tg[zeros_tg]

        zeros2 = self.tracked_stracks_ids2 != 0
        zeros2_tg = nonzero_indices_1d(Tensor(zeros2) == True)

        self.tracked_stracks_states_tg = self.tracked_stracks_states_tg[zeros2_tg]
        self.tracked_stracks_values_tg = self.tracked_stracks_values_tg[zeros2_tg]
        self.tracked_stracks_startframes_tg = self.tracked_stracks_startframes_tg[zeros2_tg]
        self.tracked_stracks_fids_tg = self.tracked_stracks_fids_tg[zeros2_tg]

        self.tracked_stracks_ids = self.tracked_stracks_ids2[zeros2]
        
        zeros = nonzero_indices_1d(self.lost_stracks_ids_tg != 0).cast(dtype=dtypes.int)
        self.lost_stracks_ids_tg = self.lost_stracks_ids_tg[zeros]
        self.lost_stracks_fids_tg = self.lost_stracks_fids_tg[zeros]
        self.lost_stracks_startframes_tg = self.lost_stracks_startframes_tg[zeros]
        self.lost_stracks_states_tg = self.lost_stracks_states_tg[zeros]
        self.lost_stracks_bools_tg = self.lost_stracks_bools_tg[zeros]
        self.lost_stracks_values_tg = self.lost_stracks_values_tg[zeros]
        self.lost_stracks_means_tg = self.lost_stracks_means_tg[zeros]
        self.lost_stracks_covs_tg = self.lost_stracks_covs_tg[zeros]
        
        v,m,i = output_stracks_values, self.tracked_stracks_means_tg.numpy(), output_stracks_ids2
        return v,m,i

def nonzero_indices_1d(mask: Tensor) -> Tensor:
    size = mask.shape[0]
    count = int(mask.sum().item())

    if size == 0 or count == 0:
        return Tensor([])  # empty mask or all False

    if count == 1:
        # quick path: single True → return its index
        return Tensor([(mask * Tensor.arange(size)).sum().item()])

    idxs = Tensor.arange(size)
    masked = idxs * mask
    sorted_vals, sorted_idxs = masked.sort(descending=True)
    return sorted_idxs[:count][::-1]


def ious(atlbrs, btlbrs):
    ious = Tensor.zeros((atlbrs.shape[0], btlbrs.shape[0]), dtype=dtypes.float32)
    if ious.shape[0] == 0:
        return ious
    ious = bbox_ious(atlbrs, btlbrs)
    return ious

def iou_distance(atlbrs, btlbrs):
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious
    return cost_matrix

def fuse_score(cost_matrix, det_values_tg):
    if cost_matrix.shape[0] == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores_tg = (det_values_tg)[:, 4]
    det_scores_tg = det_scores_tg.unsqueeze(0).expand(cost_matrix.shape[0], -1)
    fuse_sim = iou_sim * det_scores_tg
    fuse_cost = 1 - fuse_sim
    return fuse_cost

def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return Tensor.empty((0, 2), dtype=dtypes.int), Tensor.arange(cost_matrix.shape[0]), Tensor.arange(cost_matrix.shape[1])
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    matched_mask = x >= 0
    matches = np.column_stack((np.arange(len(x))[matched_mask],x[matched_mask]))
    unmatched_a = np.where(~matched_mask)[0]
    unmatched_b = np.where(y < 0)[0]
    return Tensor(matches), Tensor(unmatched_a), Tensor(unmatched_b)


#Model architecture from https://github.com/ultralytics/ultralytics/issues/189
#The upsampling class has been taken from this pull request https://github.com/tinygrad/tinygrad/pull/784 by dc-dc-dc. Now 2(?) models use upsampling. (retinet and this)

#Pre processing image functions.
def compute_transform(image, new_shape=(1280, 1280), auto=False, scaleFill=False, scaleup=True, stride=32) -> Tensor:
  shape = image.shape[:2]  # current shape [height, width]
  new_shape = (new_shape, new_shape) if isinstance(new_shape, int) else new_shape
  r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
  r = min(r, 1.0) if not scaleup else r
  new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
  dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
  dw, dh = (np.mod(dw, stride), np.mod(dh, stride)) if auto else (0.0, 0.0)
  new_unpad = (new_shape[1], new_shape[0]) if scaleFill else new_unpad
  dw /= 2
  dh /= 2
  image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR) if shape[::-1] != new_unpad else image
  top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
  left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
  image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
  return Tensor(image)

def preprocess(im, imgsz=1280, model_stride=32, model_pt=True):
  im = compute_transform(im, new_shape=imgsz, auto=True, stride=model_stride)
  im = im.unsqueeze(0)
  im = im[..., ::-1].permute(0, 3, 1, 2)
  im = im / 255.0
  return im

# utility functions for forward pass.
def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
  lt, rb = distance.chunk(2, dim)
  x1y1 = anchor_points - lt
  x2y2 = anchor_points + rb
  if xywh:
    c_xy = (x1y1 + x2y2) / 2
    wh = x2y2 - x1y1
    return c_xy.cat(wh, dim=1)
  return x1y1.cat(x2y2, dim=1)

def make_anchors(feats, strides, grid_cell_offset=0.5):
  anchor_points, stride_tensor = [], []
  assert feats is not None
  for i, stride in enumerate(strides):
    _, _, h, w = feats[i].shape
    sx = Tensor.arange(w) + grid_cell_offset
    sy = Tensor.arange(h) + grid_cell_offset

    # this is np.meshgrid but in tinygrad
    sx = sx.reshape(1, -1).repeat([h, 1]).reshape(-1)
    sy = sy.reshape(-1, 1).repeat([1, w]).reshape(-1)

    anchor_points.append(Tensor.stack(sx, sy, dim=-1).reshape(-1, 2))
    stride_tensor.append(Tensor.full((h * w), stride))
  anchor_points = anchor_points[0].cat(anchor_points[1], anchor_points[2])
  stride_tensor = stride_tensor[0].cat(stride_tensor[1], stride_tensor[2]).unsqueeze(1)
  return anchor_points, stride_tensor

# this function is from the original implementation
def autopad(k, p=None, d=1):  # kernel, padding, dilation
  if d > 1:
    k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
  if p is None:
    p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
  return p

def clip_boxes(boxes, shape):
  boxes[..., [0, 2]] = np.clip(boxes[..., [0, 2]], 0, shape[1])  # x1, x2
  boxes[..., [1, 3]] = np.clip(boxes[..., [1, 3]], 0, shape[0])  # y1, y2
  return boxes

def scale_boxes(img1_shape, predictions, img0_shape, ratio_pad=None):
  gain = ratio_pad if ratio_pad else min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
  pad = ((img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2)
  for pred in predictions:
    boxes_np = pred[:4].numpy() if isinstance(pred[:4], Tensor) else pred[:4]
    boxes_np[..., [0, 2]] -= pad[0]
    boxes_np[..., [1, 3]] -= pad[1]
    boxes_np[..., :4] /= gain
    boxes_np = clip_boxes(boxes_np, img0_shape)
    pred[:4] = boxes_np
  return predictions

def get_variant_multiples(variant):
  return {'n':(0.33, 0.25, 2.0), 's':(0.33, 0.50, 2.0), 'm':(0.67, 0.75, 1.5), 'l':(1.0, 1.0, 1.0), 'x':(1, 1.25, 1.0) }.get(variant, None)

def label_predictions(all_predictions):
  class_index_count = defaultdict(int)
  for pred in all_predictions:
    class_id = int(pred[-1])
    if pred[-2] != 0: class_index_count[class_id] += 1

  return dict(class_index_count)

#this is taken from https://github.com/tinygrad/tinygrad/pull/784/files by dc-dc-dc (Now 2 models use upsampling)
class Upsample:
  def __init__(self, scale_factor:int, mode: str = "nearest") -> None:
    assert mode == "nearest" # only mode supported for now
    self.mode = mode
    self.scale_factor = scale_factor

  def __call__(self, x: Tensor) -> Tensor:
    assert len(x.shape) > 2 and len(x.shape) <= 5
    (b, c), _lens = x.shape[:2], len(x.shape[2:])
    tmp = x.reshape([b, c, -1] + [1] * _lens) * Tensor.ones(*[1, 1, 1] + [self.scale_factor] * _lens)
    return tmp.reshape(list(x.shape) + [self.scale_factor] * _lens).permute([0, 1] + list(chain.from_iterable([[y+2, y+2+_lens] for y in range(_lens)]))).reshape([b, c] + [x * self.scale_factor for x in x.shape[2:]])

class Conv_Block:
  def __init__(self, c1, c2, kernel_size=1, stride=1, groups=1, dilation=1, padding=None):
    self.conv = Conv2d(c1,c2, kernel_size, stride, padding=autopad(kernel_size, padding, dilation), bias=False, groups=groups, dilation=dilation)
    self.bn = BatchNorm2d(c2, eps=0.001)

  def __call__(self, x):
    return self.bn(self.conv(x)).silu()

class Bottleneck:
  def __init__(self, c1, c2 , shortcut: bool, g=1, kernels: list = (3,3), channel_factor=0.5):
    c_ = int(c2 * channel_factor)
    self.cv1 = Conv_Block(c1, c_, kernel_size=kernels[0], stride=1, padding=None)
    self.cv2 = Conv_Block(c_, c2, kernel_size=kernels[1], stride=1, padding=None, groups=g)
    self.residual = c1 == c2 and shortcut

  def __call__(self, x):
    return x + self.cv2(self.cv1(x)) if self.residual else self.cv2(self.cv1(x))

class C2f:
  def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
    self.c = int(c2 * e)
    self.cv1 = Conv_Block(c1, 2 * self.c, 1,)
    self.cv2 = Conv_Block((2 + n) * self.c, c2, 1)
    self.bottleneck = [Bottleneck(self.c, self.c, shortcut, g, kernels=[(3, 3), (3, 3)], channel_factor=1.0) for _ in range(n)]

  def __call__(self, x):
    y= list(self.cv1(x).chunk(2, 1))
    y.extend(m(y[-1]) for m in self.bottleneck)
    z = y[0]
    for i in y[1:]: z = z.cat(i, dim=1)
    return self.cv2(z)

class SPPF:
  def __init__(self, c1, c2, k=5):
    c_ = c1 // 2  # hidden channels
    self.cv1 = Conv_Block(c1, c_, 1, 1, padding=None)
    self.cv2 = Conv_Block(c_ * 4, c2, 1, 1, padding=None)

    # TODO: this pads with 0s, whereas torch function pads with -infinity. This results in a < 2% difference in prediction which does not make a difference visually.
    self.maxpool = lambda x : x.pad((k // 2, k // 2, k // 2, k // 2)).max_pool2d(kernel_size=k, stride=1)

  def __call__(self, x):
    x = self.cv1(x)
    x2 = self.maxpool(x)
    x3 = self.maxpool(x2)
    x4 = self.maxpool(x3)
    return self.cv2(x.cat(x2, x3, x4, dim=1))

class DFL:
  def __init__(self, c1=16):
    self.conv = Conv2d(c1, 1, 1, bias=False)
    x = Tensor.arange(c1)
    self.conv.weight.replace(x.reshape(1, c1, 1, 1))
    self.c1 = c1

  def __call__(self, x):
    b, c, a = x.shape # batch, channels, anchors
    return self.conv(x.reshape(b, 4, self.c1, a).transpose(2, 1).softmax(1)).reshape(b, 4, a)

#backbone
class Darknet:
  def __init__(self, w, r, d):
    self.b1 = [Conv_Block(c1=3, c2= int(64*w), kernel_size=3, stride=2, padding=1), Conv_Block(int(64*w), int(128*w), kernel_size=3, stride=2, padding=1)]
    self.b2 = [C2f(c1=int(128*w), c2=int(128*w), n=round(3*d), shortcut=True), Conv_Block(int(128*w), int(256*w), 3, 2, 1), C2f(int(256*w), int(256*w), round(6*d), True)]
    self.b3 = [Conv_Block(int(256*w), int(512*w), kernel_size=3, stride=2, padding=1), C2f(int(512*w), int(512*w), round(6*d), True)]
    self.b4 = [Conv_Block(int(512*w), int(512*w*r), kernel_size=3, stride=2, padding=1), C2f(int(512*w*r), int(512*w*r), round(3*d), True)]
    self.b5 = [SPPF(int(512*w*r), int(512*w*r), 5)]

  def return_modules(self):
    return [*self.b1, *self.b2, *self.b3, *self.b4, *self.b5]

  def __call__(self, x):
    x1 = x.sequential(self.b1)
    x2 = x1.sequential(self.b2)
    x3 = x2.sequential(self.b3)
    x4 = x3.sequential(self.b4)
    x5 = x4.sequential(self.b5)
    return (x2, x3, x5)

#yolo fpn (neck)
class Yolov8NECK:
  def __init__(self, w, r, d):  #width_multiple, ratio_multiple, depth_multiple
    self.up = Upsample(2, mode='nearest')
    self.n1 = C2f(c1=int(512*w*(1+r)), c2=int(512*w), n=round(3*d), shortcut=False)
    self.n2 = C2f(c1=int(768*w), c2=int(256*w), n=round(3*d), shortcut=False)
    self.n3 = Conv_Block(c1=int(256*w), c2=int(256*w), kernel_size=3, stride=2, padding=1)
    self.n4 = C2f(c1=int(768*w), c2=int(512*w), n=round(3*d), shortcut=False)
    self.n5 = Conv_Block(c1=int(512* w), c2=int(512 * w), kernel_size=3, stride=2, padding=1)
    self.n6 = C2f(c1=int(512*w*(1+r)), c2=int(512*w*r), n=round(3*d), shortcut=False)

  def return_modules(self):
    return [self.n1, self.n2, self.n3, self.n4, self.n5, self.n6]

  def __call__(self, p3, p4, p5):
    x = self.n1(self.up(p5).cat(p4, dim=1))
    head_1 = self.n2(self.up(x).cat(p3, dim=1))
    head_2 = self.n4(self.n3(head_1).cat(x, dim=1))
    head_3 = self.n6(self.n5(head_2).cat(p5, dim=1))
    return [head_1, head_2, head_3]

#task specific head.
class DetectionHead:
  def __init__(self, nc=80, filters=()):
    self.ch = 16
    self.nc = nc  # number of classes
    self.nl = len(filters)
    self.no = nc + self.ch * 4  #
    self.stride = [8, 16, 32]
    c1 = max(filters[0], self.nc)
    c2 = max((filters[0] // 4, self.ch * 4))
    self.dfl = DFL(self.ch)
    self.cv3 = [[Conv_Block(x, c1, 3), Conv_Block(c1, c1, 3), Conv2d(c1, self.nc, 1)] for x in filters]
    self.cv2 = [[Conv_Block(x, c2, 3), Conv_Block(c2, c2, 3), Conv2d(c2, 4 * self.ch, 1)] for x in filters]

  def __call__(self, x):
    for i in range(self.nl):
      x[i] = (x[i].sequential(self.cv2[i]).cat(x[i].sequential(self.cv3[i]), dim=1))
    self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
    y = [(i.reshape(x[0].shape[0], self.no, -1)) for i in x]
    x_cat = y[0].cat(y[1], y[2], dim=2)
    box, cls = x_cat[:, :self.ch * 4], x_cat[:, self.ch * 4:]
    dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
    z = dbox.cat(cls.sigmoid(), dim=1)
    return z

class YOLOv8:
  def __init__(self, w, r,  d, num_classes): #width_multiple, ratio_multiple, depth_multiple
    self.net = Darknet(w, r, d)
    self.fpn = Yolov8NECK(w, r, d)
    self.head = DetectionHead(num_classes, filters=(int(256*w), int(512*w), int(512*w*r)))

  def __call__(self, x):
    x = self.net(x)
    x = self.fpn(*x)
    x = self.head(x)
    # TODO: postprocess needs to be in the model to be compiled to webgpu
    return postprocess(x)

  def return_all_trainable_modules(self):
    backbone_modules = [*range(10)]
    yolov8neck_modules = [12, 15, 16, 18, 19, 21]
    yolov8_head_weights = [(22, self.head)]
    return [*zip(backbone_modules, self.net.return_modules()), *zip(yolov8neck_modules, self.fpn.return_modules()), *yolov8_head_weights]

def convert_f16_safetensor_to_f32(input_file: Path, output_file: Path):
  with open(input_file, 'rb') as f:
    metadata_length = int.from_bytes(f.read(8), 'little')
    metadata = json.loads(f.read(metadata_length).decode())
    float32_values = np.fromfile(f, dtype=np.float16).astype(np.float32)

  for v in metadata.values():
    if v["dtype"] == "F16": v.update({"dtype": "F32", "data_offsets": [offset * 2 for offset in v["data_offsets"]]})

  with open(output_file, 'wb') as f:
    new_metadata_bytes = json.dumps(metadata).encode()
    f.write(len(new_metadata_bytes).to_bytes(8, 'little'))
    f.write(new_metadata_bytes)
    float32_values.tofile(f)

def compute_iou_matrix(boxes):
  x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
  areas = (x2 - x1) * (y2 - y1)
  x1 = Tensor.maximum(x1[:, None], x1[None, :])
  y1 = Tensor.maximum(y1[:, None], y1[None, :])
  x2 = Tensor.minimum(x2[:, None], x2[None, :])
  y2 = Tensor.minimum(y2[:, None], y2[None, :])
  w = Tensor.maximum(Tensor(0), x2 - x1)
  h = Tensor.maximum(Tensor(0), y2 - y1)
  intersection = w * h
  union = areas[:, None] + areas[None, :] - intersection
  return intersection / union

def postprocess(output, max_det=300, conf_threshold=0.25, iou_threshold=0.45):
  xc, yc, w, h, class_scores = output[0][0], output[0][1], output[0][2], output[0][3], output[0][4:]
  class_ids = Tensor.argmax(class_scores, axis=0)
  probs = Tensor.max(class_scores, axis=0)
  probs = Tensor.where(probs >= conf_threshold, probs, 0)
  x1 = xc - w / 2
  y1 = yc - h / 2
  x2 = xc + w / 2
  y2 = yc + h / 2
  boxes = Tensor.stack(x1, y1, x2, y2, probs, class_ids, dim=1)
  order = Tensor.topk(probs, max_det)[1]
  boxes = boxes[order]
  iou = compute_iou_matrix(boxes[:, :4])
  iou = Tensor.triu(iou, diagonal=1)
  same_class_mask = boxes[:, -1][:, None] == boxes[:, -1][None, :]
  high_iou_mask = (iou > iou_threshold) & same_class_mask
  no_overlap_mask = high_iou_mask.sum(axis=0) == 0
  boxes = boxes * no_overlap_mask.unsqueeze(-1)
  return boxes

def get_weights_location(yolo_variant: str) -> Path:
  weights_location = Path(__file__).parents[1] / "weights" / f'yolov8{yolo_variant}.safetensors'
  fetch(f'https://gitlab.com/r3sist/yolov8_weights/-/raw/master/yolov8{yolo_variant}.safetensors', weights_location)
  f32_weights = weights_location.with_name(f"{weights_location.stem}_f32.safetensors")
  if not f32_weights.exists(): convert_f16_safetensor_to_f32(weights_location, f32_weights)
  return f32_weights

def draw_predictions_on_frame(frame, predictions, class_labels, color_dict):
  font = cv2.FONT_HERSHEY_SIMPLEX

  def is_bright_color(color):
    r, g, b = color
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    return brightness > 127

  h, w, _ = frame.shape
  box_thickness = int((h + w) / 400)
  font_scale = (h + w) / 2500

  for pred in predictions:
    x1, y1, x2, y2, id, class_id = pred
    x1, y1, x2, y2, class_id = map(int, (x1, y1, x2, y2, class_id))
    color = color_dict[class_labels[class_id]]
    label = f"{class_labels[class_id]} {int(id)}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
    text_size, _ = cv2.getTextSize(label, font, font_scale, 1)
    label_y, bg_y = (y1 - 4, y1 - text_size[1] - 4) if y1 - text_size[1] - 4 > 0 else (y1 + text_size[1], y1)
    cv2.rectangle(frame, (x1, bg_y), (x1 + text_size[0], bg_y + text_size[1]), color, -1)
    font_color = (0, 0, 0) if is_bright_color(color) else (255, 255, 255)
    cv2.putText(frame, label, (x1, label_y), font, font_scale, font_color, 1, cv2.LINE_AA)


class Args:
    def __init__(self):
        self.track_thresh = 0.6
        self.track_buffer = 60 #frames, was 30
        self.mot20 = False
        self.match_thresh = 0.9

tracker = BYTETracker(Args())

@TinyJit
def do_inf(image):
  predictions = yolo_infer(image)
  return predictions

from urllib.request import urlopen, urlretrieve

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print("Error: Video URL or path not provided.")
    sys.exit(1)

  video_path = sys.argv[1]
  yolo_variant = sys.argv[2] if len(sys.argv) >= 3 else (print("No variant given, so choosing 'n' as default.") or 'n')
  print(f'Running inference for YOLOv8 variant: {yolo_variant}')

  output_folder_path = Path('./video_outputs')
  output_folder_path.mkdir(parents=True, exist_ok=True)

  # Download and open the video
  local_video_path = fetch(video_path).as_posix()
  cap = cv2.VideoCapture(local_video_path)

  if not cap.isOpened():
    print("Error: Could not open video.")
    sys.exit(1)

  fps = int(cap.get(cv2.CAP_PROP_FPS))
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  out_path = (output_folder_path / f"{Path(local_video_path).stem}_output.mp4").as_posix()
  out_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

  # Load YOLOv8 model
  depth, width_mult, ratio = get_variant_multiples(yolo_variant)
  yolo_infer = YOLOv8(w=width_mult, r=ratio, d=depth, num_classes=80)
  state_dict = safe_load(get_weights_location(yolo_variant))
  load_state_dict(yolo_infer, state_dict)

  class_labels = fetch('https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names').read_text().split("\n")
  color_dict = {label: tuple((((i+1) * 50) % 256, ((i+1) * 100) % 256, ((i+1) * 150) % 256)) for i, label in enumerate(class_labels)}
  
  
  frame_count = 0
  people = set()
  outs = []
  expected_values = pickle.load(open('values.pkl', 'rb'))
  expected_values2 = pickle.load(open('values2.pkl', 'rb'))
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    frame_count += 1
    
    pre_processed = preprocess(frame)
    predictions = do_inf(pre_processed)
    values, means, track_ids = tracker.update(predictions, [1280,1280], [1280,1280])
    pred_track = np.array([np.append(tlbr_np(v,m), [tid,v[5]]) for v,m,tid in zip(values,means,track_ids)], dtype=np.float32)

    # sanity check print people
    for v,tid in zip(values,track_ids):
      if v[5] == 0 and tid != 0: 
        people.add(tid)
    
    pred_track = scale_boxes(pre_processed.shape[2:], pred_track, frame.shape)
    predictions = predictions.numpy()
    predictions = scale_boxes(pre_processed.shape[2:], predictions, frame.shape)

    # Draw predictions
    #draw_predictions_on_frame(frame, predictions, class_labels, color_dict)
    draw_predictions_on_frame(frame, pred_track, class_labels, color_dict)

    out_writer.write(frame)
    
    #outs.append(values)
    if sys.argv[1] == "https://motchallenge.net/sequenceVideos/MOT17-08-DPM-raw.mp4":
        if not np.array_equal(np.array(expected_values[frame_count - 1]), values):
          print("wrong output")
          exit()
          #outs.append(values)
    else:
        if not np.array_equal(np.array(expected_values2[frame_count - 1]), values):
          print("wrong output")
          exit()
    


    if frame_count % 10 == 0:
      print(f"Processed frame {frame_count}")
      print(len(people))

  #if sys.argv[1] == "https://motchallenge.net/sequenceVideos/MOT17-08-DPM-raw.mp4":
  #    pickle.dump(outs, open('values.pkl', 'wb'))
  #else:
  #   pickle.dump(outs, open('values2.pkl', 'wb'))
  cap.release()
  out_writer.release()
  print(f"Saved processed video to {out_path}")

#https://motchallenge.net/sequenceVideos/MOT17-08-DPM-raw.mp4 73
#https://motchallenge.net/sequenceVideos/MOT17-03-FRCNN-raw.mp4 173




