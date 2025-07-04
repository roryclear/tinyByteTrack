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
        
        motion_mat_tg = Tensor(self._motion_mat,dtype=dtypes.float32)

        r = std_pos_tg.cat(std_vel_tg)
        sqr = Tensor.square(r).T
        batch_size = sqr.shape[0]
        dim = sqr.shape[1]
        motion_cov_tg = Tensor.eye(dim).reshape(1, dim, dim) * sqr.reshape(batch_size, dim, 1)
        mean_tg = Tensor.dot(mean_tg, motion_mat_tg.T)
        left_tg = Tensor.dot(motion_mat_tg,covariance_tg)
        covariance_tg = Tensor.dot(left_tg, motion_mat_tg.T) + motion_cov_tg
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

    def update_batch(self, means, covariances, measurements):
        if means.shape[0] == 0: return means, covariances
        mean_tg = Tensor(means, dtype=dtypes.float32)
        covariances_tg = Tensor(covariances, dtype=dtypes.float32)
        projected_means_tg, projected_covs_tg = self.project_batch(mean_tg, covariances_tg)
        projected_means = projected_means_tg.numpy()
        chol_factors_tg = self.cholesky(projected_covs_tg)
        chol_factors = chol_factors_tg.numpy()
        projected_covs = projected_covs_tg.numpy()
        update_mat_T = self._update_mat.T

        R = np.einsum('ijk,kl->ilj', covariances, update_mat_T)
        R_tg = Tensor(R,dtype=dtypes.float32)
        y_tg = self.solve_all_triangular(chol_factors_tg, R_tg)
        chol_factors_T = np.transpose(chol_factors, (0, 2, 1))
        chol_factors_T_tg = Tensor(chol_factors_T,dtype=dtypes.float32)
        kalman_gain_tg = self.solve_all_triangular(chol_factors_T_tg, y_tg)
        kalman_gain = kalman_gain_tg.numpy()
        innovation = measurements - projected_means
        new_means = []
        new_means = means + np.einsum('ij,ijk->ik', innovation, kalman_gain)
        new_covariances = []
        new_covariances = covariances - np.einsum('nji,njk,nkl->nil', kalman_gain, projected_covs, kalman_gain)
        return np.array(new_means), np.array(new_covariances)

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

def tlwh_to_xyah_batch(tlwh_array):
    ret = np.asarray(tlwh_array).copy()
    ret[:, :2] += ret[:, 2:] / 2
    ret[:, 2] /= ret[:, 3]
    return ret

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

        self.lost_stracks_ids_tg = Tensor.empty()
        self.lost_stracks_fids_tg = Tensor.empty()
        self.lost_stracks_startframes_tg = Tensor.empty()
        self.lost_stracks_states_tg = Tensor.empty()
        self.lost_stracks_bools_tg = Tensor.empty()
        self.lost_stracks_values_tg = Tensor.empty()
        self.lost_stracks_means_tg = Tensor.empty(dtype=dtypes.float32)
        self.lost_stracks_covs_tg = Tensor.empty(dtype=dtypes.float32)

        self.tracked_stracks_values = []
        self.lost_stracks_values = []

        self.tracked_stracks_means = []
        self.tracked_stracks_bools = []
        self.tracked_stracks_covs = []
        self.tracked_stracks_ids = []
        self.tracked_stracks_fids = []
        self.tracked_stracks_startframes = []
        self.tracked_stracks_states = []
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
        activated_stracks_values = []
        activated_stracks_means = []
        activated_stracks_bools = []
        activated_stracks_covs = []
        activated_stracks_ids = []
        activated_stracks_fids = []
        activated_stracks_startframes = []
        activated_stracks_states = []
        refind_stracks_values = []
        refind_stracks_means = []
        refind_stracks_bools = []
        refind_stracks_covs = []
        refind_stracks_ids = []
        refind_stracks_fids = []
        refind_stracks_startframes = []
        refind_stracks_states = []
        lost_stracks_means = []
        lost_stracks_bools = []
        lost_stracks_covs = []
        lost_stracks_values = []
        lost_stracks_startframes = []
        lost_stracks_states = []
        lost_stracks_ids = []
        lost_stracks_fids = []
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
        dets_score_classes = dets_score_classes_tg.numpy()
        dets_score_classes_second = dets_second.cat(scores.reshape(-1,1), dim=1).cat(classes.reshape(-1,1), dim=1)
        dets_score_classes_second = dets_score_classes_second.numpy()
      
        
        mask = np.array(self.tracked_stracks_bools).astype(bool)
        original_indices = np.where(mask)[0]
        original_indices_tg = Tensor(original_indices)
        mask_tg = Tensor(mask)

        self.tracked_stracks_ids_tg = Tensor(self.tracked_stracks_ids)
        self.tracked_stracks_fids_tg = Tensor(self.tracked_stracks_fids)
        self.tracked_stracks_bools_tg = Tensor(self.tracked_stracks_bools)
        self.tracked_stracks_startframes_tg = Tensor(self.tracked_stracks_startframes)
        self.tracked_stracks_states_tg = Tensor(self.tracked_stracks_states)
        self.tracked_stracks_values_tg = Tensor(self.tracked_stracks_values)
        self.tracked_stracks_covs_tg = Tensor(self.tracked_stracks_covs,dtype=dtypes.float32)
        self.tracked_stracks_means_tg = Tensor(self.tracked_stracks_means,dtype=dtypes.float32)

        tracked_stracks_ids_tg = self.tracked_stracks_ids_tg * mask_tg
        unconfirmed_ids_tg = self.tracked_stracks_ids_tg * ~mask_tg

        id_mask_tg = unconfirmed_ids_tg != 0
        id_mask = id_mask_tg.numpy()
        unconfirmed_ids = unconfirmed_ids_tg.numpy()
        unconfirmed_ids = unconfirmed_ids[id_mask].tolist()
        unconfirmed_values = self.tracked_stracks_values_tg.numpy()
        unconfirmed_values = unconfirmed_values[id_mask].tolist()
        unconfirmed_covs = self.tracked_stracks_covs_tg.numpy()
        unconfirmed_covs = unconfirmed_covs[id_mask].tolist()
        unconfirmed_means = self.tracked_stracks_means_tg.numpy()
        unconfirmed_means = unconfirmed_means[id_mask].tolist()
        unconfirmed_startframes = self.tracked_stracks_startframes_tg.numpy()
        unconfirmed_startframes = unconfirmed_startframes[id_mask].tolist()


        if len(self.lost_stracks_values_tg.shape) > 1 and self.lost_stracks_values_tg.shape[0] > 0:
            tracked_stracks_ids_tg = tracked_stracks_ids_tg.cat(self.lost_stracks_ids_tg)
            self.tracked_stracks_fids_tg = self.tracked_stracks_fids_tg.cat(self.lost_stracks_fids_tg)
            self.tracked_stracks_bools_tg = self.tracked_stracks_bools_tg.cat(self.lost_stracks_bools_tg)
            self.tracked_stracks_startframes_tg = self.tracked_stracks_startframes_tg.cat(self.lost_stracks_startframes_tg)
            self.tracked_stracks_states_tg = self.tracked_stracks_states_tg.cat(self.lost_stracks_states_tg)
            self.tracked_stracks_values_tg = self.tracked_stracks_values_tg.cat(self.lost_stracks_values_tg)

        tracked_stracks_ids = tracked_stracks_ids_tg.numpy() 
        id_mask = tracked_stracks_ids != 0
        tracked_stracks_ids = tracked_stracks_ids[id_mask].tolist()
        tracked_stracks_fids = self.tracked_stracks_fids_tg.numpy()
        tracked_stracks_fids = tracked_stracks_fids[id_mask].tolist()
        tracked_stracks_bools = self.tracked_stracks_bools_tg.numpy()
        tracked_stracks_bools = tracked_stracks_bools[id_mask].tolist()
        tracked_stracks_startframes = self.tracked_stracks_startframes_tg.numpy()
        tracked_stracks_startframes = tracked_stracks_startframes[id_mask].tolist()
        tracked_stracks_states = self.tracked_stracks_states_tg.numpy()
        tracked_stracks_states = tracked_stracks_states[id_mask].tolist()
        tracked_stracks_values = self.tracked_stracks_values_tg.numpy()
        tracked_stracks_values = tracked_stracks_values[id_mask].tolist()

        if len(self.tracked_stracks_means) > 0:
            self.tracked_stracks_means_tg, self.tracked_stracks_covs_tg = self.kalman_filter.multi_predict(self.tracked_stracks_means_tg, self.tracked_stracks_covs_tg)
            self.tracked_stracks_means = self.tracked_stracks_means_tg.numpy()
            self.tracked_stracks_covs =  self.tracked_stracks_covs_tg.numpy()
        if len(self.lost_stracks_means) > 0:
            self.lost_stracks_means_tg, self.lost_stracks_covs_tg = self.kalman_filter.multi_predict(self.lost_stracks_means_tg, self.lost_stracks_covs_tg)
            self.lost_stracks_means = self.lost_stracks_means_tg.numpy()
            self.lost_stracks_covs =  self.lost_stracks_covs_tg.numpy()
        
        means_in_tg = self.tracked_stracks_means_tg[original_indices_tg]
        covs_in_tg = self.tracked_stracks_covs_tg[original_indices_tg]
        
        if len(self.lost_stracks_means_tg.shape) > 0 and self.lost_stracks_means_tg.shape[0] > 0:
            means_in_tg = means_in_tg.cat(self.lost_stracks_means_tg)
            covs_in_tg = covs_in_tg.cat(self.lost_stracks_covs_tg)

        means_in = means_in_tg.numpy().tolist()
        covs_in = covs_in_tg.numpy().tolist()

        atlbrs_tg = tlbr_np_batch2(means_in_tg)
        btlbrs_tg = tlbr_np_batch3(dets_score_classes_tg)
        dists_tg = iou_distance(atlbrs_tg, btlbrs_tg)
        dists_tg = fuse_score(dists_tg, dets_score_classes_tg)
        dists = dists_tg.numpy()
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.args.match_thresh)

        
        det_values_arr = [dets_score_classes[i] for _, i in matches]
        if len(matches) > 0:
            xyahs = tlwh_to_xyah_batch(np.array(det_values_arr)[:, :4])
            means = np.array([means_in[itracked] for itracked, _ in matches])
            covs = np.array([covs_in[itracked] for itracked, _ in matches])
            updated_means, updated_covs = self.kalman_filter.update_batch(means, covs, xyahs)

            for idx, (itracked, _) in enumerate(matches):
                if itracked < len(original_indices):
                    self.tracked_stracks_means[original_indices[itracked]] = updated_means[idx]
                    self.tracked_stracks_covs[original_indices[itracked]] = updated_covs[idx]
                else:
                    self.lost_stracks_means[itracked - len(original_indices)] = updated_means[idx]
                    self.lost_stracks_covs[itracked - len(original_indices)] = updated_covs[idx]

                tracked_stracks_fids[itracked] = self.frame_id
                if itracked < len(self.tracked_stracks_fids):
                    self.tracked_stracks_fids[itracked] = self.frame_id

                if tracked_stracks_states[itracked] == TrackState.Tracked:
                    activated_stracks_values.append(tracked_stracks_values[itracked])
                    activated_stracks_means.append(self.tracked_stracks_means[original_indices[itracked]])
                    activated_stracks_bools.append(tracked_stracks_bools[itracked])
                    activated_stracks_covs.append(self.tracked_stracks_covs[original_indices[itracked]])
                    activated_stracks_ids.append(tracked_stracks_ids[itracked])
                    activated_stracks_fids.append(tracked_stracks_fids[itracked])
                    activated_stracks_startframes.append(tracked_stracks_startframes[itracked])
                    activated_stracks_states.append(tracked_stracks_states[itracked])
                else:
                    tracked_stracks_states[itracked] = TrackState.Tracked
                    tracked_stracks_bools[itracked] = True
                    refind_stracks_values.append(tracked_stracks_values[itracked])
                    if itracked < len(original_indices):
                        refind_stracks_means.append(self.tracked_stracks_means[original_indices[itracked]])
                        refind_stracks_covs.append(self.tracked_stracks_covs[original_indices[itracked]])
                    else:
                        refind_stracks_means.append(self.lost_stracks_means[itracked - len(original_indices)])
                        refind_stracks_covs.append(self.lost_stracks_covs[itracked - len(original_indices)])
                    refind_stracks_bools.append(True)
                    refind_stracks_ids.append(tracked_stracks_ids[itracked])
                    refind_stracks_fids.append(tracked_stracks_fids[itracked])
                    refind_stracks_startframes.append(tracked_stracks_startframes[itracked])
                    refind_stracks_states.append(tracked_stracks_states[itracked])
        
        tracked_indices = [i for i in u_track if tracked_stracks_states[i] == TrackState.Tracked]
        means = np.array([self.tracked_stracks_means[original_indices[i]] for i in tracked_indices])
        atlbrs = np.empty(len(means))
        if len(tracked_indices) > 0:
            atlbrs = means[:, :4].copy()
            atlbrs[:, 2] *= atlbrs[:, 3]
            atlbrs[:, :2] -= atlbrs[:, 2:] / 2
            atlbrs[:, 2:] += atlbrs[:, :2]
        btlbrs = dets_score_classes_second[:, :4].copy()
        btlbrs[:, 2:] += btlbrs[:, :2]

        atlbrs_tg = Tensor(atlbrs,dtype=dtypes.float32)
        btlbrs_tg = Tensor(btlbrs,dtype=dtypes.float32)
        dists_tg = iou_distance(atlbrs_tg, btlbrs_tg)
        dists = dists_tg.numpy()

        matches, u_track2, _ = linear_assignment(dists, thresh=0.5)

        # Mark unmatched tracks as lost
        for i in range(len(u_track2)):
            self.tracked_stracks_states[original_indices[u_track[u_track2[i]]]] = TrackState.Lost

        # Build inputs for batch update
        xyahs = tlwh_to_xyah_batch(dets_score_classes_second[matches[:, 1]][:, :4])

        means = np.array([
            self.tracked_stracks_means[original_indices[u_track[itracked]]]
            for itracked, _ in matches
        ])
        covs = np.array([
            self.tracked_stracks_covs[original_indices[u_track[itracked]]]
            for itracked, _ in matches
        ])

        # Apply batched Kalman update
        updated_means, updated_covs = self.kalman_filter.update_batch(means, covs, xyahs)

        # Assign updated values back
        for i, (itracked, idet) in enumerate(matches):
            self.tracked_stracks_means[original_indices[u_track[itracked]]] = updated_means[i]
            self.tracked_stracks_covs[original_indices[u_track[itracked]]] = updated_covs[i]
            self.tracked_stracks_values[original_indices[u_track[itracked]]][4] = dets_score_classes_second[idet][4]
            tracked_stracks_fids[u_track[itracked]] = self.frame_id

            self.tracked_stracks_fids[original_indices[u_track[itracked]]] = self.frame_id
            self.tracked_stracks_states[original_indices[u_track[itracked]]] = TrackState.Tracked

            activated_stracks_values.append(self.tracked_stracks_values[original_indices[u_track[itracked]]])
            activated_stracks_means.append(self.tracked_stracks_means[original_indices[u_track[itracked]]])
            activated_stracks_bools.append(tracked_stracks_bools[u_track[itracked]])
            activated_stracks_covs.append(self.tracked_stracks_covs[original_indices[u_track[itracked]]])
            activated_stracks_ids.append(tracked_stracks_ids[u_track[itracked]])
            activated_stracks_fids.append(self.frame_id)
            activated_stracks_startframes.append(tracked_stracks_startframes[u_track[itracked]])
            activated_stracks_states.append(tracked_stracks_states[u_track[itracked]])
        
        u_track3 = np.asarray(u_track)[np.asarray(u_track2)]
        lost_stracks_values = (np.array(self.tracked_stracks_values)[original_indices][u_track3]).tolist()
        lost_stracks_means = (np.array(self.tracked_stracks_means)[original_indices[u_track3]]).tolist()
        lost_stracks_bools = (np.array(tracked_stracks_bools)[u_track3]).tolist()
        lost_stracks_ids = (np.array(tracked_stracks_ids)[u_track3]).tolist()
        lost_stracks_fids = (np.array(tracked_stracks_fids)[u_track3]).tolist()
        lost_stracks_covs = (np.array(self.tracked_stracks_covs)[original_indices[u_track3]]).tolist()
        lost_stracks_startframes = (np.array(tracked_stracks_startframes)[u_track3]).tolist()
        lost_stracks_states = [TrackState.Lost] * len(u_track2)
                
        u_detection_np = np.array(u_detection)
        dets_score_classes_second = np.array(dets_score_classes)[u_detection_np]
        
        unconfirmed_means_tg = Tensor(unconfirmed_means,dtype=dtypes.float32)
        dets_score_classes_second_tg = Tensor(dets_score_classes_second)

        atlbrs_tg = tlbr_np_batch2(unconfirmed_means_tg)
        btlbrs_tg = tlbr_np_batch3(dets_score_classes_second_tg)

        dists_tg = iou_distance(atlbrs_tg, btlbrs_tg)
        dists_tg = fuse_score(dists_tg, dets_score_classes_second_tg)
        dists = dists_tg.numpy()
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)

        tracks_values = []

        if len(matches) > 0:
            itracked_arr = np.array(matches)[:, 0]
            tracks_values = [unconfirmed_values[i] for i in itracked_arr]
            scores = dets_score_classes_second[matches[:, 1]][:, 4]
            xyahs = tlwh_to_xyah_batch(dets_score_classes_second[matches[:, 1]][:, :4])
            means = np.array([unconfirmed_means[i] for i in itracked_arr])
            covs = np.array([unconfirmed_covs[i] for i in itracked_arr])
            updated_means, updated_covs = self.kalman_filter.update_batch(means, covs, xyahs)
            for i in range(len(itracked_arr)):
                activated_stracks_means.append(updated_means[i])
                activated_stracks_covs.append(updated_covs[i])
                activated_stracks_ids.append(unconfirmed_ids[itracked_arr[i]])
                activated_stracks_fids.append(self.frame_id)
                activated_stracks_startframes.append(unconfirmed_startframes[itracked_arr[i]])
                activated_stracks_states.append(TrackState.Tracked)

            for i in range(len(itracked_arr)):
                tracks_values[i][4] = scores[i] 

                if unconfirmed_ids[itracked_arr[i]] in self.tracked_stracks_ids:
                  idx = self.tracked_stracks_ids.index(unconfirmed_ids[itracked_arr[i]])
                  self.tracked_stracks_bools[idx] = True
                  self.tracked_stracks_states[idx] = TrackState.Tracked

        # todo just add to these instead of updated_?
        activated_stracks_bools.extend([False for i in np.array(matches)[:, 0]])
        activated_stracks_values.extend(tracks_values)

        u_unconfirmed_np = np.asarray(u_unconfirmed)
        ids = np.fromiter((unconfirmed_ids[key] for key in u_unconfirmed_np), dtype=object)
        if ids.size > 0:
            removed_stracks_ids.extend(ids.tolist())

        u_detection = np.asarray(u_detection)
        track_scores = dets_score_classes_second[u_detection, 4]  # Direct score access
        valid_mask = track_scores >= self.det_thresh
        valid_indices = u_detection[valid_mask].tolist()  # Convert to list of integers

        
        xyahs = tlwh_to_xyah_batch(dets_score_classes_second[u_detection[valid_mask]][:,:4])
        for i in range(len(valid_indices)):
            self._count += 1
            activated_stracks_ids.append(self._count)
            x, y = self.kalman_filter.initiate(xyahs[i])
            activated_stracks_means.append(x)
            activated_stracks_covs.append(y)
            if self.frame_id == 1: activated_stracks_bools.append(True)
            activated_stracks_fids.append(self.frame_id)

        activated_stracks_values.extend(dets_score_classes_second[valid_indices])

        self.lost_stracks_fids_tg = Tensor(self.lost_stracks_fids)
        remove_mask_tg = (self.frame_id - self.lost_stracks_fids_tg) > self.max_time_lost
        remove_mask = remove_mask_tg.numpy()

        remove_mask = (self.frame_id - np.array(self.lost_stracks_fids)) > self.max_time_lost
        self.lost_stracks_means = np.array(self.lost_stracks_means)[~remove_mask]
        self.lost_stracks_bools = np.array(self.lost_stracks_bools)[~remove_mask]
        self.lost_stracks_values = (np.array(self.lost_stracks_values)[~remove_mask])
        self.lost_stracks_covs = np.array(self.lost_stracks_covs)[~remove_mask]
        self.lost_stracks_ids = np.array(self.lost_stracks_ids)[~remove_mask]
        self.lost_stracks_fids = np.array(self.lost_stracks_fids)[~remove_mask]
        self.lost_stracks_startframes = np.array(self.lost_stracks_startframes)[~remove_mask]
        self.lost_stracks_states = np.array(self.lost_stracks_states)[~remove_mask]

        self.tracked_stracks_states_tg = Tensor(self.tracked_stracks_states)
        mask_tg = self.tracked_stracks_states_tg == TrackState.Tracked
        self.tracked_stracks_ids_tg = Tensor(self.tracked_stracks_ids)
        self.tracked_stracks_ids_tg *= mask_tg
        self.tracked_stracks_ids = self.tracked_stracks_ids_tg.numpy()
        mask = self.tracked_stracks_ids != 0

        self.tracked_stracks_values = np.array(self.tracked_stracks_values)[mask]
        self.tracked_stracks_means = np.array(self.tracked_stracks_means)[mask]
        self.tracked_stracks_bools = np.array(self.tracked_stracks_bools)[mask]
        self.tracked_stracks_covs = np.array(self.tracked_stracks_covs)[mask]
        self.tracked_stracks_fids = np.array(self.tracked_stracks_fids)[mask]
        self.tracked_stracks_ids = np.array(self.tracked_stracks_ids)[mask]
        self.tracked_stracks_startframes = np.array(self.tracked_stracks_startframes)[mask]
        self.tracked_stracks_states = np.array(self.tracked_stracks_states)[mask]
        
        self.tracked_stracks_ids_tg = Tensor(self.tracked_stracks_ids,dtype=dtypes.int)
        activated_stracks_ids_tg = Tensor(activated_stracks_ids,dtype=dtypes.int)
        self.tracked_stracks_fids_tg = Tensor(self.tracked_stracks_fids,dtype=dtypes.int)
        activated_stracks_fids_tg = Tensor(activated_stracks_fids,dtype=dtypes.int)
        self.tracked_stracks_states_tg = Tensor(self.tracked_stracks_states,dtype=dtypes.int)
        activated_stracks_states_tg = Tensor(activated_stracks_states,dtype=dtypes.int)
        self.tracked_stracks_startframes_tg = Tensor(self.tracked_stracks_startframes,dtype=dtypes.int)
        activated_stracks_startframes_tg = Tensor(activated_stracks_startframes)
        self.tracked_stracks_bools_tg = Tensor(self.tracked_stracks_bools,dtype=dtypes.bool)
        activated_stracks_bools_tg = Tensor(activated_stracks_bools,dtype=dtypes.bool)
        self.tracked_stracks_values_tg = Tensor(self.tracked_stracks_values,dtype=dtypes.float32)
        activated_stracks_values_tg = Tensor(activated_stracks_values,dtype=dtypes.float32)
        self.tracked_stracks_means_tg = Tensor(self.tracked_stracks_means,dtype=dtypes.float32)
        activated_stracks_means_tg = Tensor(activated_stracks_means,dtype=dtypes.float32)
        self.tracked_stracks_covs_tg = Tensor(self.tracked_stracks_covs,dtype=dtypes.float32)
        activated_stracks_covs_tg = Tensor(activated_stracks_covs,dtype=dtypes.float32)
        
        a_exp = activated_stracks_ids_tg.reshape(-1, 1)
        b_exp = self.tracked_stracks_ids_tg.reshape(1, -1)
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
        refind_stracks_fids_tg = Tensor(refind_stracks_fids)
        refind_stracks_bools_tg = Tensor(refind_stracks_bools)
        refind_stracks_states_tg = Tensor(refind_stracks_states,dtype=dtypes.int)
        refind_stracks_startframes_tg = Tensor(refind_stracks_startframes)
        refind_stracks_values_tg = Tensor(refind_stracks_values)
        refind_stracks_means_tg = Tensor(refind_stracks_means)
        refind_stracks_covs_tg = Tensor(refind_stracks_covs)

        self.tracked_stracks_fids_tg = self.tracked_stracks_fids_tg.cat(refind_stracks_fids_tg)
        self.tracked_stracks_ids_tg = self.tracked_stracks_ids_tg.cat(refind_stracks_ids_tg)
        self.tracked_stracks_bools_tg = self.tracked_stracks_bools_tg.cat(refind_stracks_bools_tg)
        self.tracked_stracks_states_tg = self.tracked_stracks_states_tg.cat(refind_stracks_states_tg)
        self.tracked_stracks_startframes_tg = self.tracked_stracks_startframes_tg.cat(refind_stracks_startframes_tg)
        if refind_stracks_means_tg.shape[0] > 0:
            self.tracked_stracks_means_tg = self.tracked_stracks_means_tg.cat(refind_stracks_means_tg)
            self.tracked_stracks_values_tg = self.tracked_stracks_values_tg.cat(refind_stracks_values_tg)
            self.tracked_stracks_covs_tg = self.tracked_stracks_covs_tg.cat(refind_stracks_covs_tg)
      
      
        self.lost_stracks_ids_tg = Tensor(self.lost_stracks_ids)
        a_exp = self.lost_stracks_ids_tg.reshape(-1, 1)
        b_exp = self.tracked_stracks_ids_tg.reshape(1, -1)
        matches = (a_exp == b_exp).float()
        match_counts = matches.sum(axis=1)
        mask_tg = (match_counts == 0)

        self.lost_stracks_ids_tg = Tensor(self.lost_stracks_ids)
        self.lost_stracks_fids_tg = Tensor(self.lost_stracks_fids)
        self.lost_stracks_startframes_tg = Tensor(self.lost_stracks_startframes)
        self.lost_stracks_states_tg = Tensor(self.lost_stracks_states)
        self.lost_stracks_bools_tg = Tensor(self.lost_stracks_bools)
        self.lost_stracks_values_tg = Tensor(self.lost_stracks_values)
        self.lost_stracks_means_tg = Tensor(self.lost_stracks_means,dtype=dtypes.float32)
        self.lost_stracks_covs_tg = Tensor(self.lost_stracks_covs,dtype=dtypes.float32)

        self.lost_stracks_ids_tg *= mask_tg

        if self.lost_stracks_means_tg.shape[0] > 0: self.lost_stracks_means_tg[:,7] = 0

        if self.tracked_stracks_means_tg.shape[0] > 0: 
            mask = self.tracked_stracks_states_tg == 2
            self.tracked_stracks_means_tg[:,7] *= mask

        lost_stracks_values_tg = Tensor(lost_stracks_values)
        lost_stracks_means_tg = Tensor(lost_stracks_means,dtype=dtypes.float32)
        lost_stracks_bools_tg = Tensor(lost_stracks_bools)
        lost_stracks_covs_tg = Tensor(lost_stracks_covs)
        lost_stracks_ids_tg = Tensor(lost_stracks_ids)
        lost_stracks_fids_tg = Tensor(lost_stracks_fids)
        lost_stracks_startframes_tg = Tensor(lost_stracks_startframes)
        lost_stracks_states_tg = Tensor(lost_stracks_states)

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

        self.tracked_stracks_states = self.tracked_stracks_states_tg.numpy()
        self.tracked_stracks_startframes = self.tracked_stracks_startframes_tg.numpy()
        self.tracked_stracks_values = self.tracked_stracks_values_tg.numpy()
        self.tracked_stracks_means = self.tracked_stracks_means_tg.numpy()
        self.tracked_stracks_covs = self.tracked_stracks_covs_tg.numpy()
        self.tracked_stracks_bools = self.tracked_stracks_bools_tg.numpy()
        self.tracked_stracks_fids = self.tracked_stracks_fids_tg.numpy()
        self.tracked_stracks_ids = self.tracked_stracks_ids_tg.numpy().tolist()
        self.lost_stracks_values = self.lost_stracks_values_tg.numpy()
        self.lost_stracks_means = self.lost_stracks_means_tg.numpy()
        self.lost_stracks_bools = self.lost_stracks_bools_tg.numpy()
        self.lost_stracks_covs = self.lost_stracks_covs_tg.numpy()
        self.lost_stracks_ids = self.lost_stracks_ids_tg.numpy()
        self.lost_stracks_fids = self.lost_stracks_fids_tg.numpy()
        self.lost_stracks_startframes = self.lost_stracks_startframes_tg.numpy()
        self.lost_stracks_states = self.lost_stracks_states_tg.numpy()

        output_stracks_means_tg = self.tracked_stracks_means_tg * self.tracked_stracks_bools_tg.unsqueeze(-1)
        output_stracks_values_tg = self.tracked_stracks_values_tg * self.tracked_stracks_bools_tg.unsqueeze(-1)
        output_stracks_ids_tg = self.tracked_stracks_ids_tg * self.tracked_stracks_bools_tg
        
        zeros = self.lost_stracks_ids != 0
        self.lost_stracks_ids = self.lost_stracks_ids[zeros]
        self.lost_stracks_fids = self.lost_stracks_fids[zeros]
        self.lost_stracks_startframes = self.lost_stracks_startframes[zeros]
        self.lost_stracks_states = self.lost_stracks_states[zeros]
        self.lost_stracks_bools = self.lost_stracks_bools[zeros]
        self.lost_stracks_values = self.lost_stracks_values[zeros]
        self.lost_stracks_means = self.lost_stracks_means[zeros]
        self.lost_stracks_covs = self.lost_stracks_covs[zeros]
        self.lost_stracks_means_tg = Tensor(self.lost_stracks_means)
        self.lost_stracks_covs_tg = Tensor(self.lost_stracks_covs)
        v,m,i = output_stracks_values_tg.numpy(), output_stracks_means_tg.numpy(), output_stracks_ids_tg.numpy()
        return v,m,i


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
        return np.empty((0, 2), dtype=int), np.arange(cost_matrix.shape[0]), np.arange(cost_matrix.shape[1])
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    matched_mask = x >= 0
    matches = np.column_stack((np.arange(len(x))[matched_mask],x[matched_mask]))
    unmatched_a = np.where(~matched_mask)[0]
    unmatched_b = np.where(y < 0)[0]
    return matches, unmatched_a, unmatched_b


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
  #outs = []
  expected_values = pickle.load(open('values.pkl', 'rb'))
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
    

    if sys.argv[1] == "https://motchallenge.net/sequenceVideos/MOT17-08-DPM-raw.mp4":
        if not np.array_equal(np.array(expected_values[frame_count - 1]), values):
          print("wrong output")
          exit()
    #    outs.append(values)


    if frame_count % 10 == 0:
      print(f"Processed frame {frame_count}")
      print(len(people))

  #pickle.dump(outs, open('values.pkl', 'wb'))
  cap.release()
  out_writer.release()
  print(f"Saved processed video to {out_path}")

#https://motchallenge.net/sequenceVideos/MOT17-08-DPM-raw.mp4 73
#https://motchallenge.net/sequenceVideos/MOT17-03-FRCNN-raw.mp4 173

