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
from yolox.tracker.kalman_filter import KalmanFilter
from cython_bbox import bbox_overlaps as bbox_ious

class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3
    Replaced = 4

class STrack():
    _count = 0
    track_id = 0
    is_activated = False
    state = TrackState.New
    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0
    
    # multi-camera
    location = (np.inf, np.inf)

    shared_kalman = KalmanFilter()
    def __init__(self,values):
        # wait activate
        self.values = values
        self.kalman_filter = None
        self.mean = None
        self.covariance = None
        self.is_activated = False

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

def tlwh_np(values,mean):
    """Get current position in bounding box format `(top left x, top left y,
            width, height)`.
    """
    if mean is None:
        return values[:4].copy()
    ret = mean[:4].copy()
    ret[2] *= ret[3]
    ret[:2] -= ret[2:] / 2
    return ret

def tlwh_to_xyah(tlwh):
    """Convert bounding box to format `(center x, center y, aspect ratio,
    height)`, where the aspect ratio is `width / height`.
    """
    ret = np.asarray(tlwh).copy()
    ret[:2] += ret[2:] / 2
    ret[2] /= ret[3]
    return ret

class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, output_results, img_info, img_size):
        lost_stracks_values = [track.values for track in self.lost_stracks]
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

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
        dets_score_classes = dets.cat(scores.reshape(-1,1), dim=1).cat(classes.reshape(-1,1), dim=1)
        dets_score_classes = dets_score_classes.numpy()
        dets_score_classes_second = dets_second.cat(scores.reshape(-1,1), dim=1).cat(classes.reshape(-1,1), dim=1)
        dets_score_classes_second = dets_score_classes_second.numpy()
        detections = [STrack(d) for d in dets_score_classes]

        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for i in range(len(self.tracked_stracks)):
            track = self.tracked_stracks[i]
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        
        tracked_stracks_values = [track.values for track in tracked_stracks]
        
        ids_tracked = np.array([t.track_id for t in tracked_stracks])
        ids_lost = np.array([t.track_id for t in self.lost_stracks])
        keep_a, keep_b = joint_stracks_indices(ids_tracked, ids_lost)
        strack_pool = [tracked_stracks[i] for i in keep_a] + [self.lost_stracks[i] for i in keep_b]
        strack_values = [tracked_stracks_values[i] for i in keep_a] + [lost_stracks_values[i] for i in keep_b]
        # Predict the current location with KF
        if len(strack_pool) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in strack_pool])
            multi_covariance = np.asarray([st.covariance for st in strack_pool])
            for i in range(len(strack_pool)):
                st = strack_pool[i]
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i in range(len(strack_pool)):
                strack_pool[i].mean = multi_mean[i]
                strack_pool[i].covariance = multi_covariance[i]
        
        atlbrs = [tlbr_np(values, track.mean) for values, track in zip(strack_values, strack_pool)]
        bmeans = [track.mean for track in detections]
        btlbrs = [tlbr_np(value,mean) for value,mean in zip(dets_score_classes,bmeans)]
        dists = iou_distance(atlbrs, btlbrs)
        dists = fuse_score(dists, dets_score_classes)
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.args.match_thresh)

        det_values_arr = [dets_score_classes[i] for _, i in matches]
        det_means = [detections[i].mean for _, i in matches]
        for idx, (itracked, idet) in enumerate(matches):
            track = strack_pool[itracked]
            det_values = det_values_arr[idx]
            det_mean = det_means[idx]
            det_xyah = tlwh_to_xyah(tlwh_np(det_values, det_mean))
            track.mean, track.covariance = track.kalman_filter.update(track.mean, track.covariance, det_xyah)
            track.frame_id = self.frame_id
            if track.state == TrackState.Tracked:
                track.tracklet_len += 1
                activated_starcks.append(track)
            else:
                track.tracklet_len = 0
                track.state = TrackState.Tracked
                track.is_activated = True
                refind_stracks.append(track)

        r_tracked_stracks = []
        for i in range(len(u_track)):
            if strack_pool[u_track[i]].state == TrackState.Tracked:
                r_tracked_stracks.append(strack_pool[u_track[i]])

        r_values = [track.values for track in r_tracked_stracks]
        r_means = [track.mean for track in r_tracked_stracks]

        det_values = dets_score_classes_second
        det_means = [None] * len(det_values)  # all means are None initially

        atlbrs = [tlbr_np(v, m) for v, m in zip(r_values, r_means)]
        btlbrs = [tlbr_np(v, m) for v, m in zip(det_values, det_means)]
        dists = iou_distance(atlbrs, btlbrs)

        matches, u_track, _ = linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            t_val = r_values[itracked]
            d_val = det_values[idet]
            d_mean = det_means[idet]

            xyah = tlwh_to_xyah(tlwh_np(d_val, d_mean))
            track.mean, track.covariance = track.kalman_filter.update(track.mean, track.covariance, xyah)
            t_val[4] = d_val[4]  # Update score
            track.frame_id = self.frame_id
            
            if track.state == TrackState.Tracked:
                track.tracklet_len += 1
                activated_starcks.append(track)
            else:
                track.tracklet_len = 0
                track.state = TrackState.Tracked
                track.is_activated = True
                refind_stracks.append(track)

        refind_stracks_values = [t.values for t in refind_stracks]

        for i in range(len(u_track)):
            track = r_tracked_stracks[u_track[i]]
            if not track.state == TrackState.Lost:
                track.state = TrackState.Lost
                lost_stracks.append(track)


        u_detection_np = np.array(u_detection)
        detections = np.array(detections)[u_detection_np]
        dets_score_classes_second = np.array([det.values for det in detections])


        atlbrs = [tlbr_np(track.values,track.mean) for track in unconfirmed]
        btlbrs = [tlbr_np(track.values,track.mean) for track in detections]
        dists = iou_distance(atlbrs, btlbrs)
        dists = fuse_score(dists, dets_score_classes_second)
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)

        activated_starcks_values = [t.values for t in activated_starcks]
        tracks_values = []
        
        if len(matches) > 0:
            matches_arr = np.array(matches)
            itracked_arr = matches_arr[:, 0]
            idet_arr = matches_arr[:, 1]

            tracks = [unconfirmed[i] for i in itracked_arr]
            tracks_values = [t.values for t in tracks]
            dets = [detections[i] for i in idet_arr]

            means = [t.mean for t in tracks]
            covariances = [t.covariance for t in tracks]
            det_values = dets_score_classes_second[idet_arr]
            tlwhs = det_values[:, :4]
            scores = det_values[:, 4]

            kf = tracks[0].kalman_filter
            updated_means = []
            updated_covs = []

            for mean, cov, tlwh in zip(means, covariances, tlwhs):
                new_mean, new_cov = kf.update(mean, cov, tlwh_to_xyah(tlwh))
                updated_means.append(new_mean)
                updated_covs.append(new_cov)

            updated_scores = scores
            frame_id_val = self.frame_id
            for track, mean, cov, score, values in zip(tracks, updated_means, updated_covs, updated_scores, tracks_values):
                track.mean = mean
                track.covariance = cov
                values[4] = score
                track.frame_id = frame_id_val
                track.tracklet_len += 1
                track.state = TrackState.Tracked
                track.is_activated = True

            activated_starcks.extend(tracks)

        activated_starcks_values.extend(tracks_values)
        

        u_unconfirmed_np = np.asarray(u_unconfirmed)
        tracks = np.fromiter((unconfirmed[key] for key in u_unconfirmed_np), dtype=object)
        if tracks.size > 0:
            np.vectorize(lambda t: setattr(t, 'state', TrackState.Removed))(tracks)
            removed_stracks.extend(tracks.tolist())

        u_detection = np.asarray(u_detection)
        track_scores = dets_score_classes_second[u_detection, 4]  # Direct score access
        valid_mask = track_scores >= self.det_thresh
        valid_indices = u_detection[valid_mask].tolist()  # Convert to list of integers

        # Get tracks using proper list indexing
        valid_tracks = [detections[i] for i in valid_indices]  # Now works correctly
        valid_values = dets_score_classes_second[valid_indices]  # Get corresponding values

        # Batch activation
        for track, vals in zip(valid_tracks, valid_values):
            track.kalman_filter = self.kalman_filter
            track.track_id = STrack._count = STrack._count + 1
            track.mean, track.covariance = track.kalman_filter.initiate(
                tlwh_to_xyah(vals[:4]))
            
            track.tracklet_len = 0
            track.state = TrackState.Tracked
            if self.frame_id == 1:
                track.is_activated = True
            track.frame_id = self.frame_id
            track.start_frame = self.frame_id

        activated_starcks_values.extend(valid_values)
        activated_starcks.extend(valid_tracks)

        lost_stracks_array = np.array(self.lost_stracks, dtype=object)
        frame_ids = np.array([t.frame_id for t in self.lost_stracks], dtype=int)
        remove_mask = (self.frame_id - frame_ids) > self.max_time_lost
        removed_stracks.extend(lost_stracks_array[remove_mask].tolist())
        for t in lost_stracks_array[remove_mask]: t.state = TrackState.Removed
        self.lost_stracks = lost_stracks_array[~remove_mask].tolist()
        tracked_array = np.array(self.tracked_stracks, dtype=object)
        states = np.array([t.state for t in self.tracked_stracks], dtype=int)
        mask = states == TrackState.Tracked
        self.tracked_stracks = tracked_array[mask].tolist()
        ids_tracked = np.array([t.track_id for t in self.tracked_stracks])
        ids_activated = np.array([t.track_id for t in activated_starcks])
        keep_tracked, keep_activated = joint_stracks_indices(ids_tracked, ids_activated)

        tracked_stracks_values = [t.values for t in self.tracked_stracks]

        self.tracked_stracks = [self.tracked_stracks[i] for i in keep_tracked] + [activated_starcks[i] for i in keep_activated]
        keep_tracked_values = [tracked_stracks_values[i] for i in keep_tracked] + [activated_starcks_values[i] for i in keep_activated]


        ids_tracked = np.array([t.track_id for t in self.tracked_stracks])
        ids_refind = np.array([t.track_id for t in refind_stracks])
        keep_tracked, keep_refind = joint_stracks_indices(ids_tracked, ids_refind)
        values_a = [keep_tracked_values[i] for i in keep_tracked] + [refind_stracks_values[i] for i in keep_refind]
        self.tracked_stracks = [self.tracked_stracks[i] for i in keep_tracked] + [refind_stracks[i] for i in keep_refind]
        self.lost_stracks = [t for t in self.lost_stracks if t not in self.tracked_stracks]
        self.lost_stracks.extend([t for t in lost_stracks if t not in self.tracked_stracks])
        self.lost_stracks = [t for t in self.lost_stracks if t not in self.removed_stracks]
        self.removed_stracks.extend(removed_stracks)
        
        values_a = [track.values for track in self.tracked_stracks]
        mean_a = [track.mean for track in self.tracked_stracks]
        frame_id_a = [track.frame_id for track in self.tracked_stracks]
        start_frame_a = [track.start_frame for track in self.tracked_stracks]

        values_b = [track.values for track in self.lost_stracks]
        mean_b = [track.mean for track in self.lost_stracks]
        frame_id_b = [track.frame_id for track in self.lost_stracks]
        start_frame_b = [track.start_frame for track in self.lost_stracks]
        keep_a, keep_b = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks,
            values_a, mean_a, frame_id_a, start_frame_a,
            values_b, mean_b, frame_id_b, start_frame_b
        )
        self.tracked_stracks = [track for track, keep in zip(self.tracked_stracks, keep_a) if keep]
        self.lost_stracks = [track for track, keep in zip(self.lost_stracks, keep_b) if keep]
          
        tracked_stracks = np.array(self.tracked_stracks)
        is_activated = np.array([track.is_activated for track in tracked_stracks])
        output_stracks = tracked_stracks[is_activated].tolist()

        return output_stracks


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious
    
    ious = bbox_ious(np.ascontiguousarray(atlbrs, dtype=np.float), np.ascontiguousarray(btlbrs, dtype=np.float))
    return ious


def joint_stracks_indices(ids_a, ids_b):
    mask_b = ~np.isin(ids_b, ids_a)
    return np.arange(len(ids_a)), np.where(mask_b)[0]

def iou_distance(atlbrs, btlbrs):
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious
    return cost_matrix

def remove_duplicate_stracks(stracksa, stracksb, 
                            values_a, mean_a, frame_id_a, start_frame_a,
                            values_b, mean_b, frame_id_b, start_frame_b):
    """
    Args:
        stracksa, stracksb: Lists of objects (kept for length reference)
        values_a, values_b: List of values properties for each track
        mean_a, mean_b: List of mean properties for each track
        frame_id_a, frame_id_b: List of frame_id properties
        start_frame_a, start_frame_b: List of start_frame properties
    Returns:
        keep_a: Boolean mask of which tracks to keep from stracksa
        keep_b: Boolean mask of which tracks to keep from stracksb
    """
    atlbrs = [tlbr_np(values, mean) for values, mean in zip(values_a, mean_a)]
    btlbrs = [tlbr_np(values, mean) for values, mean in zip(values_b, mean_b)]
    pdist = iou_distance(atlbrs, btlbrs)
    pairs = np.where(pdist < 0.15)
    
    if pairs[0].size == 0: 
        return np.ones(len(stracksa), dtype=bool), np.ones(len(stracksb), dtype=bool)
        
    p_idx, q_idx = pairs[0], pairs[1]
    timep = np.array([frame_id_a[i] - start_frame_a[i] for i in p_idx])
    timeq = np.array([frame_id_b[i] - start_frame_b[i] for i in q_idx])
    
    keep_p = timep <= timeq
    keep_q = ~keep_p
    dupa = p_idx[~keep_p]
    dupb = q_idx[~keep_q]
    
    mask_a = np.ones(len(stracksa), dtype=bool)
    mask_a[dupa] = False
    mask_b = np.ones(len(stracksb), dtype=bool)
    mask_b[dupb] = False
    
    return mask_a, mask_b

def fuse_score(cost_matrix, det_values):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det[4] for det in det_values])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
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
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    frame_count += 1
    
    pre_processed = preprocess(frame)
    predictions = do_inf(pre_processed)
    online_targets = tracker.update(predictions, [1280,1280], [1280,1280])
    pred_track = np.array([np.append(tlbr_np(p.values,p.mean), [p.track_id,p.values[5]]) for p in online_targets], dtype=np.float32) # track_id as accuracy hack

    # sanity check print people
    for p in online_targets:
      if p.values[5] == 0: 
        people.add(p.track_id)
    
    pred_track = scale_boxes(pre_processed.shape[2:], pred_track, frame.shape)
    predictions = predictions.numpy()
    predictions = scale_boxes(pre_processed.shape[2:], predictions, frame.shape)

    # Draw predictions
    #draw_predictions_on_frame(frame, predictions, class_labels, color_dict)
    draw_predictions_on_frame(frame, pred_track, class_labels, color_dict)

    out_writer.write(frame)

    if frame_count % 10 == 0:
      print(f"Processed frame {frame_count}")
      print(len(people))

  cap.release()
  out_writer.release()
  print(f"Saved processed video to {out_path}")

#https://motchallenge.net/sequenceVideos/MOT17-08-DPM-raw.mp4
#https://motchallenge.net/sequenceVideos/MOT17-03-FRCNN-raw.mp4

