import numpy as np

from .kalman_filter import KalmanFilter
from tinygrad import Tensor
from collections import OrderedDict
from cython_bbox import bbox_overlaps as bbox_ious
import lap

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

def tlbr(x):
    """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
    `(top left, bottom right)`.
    """
    ret = tlwh(x).copy()
    ret[2:] += ret[:2]
    return ret

def tlwh(x):
    """Get current position in bounding box format `(top left x, top left y,
            width, height)`.
    """
    if x.mean is None:
        return x.values[:4].copy()
    ret = x.mean[:4].copy()
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

def activate(strack, kalman_filter, frame_id):
    """Start a new tracklet"""
    strack.kalman_filter = kalman_filter
    strack.track_id = STrack._count = STrack._count + 1
    strack.mean, strack.covariance = strack.kalman_filter.initiate(tlwh_to_xyah(strack.values[:4]))

    strack.tracklet_len = 0
    strack.state = TrackState.Tracked
    if frame_id == 1:
        strack.is_activated = True
    # self.is_activated = True
    strack.frame_id = frame_id
    strack.start_frame = frame_id

def re_activate(strack, new_track, frame_id, new_id=False):
    strack.mean, strack.covariance = strack.kalman_filter.update(
        strack.mean, strack.covariance, tlwh_to_xyah(tlwh(new_track))
    )
    strack.tracklet_len = 0
    strack.state = TrackState.Tracked
    strack.is_activated = True
    strack.frame_id = frame_id
    if new_id:
        strack.track_id = STrack._count = STrack._count + 1
    strack.values[4] = new_track.values[4]


def update(strack, new_track, frame_id):
    """
    Update a matched track
    :type new_track: STrack
    :type frame_id: int
    :type update_feature: bool
    :return:
    """
    strack.frame_id = frame_id
    strack.tracklet_len += 1

    new_tlwh = new_track.values[:4]
    strack.mean, strack.covariance = strack.kalman_filter.update(
        strack.mean, strack.covariance, tlwh_to_xyah(new_tlwh))
    strack.state = TrackState.Tracked
    strack.is_activated = True
    strack.values[4] = new_track.values[4]

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
        # todo this is dumb?, it's a [(300,4),(300),(300)] thing, just use an np (300,6) before trying tinygrad
        #detections = [STrack(tlwh, s, c) for (tlwh, s, c) in zip(dets, scores_keep, classes)]
        detections = [STrack(d) for d in dets_score_classes]

        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for i in range(len(self.tracked_stracks)):
            track = self.tracked_stracks[i]
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
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
        
        dists = iou_distance(strack_pool, detections)
        dists = fuse_score(dists, detections)
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.args.match_thresh)

        for i in range(len(matches)):
            itracked, idet = matches[i]
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                update(track, detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                re_activate(track, det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        #detections_second = [STrack(tlwh, s, c) for (tlwh, s, c) in zip(dets_second, scores_second, classes)]
        detections_second = [STrack(d) for d in dets_score_classes_second]

        r_tracked_stracks = []
        for i in range(len(u_track)):
            if strack_pool[u_track[i]].state == TrackState.Tracked:
                r_tracked_stracks.append(strack_pool[u_track[i]])
        
        dists = iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, _ = linear_assignment(dists, thresh=0.5)
        for i in range(len(matches)):
            itracked, idet = matches[i]
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                update(track, det, self.frame_id)
                activated_starcks.append(track)
            else:
                re_activate(track, det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for i in range(len(u_track)):
            track = r_tracked_stracks[u_track[i]]
            if not track.state == TrackState.Lost:
                track.state = TrackState.Lost
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        temp_detections = []
        for i in range(len(u_detection)):
            temp_detections.append(detections[u_detection[i]])
        detections = temp_detections

        atlbrs = [tlbr(track) for track in unconfirmed]
        btlbrs = [tlbr(track) for track in detections]
        _ious = ious(atlbrs, btlbrs)
        dists = 1 - _ious

        if not self.args.mot20:
            dists = fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
        for i in range(len(matches)):
            itracked, idet = matches[i]
            update(unconfirmed[itracked], detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for i in range(len(u_unconfirmed)):
            track = unconfirmed[u_unconfirmed[i]]
            track.state = TrackState.Removed
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for i in range(len(u_detection)):
            inew = u_detection[i]
            track = detections[inew]
            if track.values[4] < self.det_thresh:
                continue
            activate(track, self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for i in range(len(self.lost_stracks)):
            track = self.lost_stracks[i]
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.state = TrackState.Removed
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        temp_tracked = []
        for i in range(len(self.tracked_stracks)):
            if self.tracked_stracks[i].state == TrackState.Tracked:
                temp_tracked.append(self.tracked_stracks[i])
        self.tracked_stracks = temp_tracked
        
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = []
        for i in range(len(self.tracked_stracks)):
            if self.tracked_stracks[i].is_activated:
                output_stracks.append(self.tracked_stracks[i])

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

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for i in range(len(tlista)):
        t = tlista[i]
        exists[t.track_id] = 1
        res.append(t)
    for i in range(len(tlistb)):
        t = tlistb[i]
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for i in range(len(tlista)):
        t = tlista[i]
        stracks[t.track_id] = t
    for i in range(len(tlistb)):
        t = tlistb[i]
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for i in range(len(pairs[0])):
        p = pairs[0][i]
        q = pairs[1][i]
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = []
    for i in range(len(stracksa)):
        if i not in dupa:
            resa.append(stracksa[i])
    resb = []
    for i in range(len(stracksb)):
        if i not in dupb:
            resb.append(stracksb[i])
    return resa, resb


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """
    atlbrs = [tlbr(track) for track in atracks]
    btlbrs = [tlbr(track) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.values[4] for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost

def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b
