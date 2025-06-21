import numpy as np

from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from tinygrad import Tensor
from collections import OrderedDict

class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3
    Replaced = 4


class BaseTrack(object):
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

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    def mark_replaced(self):
        self.state = TrackState.Replaced


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self,values):

        # wait activate
        self.values = values
        self.kalman_filter = None
        self.mean = None
        self.covariance = None
        self.is_activated = False

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = tlwh(self).copy()
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
    strack.track_id = strack.next_id()
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
        strack.track_id = strack.next_id()
    strack.values[4] = new_track.values[4]

def multi_predict(stracks):
    if len(stracks) > 0:
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

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
        detections = [
            STrack(d)
            for d in dets_score_classes
        ]

        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                update(track, detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                re_activate(track,det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        #detections_second = [STrack(tlwh, s, c) for (tlwh, s, c) in zip(dets_second, scores_second, classes)]
        detections_second = [
            STrack(d)
            for d in dets_score_classes_second
        ]


        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, _ = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                update(track, det, self.frame_id)
                activated_starcks.append(track)
            else:
                re_activate(track, det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            update(unconfirmed[itracked], detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.values[4] < self.det_thresh:
                continue
            activate(track,self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb

