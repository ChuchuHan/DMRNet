from .coco import CocoDataset
from .registry import DATASETS
import numpy as np
from collections import defaultdict
from ..core.evaluation.bbox_overlaps import bbox_overlaps
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
import os.path as osp
import time
import re
from pycocotools.coco import COCO

@DATASETS.register_module
class PrwDataset(CocoDataset):
    CLASSES = ('person',)

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {cat_id: i+1 for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            info['cam_id'] = self._get_cam_id(info['file_name'])
            ann_ids = self.coco.getAnnIds(imgIds=[i])
            ann_info = self.coco.loadAnns(ann_ids)
            ann = self._parse_ann_info(ann_info, False)
            info.update(ann)
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(ann_info, self.with_mask)

    def _get_cam_id(self, im_name):
        match = re.search('c\d', im_name).group().replace('c', '')
        return int(match)

    def map_class_id_to_class_name(self, class_id):
        return self.CLASSES[class_id]

    def evaluate(self, predictions, dataset):
        if self.with_reid:
            result = self.evaluate_search(predictions, dataset)
        else:
            result = self.evaluate_detection(predictions)
        return result

    def evaluate_search(self, predictions, dataset, gallery_size=-1, iou_thresh=0.5):
        # detection
        pred_boxlists = []
        gt_boxlists = []
        for image_id, prediction in enumerate(tqdm(predictions[0])):
            if len(prediction) == 0:
                continue
            pred_boxlists.append(prediction[0][0][0])
            gt_boxlist = dataset[0].get_ann_info(image_id)['bboxes']
            gt_boxlists.append(gt_boxlist)

        # person search
        result = self.eval_search(predictions, dataset, gallery_size)
        det_result = self.eval_det(
            pred_boxlists=pred_boxlists,
            gt_boxlists=gt_boxlists,
            iou_thresh=iou_thresh)

        topk = [1, 3, 5, 10]
        result_str = "\n##################################################\n"
        result_str += "#############   gallery_size = {}   #############\n".format(gallery_size)
        result_str += "Detection_Recall: {:.2%}\n".format(np.nanmean(det_result["rec"][1]))
        result_str += "Detection_Precision: {:.2%}\n".format(np.nanmean(det_result["prec"][1]))
        result_str += "Detection_mean_Avg_Precision: {:.2%}\n".format(det_result["map"])
        result_str += "ReID_mean_Avg_Precision: {:.2%}\n".format(result["ReID_mean_Avg_Precision"])
        for i, k in enumerate(topk):
            result_str += "Top-{:2d} = {:.2%} \n".format(k, result["CMC"][i])
        result_str += "##################################################\n"
        print(result_str)
        return result_str

    def evaluate_detection(self, predictions, iou_thresh=0.5):
        pred_boxlists = []
        gt_boxlists = []
        for image_id, prediction in enumerate(predictions[0]):
            prediction = prediction[0]  #  TODO n_box * 5
            gt_boxlist = self.get_ann_info(image_id)['bboxes']
            if len(prediction) == 0:
                continue
            pred_boxlists.append(prediction)
            gt_boxlists.append(gt_boxlist)

        result = self.eval_det(
            pred_boxlists=pred_boxlists,
            gt_boxlists=gt_boxlists,
            iou_thresh=iou_thresh)

        result_str = "\n##################################################\n"
        result_str += "Detection_Recall: {:.2%}\n".format(np.nanmean(result["rec"][1]))
        result_str += "Detection_Precision: {:.2%}\n".format(np.nanmean(result["prec"][1]))
        result_str += "Detection_mean_Avg_Precision: {:.2%}\n".format(result["map"])
        result_str += "##################################################\n"
        print(result_str)
        return result_str

    def eval_det(self, pred_boxlists, gt_boxlists, iou_thresh=0.5, use_07_metric=False):
        """Evaluate on voc dataset.
        Args:
            pred_boxlists(list[BoxList]): pred boxlist, has labels and scores fields.
            gt_boxlists(list[BoxList]): ground truth boxlist, has labels field.
            iou_thresh: iou thresh
            use_07_metric: boolean
        Returns:
            dict represents the results
        """
        assert len(gt_boxlists) == len(pred_boxlists), "Length of gt and pred lists need to be same."
        prec, rec = self.calc_detection_prec_rec(pred_boxlists=pred_boxlists,
                                                 gt_boxlists=gt_boxlists, iou_thresh=iou_thresh)
        ap = self.calc_detection_ap(prec, rec, use_07_metric=use_07_metric)
        return {"prec": prec, "rec": rec, "map": np.nanmean(ap)}

    def eval_search(self, predictions, dataset, gallery_size=-1, det_thresh=0.5, ignore_cam_id=True):

        """
        gallery_det (list of ndarray): n_det x [x1, x2, y1, y2, score] per image
        gallery_feat (list of ndarray): n_det x D features per image
        probe_feat (list of ndarray): D dimensional features per probe image

        det_thresh (float): filter out gallery detections whose scores below this
        gallery_size (int): -1 for using full set
        ignore_cam_id (bool): Set to True acoording to CUHK-SYSU,
                              alyhough it's a common practice to focus on cross-cam match only.
        """

        dataset_test, dataset_query = dataset
        predictions_test, predictions_query = predictions
        probe_feat = [gt[-1] for gt in predictions_query]

        gallery_det = []
        gallery_feat = []

        for image_id, prediction in enumerate(predictions_test):
            if len(prediction) == 0:
                continue
            gallery_feat.append(prediction[-1])
            gallery_det.append(prediction[0][0][0])

        assert len(dataset_test) == len(gallery_det)
        assert len(dataset_test) == len(gallery_feat)
        assert len(dataset_query) == len(probe_feat)

        gt_roidb = dataset_test.img_infos
        query_roidb = dataset_query.img_infos

        # gt_roidb = gallery_set.record
        name_to_det_feat = {}
        for gt, det, feat in zip(gt_roidb, gallery_det, gallery_feat):
            name = gt['file_name']
            pids = gt['labels'][:, -1]
            cam_id = gt['cam_id']
            scores = det[:, 4].ravel()
            inds = np.where(scores >= det_thresh)[0]
            if len(inds) > 0:
                name_to_det_feat[name] = (det[inds], feat[inds], pids, cam_id)

        aps = []
        accs = []
        topk = [1, 3, 5, 10]
        ret = {}
        save_results = []
        for i in tqdm(range(len(dataset_query))):
            y_true, y_score = [], []
            imgs, rois = [], []
            count_gt, count_tp = 0, 0

            feat_p = probe_feat[i].ravel()

            probe_imname = query_roidb[i]['file_name']
            probe_roi = query_roidb[i]['bboxes']
            probe_pid = query_roidb[i]['labels'][:, -1]
            probe_cam = query_roidb[i]['cam_id']

            # Find all occurence of this probe
            gallery_imgs = []
            for x in gt_roidb:
                if probe_pid in x['labels'][:, -1] and x['file_name'] != probe_imname:
                    gallery_imgs.append(x)

            probe_gts = {}
            for item in gallery_imgs:
                probe_gts[item['file_name']] = item['bboxes'][item['labels'][:, -1] == probe_pid]

            # Construct gallery set for this probe
            if ignore_cam_id:
                gallery_imgs = []
                for x in gt_roidb:
                    if x['file_name'] != probe_imname:
                        gallery_imgs.append(x)
            else:
                gallery_imgs = []
                for x in gt_roidb:
                    if x['file_name'] != probe_imname and x['cam_id'] != probe_cam:
                        gallery_imgs.append(x)

            # # 1. Go through all gallery samples
            # for item in testset.targets_db:
            # Gothrough the selected gallery
            for item in gallery_imgs:
                gallery_imname = item['file_name']
                # some contain the probe (gt not empty), some not
                count_gt += (gallery_imname in probe_gts)
                # compute distance between probe and gallery dets
                if gallery_imname not in name_to_det_feat:
                    continue
                det, feat_g, _, _ = name_to_det_feat[gallery_imname]
                # get L2-normalized feature matrix NxD
                assert feat_g.size == np.prod(feat_g.shape[:2])
                feat_g = feat_g.reshape(feat_g.shape[:2])
                # compute cosine similarities
                sim = feat_g.dot(feat_p).ravel()
                # assign label for each det
                label = np.zeros(len(sim), dtype=np.int32)
                if gallery_imname in probe_gts:
                    gt = probe_gts[gallery_imname].ravel()
                    w, h = gt[2] - gt[0], gt[3] - gt[1]
                    iou_thresh = min(0.5, (w * h * 1.0) /
                                     ((w + 10) * (h + 10)))
                    inds = np.argsort(sim)[::-1]
                    sim = sim[inds]
                    det = det[inds]
                    # only set the first matched det as true positive
                    for j, roi in enumerate(det[:, :4]):
                        if self._compute_iou(roi, gt) >= iou_thresh:
                            label[j] = 1
                            count_tp += 1
                            break
                y_true.extend(list(label))
                y_score.extend(list(sim))
                imgs.extend([gallery_imname] * len(sim))
                rois.extend(list(det))

            # 2. Compute AP for this probe (need to scale by recall rate)
            y_score = np.asarray(y_score)
            y_true = np.asarray(y_true)
            assert count_tp <= count_gt
            recall_rate = count_tp * 1.0 / count_gt
            ap = 0 if count_tp == 0 else \
                average_precision_score(y_true, y_score) * recall_rate
            aps.append(ap)
            inds = np.argsort(y_score)[::-1]
            y_score = y_score[inds]
            y_true = y_true[inds]
            accs.append([min(1, sum(y_true[:k])) for k in topk])

        ReID_mean_Avg_Precision = np.nanmean(aps)
        accs = np.mean(accs, axis=0)

        result = {}
        result.update({'ReID_mean_Avg_Precision': ReID_mean_Avg_Precision})
        result.update({'CMC': accs})
        return result

    def calc_detection_prec_rec(self, gt_boxlists, pred_boxlists, iou_thresh=0.5):
        """Calculate precision and recall based on evaluation code of PASCAL VOC.
        This function calculates precision and recall of
        predicted bounding boxes obtained from a dataset which has :math:`N`
        images.
        The code is based on the evaluation code used in PASCAL VOC Challenge.
       """
        n_pos = defaultdict(int)
        score = defaultdict(list)
        match = defaultdict(list)
        for gt_boxlist, pred_boxlist in zip(gt_boxlists, pred_boxlists):
            pred_bbox = pred_boxlist[:, :4]
            pred_label = np.ones(pred_bbox.shape[0])  # TODO
            pred_score = pred_boxlist[:, -1]
            gt_bbox = gt_boxlist
            gt_label = np.ones(gt_bbox.shape[0])  # TODO
            gt_difficult = np.zeros(gt_bbox.shape[0])  # TODO

            for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
                pred_mask_l = pred_label == l
                pred_bbox_l = pred_bbox[pred_mask_l]
                pred_score_l = pred_score[pred_mask_l]
                # sort by score
                order = pred_score_l.argsort()[::-1]
                pred_bbox_l = pred_bbox_l[order]
                pred_score_l = pred_score_l[order]

                gt_mask_l = gt_label == l
                gt_bbox_l = gt_bbox[gt_mask_l]
                gt_difficult_l = gt_difficult[gt_mask_l]

                n_pos[l] += np.logical_not(gt_difficult_l).sum()
                score[l].extend(pred_score_l)

                if len(pred_bbox_l) == 0:
                    continue
                if len(gt_bbox_l) == 0:
                    match[l].extend((0,) * pred_bbox_l.shape[0])
                    continue

                # VOC evaluation follows integer typed bounding boxes.
                pred_bbox_l = pred_bbox_l.copy()
                pred_bbox_l[:, 2:] += 1
                gt_bbox_l = gt_bbox_l.copy()
                gt_bbox_l[:, 2:] += 1
                iou = bbox_overlaps(pred_bbox_l, gt_bbox_l)
                gt_index = iou.argmax(axis=1)
                # set -1 if there is no matching ground truth
                gt_index[iou.max(axis=1) < iou_thresh] = -1
                del iou

                selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
                for gt_idx in gt_index:
                    if gt_idx >= 0:
                        if gt_difficult_l[gt_idx]:
                            match[l].append(-1)
                        else:
                            if not selec[gt_idx]:
                                match[l].append(1)
                            else:
                                match[l].append(0)
                        selec[gt_idx] = True
                    else:
                        match[l].append(0)

        n_fg_class = max(n_pos.keys()) + 1
        prec = [None] * n_fg_class
        rec = [None] * n_fg_class

        for l in n_pos.keys():
            score_l = np.array(score[l])
            match_l = np.array(match[l], dtype=np.int8)

            order = score_l.argsort()[::-1]
            match_l = match_l[order]

            tp = np.cumsum(match_l == 1)
            fp = np.cumsum(match_l == 0)

            # If an element of fp + tp is 0,
            # the corresponding element of prec[l] is nan.
            prec[l] = tp / (fp + tp)
            # If n_pos[l] is 0, rec[l] is None.
            if n_pos[l] > 0:
                rec[l] = tp / n_pos[l]

        return prec, rec

    def calc_detection_ap(self, prec, rec, use_07_metric=False):
        """Calculate average precisions based on evaluation code of PASCAL VOC.
        This function calculates average precisions
        from given precisions and recalls.
        The code is based on the evaluation code used in PASCAL VOC Challenge.
        Args:
            prec (list of numpy.array): A list of arrays.
                :obj:`prec[l]` indicates precision for class :math:`l`.
                If :obj:`prec[l]` is :obj:`None`, this function returns
                :obj:`numpy.nan` for class :math:`l`.
            rec (list of numpy.array): A list of arrays.
                :obj:`rec[l]` indicates recall for class :math:`l`.
                If :obj:`rec[l]` is :obj:`None`, this function returns
                :obj:`numpy.nan` for class :math:`l`.
            use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
                for calculating average precision. The default value is
                :obj:`False`.
        Returns:
            ~numpy.ndarray:
            This function returns an array of average precisions.
            The :math:`l`-th value corresponds to the average precision
            for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
            :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.
        """

        n_fg_class = len(prec)
        ap = np.empty(n_fg_class)
        for l in range(n_fg_class):
            if prec[l] is None or rec[l] is None:
                ap[l] = np.nan
                continue

            if use_07_metric:
                # 11 point metric
                ap[l] = 0
                for t in np.arange(0.0, 1.1, 0.1):
                    if np.sum(rec[l] >= t) == 0:
                        p = 0
                    else:
                        p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                    ap[l] += p / 11
            else:
                # correct AP calculation
                # first append sentinel values at the end
                mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
                mrec = np.concatenate(([0], rec[l], [1]))
                mpre = np.maximum.accumulate(mpre[::-1])[::-1]

                # to calculate area under PR curve, look for points
                # where X axis (recall) changes value
                i = np.where(mrec[1:] != mrec[:-1])[0]

                # and sum (\Delta recall) * prec
                ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        return ap

    def _compute_iou(self, a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter * 1.0 / union


