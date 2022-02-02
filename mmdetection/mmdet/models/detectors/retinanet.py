from ..registry import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module
class RetinaNet(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 bbox_roi_extractor,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNet, self).__init__(backbone, neck, bbox_head, bbox_roi_extractor, train_cfg,
                                        test_cfg, pretrained)
