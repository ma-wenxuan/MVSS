import cv2
import numpy as np
import supervision as sv

import torch
import torchvision

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
import os
from PIL import Image

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"

# Segment-Anything checkpoint
# SAM_ENCODER_VERSION = "vit_h"
SAM_ENCODER_VERSION = "vit_b"
# SAM_CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"
SAM_CHECKPOINT_PATH = "./sam_vit_b_01ec64.pth"

# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH,
                             model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

# Building SAM Model and SAM Predictor
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)

# Predict classes and hyper-param for GroundingDINO
SOURCE_IMAGE_PATH = "../AUBO_python3/saved_videos/scene41/rgb0005.png"
CLASSES = ["objects"]
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8


# Prompting SAM with detected boxes
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


saved_videos = "/home/mwx/AUBO_python3/saved_videos"
for scene in range(60, 61):
    scene_path = os.path.join(saved_videos, f"scene{scene}")
    for frame in range(252):
        image_path = os.path.join(scene_path, f"rgb{str(frame).zfill(4)}.png")
        image = cv2.imread(image_path)

        # detect objects
        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()

        labels = [
            f"{CLASSES[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _, _
            in detections]
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

        # save the annotated grounding dino image
        cv2.imwrite("groundingdino_annotated_image.jpg", annotated_frame)
        new_detections = []
        THR_idx = []
        for i in range(len(detections)):
            if detections[i].box_area < 60000:
                THR_idx.append(i)
        THR_idx = np.array(THR_idx)
        detections.xyxy = detections.xyxy[THR_idx]
        detections.confidence = detections.confidence[THR_idx]
        detections.class_id = detections.class_id[THR_idx]
        # NMS post process
        print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy),
            torch.from_numpy(detections.confidence),
            NMS_THRESHOLD
        ).numpy().tolist()

        # detections = [detection for detection in detections if detection.box_area[0]<60000]
        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        print(f"After NMS: {len(detections.xyxy)} boxes")
        # convert detections to masks
        detections.mask = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )
        mask = np.zeros((480, 640), dtype=np.uint8)
        for i, obj in enumerate(detections.mask, start=1):
            mask[obj] = i * 10
        mask_image = Image.fromarray(mask)
        mask_image.save(os.path.join(scene_path, f"label{str(frame).zfill(4)}.png"))
        print(f"{i} objects in scene{scene} frame{frame}")
        # annotate image with detections
        # box_annotator = sv.BoxAnnotator()
        # mask_annotator = sv.MaskAnnotator()
        # labels = [
        #     f"{CLASSES[class_id]} {confidence:0.2f}"
        #     for _, _, confidence, class_id, _, _
        #     in detections]
        # annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        # annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        # save the annotated grounded-sam image
        # cv2.imwrite("grounded_sam_annotated_image.jpg", annotated_image)
