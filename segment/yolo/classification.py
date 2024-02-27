from ultralytics import YOLO
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
# from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch
import time

def bbox():
    model = YOLO("yolov8n.pt")
    # load the SAM model
    sam = sam_model_registry["vit_b"](checkpoint="weight/sam_vit_b_01ec64.pth").to(device=torch.device('cpu'))

    mask_predictor = SamPredictor(sam)


    img_path = "D:/inpaint_gan/car_ds/Input/LB_0_000329.jpg"
    img_ori = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # read in the image for visualization
    image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # run the model on the image
    start_box = time.time()
    results = model.predict(source=img_path, conf=0.25)
    predicted_boxes = results[0].boxes.xyxy

    # use cv2 to visualize the bounding boxes on the image
    for box in predicted_boxes:
        cv2.rectangle(image_bgr, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        # image_bgr = cv2.resize(image_bgr, (512, 256))
        # cv2.imshow("YOLOv8 predictions", image_bgr)
        # cv2.waitKey(0)
    end_box = time.time()
    print('box time: ',(end_box - start_box))
    cv2.imwrite("D:/inpaint_gan/test_img/segment/329_box.jpg", image_bgr)
#     return predicted_boxes, image_bgr

# def segment(predicted_boxes,image_bgr):

    # transform the YOLOv8 predicted boxes to match input format expected by SAM model
    print('start segment')
    start_seg = time.time()
    transformed_boxes = mask_predictor.transform.apply_boxes_torch(predicted_boxes, image_bgr.shape[:2])


    # run SAM model on all the boxes
    # image_bgr = img_ori
    mask_predictor.set_image(img_ori)
    masks, scores, logits = mask_predictor.predict_torch(
        boxes = transformed_boxes,
        multimask_output=False,
        point_coords=None,
        point_labels=None
    )

    # combine all masks into one for easy visualization
    image_rgb = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
    final_mask = None
    for i in range(len(masks) - 1):
        if final_mask is None:
            final_mask = np.bitwise_or(masks[i][0], masks[i+1][0])
        else:
            final_mask = np.bitwise_or(final_mask, masks[i+1][0])
        # final_mask = 255 * final_mask.numpy()
        # print(final_mask.max())
        # print(final_mask.min())        
        # print(type(final_mask))
        # print(final_mask)
    # visualize the predicted masks
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.imshow(final_mask,cmap='gray',alpha=0.7)
    plt.show()
    # image_save = cv2.cvtColor(final_mask, cv2.COLOR_RGB2BGR)
    end_seg = time.time()
    print('seg time: ',(end_seg - start_seg))
    print('save seg')
    final_mask = 255 * final_mask.numpy()
    if not cv2.imwrite("D:/inpaint_gan/test_img/segment/329_seg.jpg", final_mask):
        raise Exception("Could not write image")

bbox()
# img_seg = segment(box,img_bgr)