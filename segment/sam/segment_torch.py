
from segment_anything import SamPredictor, sam_model_registry
import torch

# load the SAM model
sam = sam_model_registry["vit_h"](checkpoint="/sam_vit_h_4b8939.pth").to(device=torch.device('cpu'))

mask_predictor = SamPredictor(sam)

# transform the YOLOv8 predicted boxes to match input format expected by SAM model
transformed_boxes = mask_predictor.transform.apply_boxes_torch(predicted_boxes, image_bgr.shape[:2])


# run SAM model on all the boxes
mask_predictor.set_image(image_bgr)
masks, scores, logits = mask_predictor.predict_torch(
   boxes = transformed_boxes,
   multimask_output=False,
   point_coords=None,
   point_labels=None
)

# combine all masks into one for easy visualization
final_mask = None
for i in range(len(masks) - 1):
  if final_mask is None:
    final_mask = np.bitwise_or(masks[i][0], masks[i+1][0])
  else:
    final_mask = np.bitwise_or(final_mask, masks[i+1][0])

# visualize the predicted masks
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)
plt.imshow(final_mask, cmap='gray', alpha=0.7)
plt.show()