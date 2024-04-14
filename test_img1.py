import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor
import matplotlib.pyplot as plt
from Utility import show_masks_on_image,show_points_on_image,show_masks_on_image_single
#import pylab

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

img_path = "data/img1.jpg"
raw_image = Image.open(img_path).convert("RGB")
plt.imshow(raw_image)
plt.savefig('output/img_1.jpg',dpi=300)

#使用这些嵌入直接将它们提供给模型以加快推理速度
inputs = processor(raw_image, return_tensors="pt").to(device)
image_embeddings = model.get_image_embeddings(inputs["pixel_values"])

input_points = [[[260, 400],[250,270],[240,190]]]#感兴趣的像素点坐标
show_points_on_image(raw_image, input_points[0])#在原图上显示感兴趣的像素点坐标

inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)
# 弹出pixel_values
inputs.pop("pixel_values", None)
inputs.update({"image_embeddings": image_embeddings})

with torch.no_grad():
    outputs = model(**inputs)

masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
scores = outputs.iou_scores
show_masks_on_image_single(raw_image, masks[0], scores)