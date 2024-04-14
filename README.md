
## 说明
在[这里](https://huggingface.co/facebook/sam-vit-base/tree/main)下载模型

```bash
#一定要先做这步
git lfs install
```
然后git clone就行了,放到该目录下面即可

### `test_point.py`
- **功能描述**：该脚本用于在图像上打点，以便调试打点的准确性。打点后的图像将被保存到`output/spoi_image.jpg`，同时带坐标的图像可在`output/img_1.jpg`中查看。
- **调试说明**：使用的点坐标为`input_points = [[[260, 400], [250, 270], [240, 190]]]`，可根据需要修改这些坐标以观察不同的输出效果。

### `test_img1.py`
- **功能描述**：在`test_point.py`中调试确认后的坐标点将被填入此文件的第22行。该脚本将生成三个带有mask的图像，分别保存为`output/smoi_image_{i+1}.jpg`，其中i从1至3。
  
### `test.py`
- **功能描述**：这是一个官方提供的演示样例，可供参考和学习。更多信息和使用示例请参见[这里](https://github.com/huggingface/notebooks/blob/main/examples/segment_anything.ipynb)。

### `Utility.py`
- **功能描述**：这是一个工具脚本，用于显示处理后的各种图像效果。

### `data目录`
- **功能描述**：存放输入图片的目录。




