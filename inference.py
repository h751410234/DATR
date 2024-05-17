import os, sys
import torch, json
import numpy as np

from main import build_model_main
from util.slconfig import SLConfig
from util.visualizer import COCOVisualizer
from util import box_ops

import datasets.transforms as T
from PIL import Image, ImageDraw, ImageFont

colors = [
    (1,1,1),
    (0.9020, 0.5412, 0.4902),  # 橙
    (0.3804, 0.5216, 0.4863),  # 深绿
    (0.56, 0.56, 0.56),  # 灰
    (0.8314, 0.5686, 0.8510),  # 紫
    (0.6392, 0.8078, 0.7765),  # 浅绿
    (1, 1, 0),  # 黄
    (0.5647, 0.6902, 0.8157),  # 浅蓝
    (0.3804, 0.2216, 0.8863)  # 红
]
int_colors = []
for color in colors:
    int_color = [int(i * 255) for i in color]
    int_colors.append(int_color)

print(int_colors)

def visualize_and_save(image, pred_dict, save_path):
    """
    使用PIL可视化检测结果并保存，考虑归一化坐标。

    参数:
    - image: PIL图像对象。
    - pred_dict: 预测字典，包含'boxes', 'box_label'和'size'键。
    - save_path: 图像保存路径。
    """
    # 获取图像尺寸
    img_width, img_height = image.size

    draw = ImageDraw.Draw(image)

    # 尝试加载默认字体，否则不使用字体
    try:
        font = ImageFont.truetype('arial.ttf', 20)
    except IOError:
        font = None

    # 遍历检测框和标签
    for box, label in zip(pred_dict['boxes'], pred_dict['box_label']):
        # 将归一化坐标转换为实际像素坐标
        cx, cy, w, h = box
        x1 = (cx - w / 2) * img_width
        y1 = (cy - h / 2) * img_height
        x2 = (cx + w / 2) * img_width
        y2 = (cy + h / 2) * img_height

        # 绘制矩形框

        #draw.rectangle([x1, y1, x2, y2], outline=colors[label], width=2)
        draw.rectangle([x1, y1, x2, y2], outline= tuple(int_colors[label]), width=4)

        # 如果可用，绘制标签
        if font:
            draw.text((x1, y1-20), label_list[label-1], fill='white', font=font)
        else:
            draw.text((x1, y1), label, fill='white')

    # 保存图像
    image.save(save_path)


# 示例使用代码，这里假设'image'已经是一个PIL图像对象
# visualize_and_save(image, pred_dict, 'path_to_save_image.jpg')


def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

model_config_path = "config.py" # change the path of the model config file
model_checkpoint_path = "model.pth" # change the path of the model checkpoint

# See our Model Zoo section in README.md for more details about our pretrained models.
args = SLConfig.fromfile(model_config_path)
args.device = 'cuda'
model, criterion, postprocessors = build_model_main(args)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
_ = model.eval()



# transform images
transform = T.Compose([
    T.RandomResize([800], max_size=1333),    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])




if __name__ == '__main__':
    img_dir = ''
    save_dir = ''


    label_list = ['person','car','train','rider','truck','mcycle','bicycle','bus']
  #  label_list = ['car',]

    #卡阈值：
    thershold = 0.2  # set a thershold

    img_list = os.listdir(img_dir)
    fliter_img_list = img_list

    for n,img_name in enumerate(fliter_img_list):
        print('已处理:',n)
        img_path = os.path.join(img_dir,img_name)
        image_ori = Image.open(img_path).convert("RGB") # load image=
        image, _ = transform(image_ori,None)

        save_img_path = os.path.join(save_dir,img_name)

        # predict images
        output= model.cuda()(image[None].cuda())



        #结果可视化
        output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]
        scores = output['scores']
        labels = output['labels']
        boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])



        select_mask = scores > thershold

        box_label = [int(item) for item in labels[select_mask]]
        pred_dict = {
            'boxes': boxes[select_mask],
            'size': torch.Tensor([image.shape[1], image.shape[2]]),
            'box_label': box_label
        }


        visualize_and_save(image_ori, pred_dict, save_img_path)
