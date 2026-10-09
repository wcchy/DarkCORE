import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# 类别名称映射
class_names = {0: 'Bicycle', 1: 'Boat', 2: 'Bottle', 3: 'Bus', 
               4: 'Car', 5: 'Cat', 6: 'Chair', 7: 'Cup',
               8: 'Dog', 9: 'Motorbike', 10: 'People', 11: 'Table'}

def load_images_and_labels(image_dir, label_dir, save_dir):
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # 遍历图片文件夹
    for image_file in os.listdir(image_dir):
        if image_file.endswith('.jpg') or image_file.endswith('.png') or image_file.endswith('.JPG') or image_file.endswith('.PNG') or image_file.endswith('.JPEG'):
            image_path = os.path.join(image_dir, image_file)
            label_path = os.path.join(label_dir, image_file.replace('.jpg', '.txt').replace('.png', '.txt')
            .replace('.JPG', '.txt').replace('.PNG', '.txt').replace('.JPEG', '.txt'))
            save_path = os.path.join(save_dir, image_file)
            
            # 加载图片
            image = Image.open(image_path)
            fig, ax = plt.subplots(1)
            ax.imshow(image)

            # 检查标签文件是否存在
            if os.path.exists(label_path):
                with open(label_path, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        class_id, x_center, y_center, width, height = map(float, line.split())
                        class_name = class_names.get(int(class_id), 'Unknown')  # 获取类别名称

                        # 转换为图像坐标系
                        x = (x_center - width / 2) * image.width
                        y = (y_center - height / 2) * image.height
                        w = width * image.width
                        h = height * image.height

                        # 创建矩形框并添加类别标签
                        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='g', facecolor='none')
                        ax.add_patch(rect)
                        ax.text(x, y, class_name, color='white', verticalalignment='top', bbox={'color': 'green', 'pad': 0})

            plt.axis('off')  # 不显示坐标尺
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)  # 保存图像
            plt.close()  # 关闭图像以释放内存

# 设置图片文件夹、标签文件夹和保存文件夹的路径
image_dir = '/home/ubuntu/datasets/exdark_yolo_dataset/val/images'
label_dir = '/home/ubuntu/datasets/exdark_yolo_dataset/val/labels'
save_dir = '/home/ubuntu/project/DIFF_mmdetection/visual/ground_truth'

load_images_and_labels(image_dir, label_dir, save_dir)
