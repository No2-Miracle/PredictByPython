from PIL import Image
from torchutils import *
import torch


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

classes_names = ['正常眼', '斜视眼']  # todo 类名
img_size = 224  # todo 图片大小
model_name = "efficientnet_b3a"  # todo 模型名称
num_classes = len(classes_names)  # todo 类别数目


def predict(image_path):
    data_transforms = get_torch_transforms(img_size=img_size)
    valid_transforms = data_transforms['val']
    # 加载网络
    model = myModel(model_name=model_name, out_features=num_classes, pretrained=False)
    weights = torch.load("model/model.pth")
    model.load_state_dict(weights)
    model.eval()
    model.to(device)

    # 读取图片
    img = Image.open(image_path)
    img = valid_transforms(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    output = model(img)
    label_id = torch.argmax(output).item()
    predict_name = classes_names[label_id]
    return predict_name
