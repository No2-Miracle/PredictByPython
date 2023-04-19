import timm
import torch.nn as nn
from torchvision import transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_torch_transforms(img_size=224):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomRotation((-5, 5)),
            transforms.RandomAutocontrast(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms


class myModel(nn.Module):
    def __init__(self, model_name='efficientnet_b3a', out_features=2,
                 pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)  # 从预训练的库中加载模型
        # classifier
        if model_name[:3] == "res":
            n_features = self.model.fc.in_features  # 修改全连接层数目
            self.model.fc = nn.Linear(n_features, out_features)  # 修改为本任务对应的类别数目
        elif model_name[:3] == "vit":
            n_features = self.model.head.in_features  # 修改全连接层数目
            self.model.head = nn.Linear(n_features, out_features)  # 修改为本任务对应的类别数目
        else:
            n_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(n_features, out_features)

    def forward(self, x):  # 前向传播
        x = self.model(x)
        return x
