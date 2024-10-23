import unittest

import torch
from torchvision import models
from torch import nn

class MaskFusion(nn.Module):
    """
    掩码表示学习网络，用于从掩码中提取特征。
    """
    def __init__(self, input_dim=224*224, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 512, 256]
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.relu = nn.ReLU()

    def forward(self, mask):
        mask_flat = mask.view(mask.size(0), -1)  # Flatten mask
        x = self.relu(self.fc1(mask_flat))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Output 256-dimensional vector
        return x


class MultiHead(nn.Module):
    def __init__(self, num_classes_type=18, num_classes_state=9):
        super().__init__()

        # 使用预训练的DenseNet121作为特征提取器
        self.backbone = models.densenet121(pretrained=True).features

        # 掩码表示学习网络
        self.mask_representation_net = MaskFusion()

        # 分类任务1：食品类型分类头
        self.type_fc1 = nn.Linear(1024 * 7 * 7, 256)
        self.type_fc2 = nn.Linear(256, 128)
        self.type_fc3 = nn.Linear(128, num_classes_type)

        # 分类任务2：食品状态分类头
        self.state_fc1 = nn.Linear(1024 * 7 * 7, 256)
        self.state_fc2 = nn.Linear(256, 128)
        self.state_fc3 = nn.Linear(128, num_classes_state)

        # 通用ReLU层
        self.relu = nn.ReLU()

    def forward(self, image, mask):
        # 图像特征提取
        features = self.backbone(image)  # 输出的尺寸是 (batch_size, 1024, 7, 7)
        features_flat = features.view(features.size(0), -1)  # Flatten features (batch_size, 1024*7*7)

        # 掩码特征提取
        mask_features = self.mask_representation_net(mask)  # 输出 256 维特征

        # 食品类型分类分支
        type_out = self.relu(self.type_fc1(features_flat))
        type_out = type_out + mask_features  # 与掩码特征相加
        type_out = self.relu(self.type_fc2(type_out))
        type_out = self.type_fc3(type_out)  # 最后输出食品类型的分类

        # 食品状态分类分支
        state_out = self.relu(self.state_fc1(features_flat))
        state_out = state_out + mask_features  # 与掩码特征相加
        state_out = self.relu(self.state_fc2(state_out))
        state_out = self.state_fc3(state_out)  # 最后输出食品状态的分类

        # Softmax 用于输出概率分布
        self.softmax = nn.Softmax(dim=1)

        return self.softmax(type_out), self.softmax(state_out)



if __name__ == '__main__':
    outputs_type = torch.randn(8, 18)  # 假设batch_size = 8, 18个类别的输出
    outputs_state = torch.randn(8, 9)  # 假设batch_size = 8, 9个类别的输出
    labels_type = torch.randint(0, 18, (8,))  # 随机生成的真实标签（类型）
    labels_state = torch.randint(0, 9, (8,))  # 随机生成的真实标签（状态）

    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs_type, labels_type) + criterion(outputs_state, labels_state)
    print(loss.item())