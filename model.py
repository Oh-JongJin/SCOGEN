import torch
import torch.nn as nn


class ScoreGenerator(nn.Module):
    def __init__(self):
        super(ScoreGenerator, self).__init__()
        self.model = nn.Sequential(
            # 입력: 잠재 벡터 (100) + 스타일 (10) + 템포 (5) + 키 (5) = 120
            nn.Linear(120, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 256 * 256),  # 256x256 이미지 생성
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(-1, 1, 256, 256)  # (batch_size, channels, height, width)
        return img


class ScoreDiscriminator(nn.Module):
    def __init__(self):
        super(ScoreDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(256 * 256, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(-1, 256 * 256)
        validity = self.model(img_flat)
        return validity
