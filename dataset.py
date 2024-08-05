import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ScoreDataset(Dataset):
    """지정된 디렉토리에서 이미지 파일을 로드

    각 이미지에 대해 변환(transform)을 적용하고, 관련 메타데이터를 반환
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.png') or f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('L')  # Convert to grayscale

        if self.transform:
            image = self.transform(image)

        # 여기에서 메타데이터(스타일, 템포, 키 등)를 로드
        # 예를 들어, 이미지 파일명에서 메타데이터를 추출하거나 별도의 메타데이터 파일을 사용할 수 있음
        style = torch.zeros(10)  # 예시: 10차원 원-핫 인코딩
        tempo = torch.zeros(5)   # 예시: 5차원 벡터
        key = torch.zeros(5)     # 예시: 5차원 벡터

        return image, (style, tempo, key)
