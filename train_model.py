import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from model import ScoreGenerator, ScoreDiscriminator
from dataset import ScoreDataset


def train():
    # 하이퍼파라미터 설정
    batch_size = 64
    num_epochs = 100
    lr = 0.0002

    # 데이터셋 및 데이터로더 설정
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = ScoreDataset('dataset path', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 모델 초기화
    generator = ScoreGenerator()
    discriminator = ScoreDiscriminator()

    # 손실 함수 및 최적화 함수 설정
    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=lr)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

    # 학습 루프
    for epoch in range(num_epochs):
        for i, (real_scores, _) in enumerate(dataloader):
            batch_size = real_scores.size(0)
            real_label = torch.ones(batch_size, 1)
            fake_label = torch.zeros(batch_size, 1)

            # Discriminator 학습
            d_optimizer.zero_grad()
            output = discriminator(real_scores)
            d_loss_real = criterion(output, real_label)

            noise = torch.randn(batch_size, 100)
            fake_scores = generator(noise)
            output = discriminator(fake_scores.detach())
            d_loss_fake = criterion(output, fake_label)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # Generator 학습
            g_optimizer.zero_grad()
            output = discriminator(fake_scores)
            g_loss = criterion(output, real_label)
            g_loss.backward()
            g_optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}')

    # 모델 저장
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')


if __name__ == "__main__":
    train()
