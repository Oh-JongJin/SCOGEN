import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import argparse
from model import ScoreGenerator
from PIL import Image


def generate_score(style, tempo, key):
    """주어진 스타일, 템포, 키를 바탕으로 악보 이미지 생성

    사전에 훈련된 Score Generator 모델 로드

    Parameter
    ---------
    style : str
        생성할 악보의 음악 스타일 (예: "classical", "jazz")
    tempo : str
        생성할 악보의 템포 (예: "slow", "moderate", "fast")
    key : str
        생성할 악보의 조성 (예: "C major", "A minor")

    Returns
    -------
    None
        생성된 악보 이미지를 파일로 저장
    """

    # 모델 로드
    generator = ScoreGenerator()
    generator.load_state_dict(torch.load('generator.pth'))
    generator.eval()

    # 입력 벡터 생성
    noise = torch.randn(1, 100)
    style_vector = encode_style(style)
    tempo_vector = encode_tempo(tempo)
    key_vector = encode_key(key)

    input_vector = torch.cat([noise, style_vector, tempo_vector, key_vector], dim=1)

    # 악보 생성
    with torch.no_grad():
        generated_score = generator(input_vector)

    # 이미지로 변환 및 저장
    img = transform_to_image(generated_score)
    img.save('generated_score.png')
    print("악보가 생성되었습니다: generated_score.png")


def encode_style(style: str) -> torch.Tensor:
    """음악 스타일을 숫자 벡터로 변환
    
    이를 통해 텍스트 정보를 모델이 이해할 수 있는 형태로 변환

    Parameter
    ---------
    style : str
        인코딩할 음악 스타일

    Returns
    -------
    torch.Tensor
        인코딩된 스타일 벡터
    """

    # 스타일을 벡터로 인코딩
    styles = ['classical', 'jazz', 'rock', 'pop', 'electronic']
    if style not in styles:
        raise ValueError(f"Unsupported style: {style}")

    index = styles.index(style)
    encoded = F.one_hot(torch.tensor(index), num_classes=len(styles))

    return encoded.float().unsqueeze(0)  # (1, num_styles)


def encode_tempo(tempo):
    """템포를 숫자로 변환한 후 정규화합니다.

    Parameter
    ---------
    tempo : str
        인코딩할 템포 (예: "slow", "moderate", "fast")

    Returns
    -------
    torch.Tensor
        인코딩된 템포 벡터
    """
    tempo_map = {'slow': 0, 'moderate': 1, 'fast': 2}
    if tempo not in tempo_map:
        raise ValueError(f"Unsupported tempo: {tempo}")

    # 템포를 벡터로 인코딩
    value = tempo_map[tempo]
    encoded = torch.tensor(value).float() / 2  # Normalize to [0, 1]

    return encoded.unsqueeze(0).unsqueeze(0)  # (1, 1)


def encode_key(key):
    """
    조성을 숫자로 변환한 후 정규화.

    Parameter
    ---------
    key : str
        인코딩할 조성 (예: "C major", "A minor")

    Returns
    -------
    torch.Tensor
        인코딩된 조성 벡터
    """

    major_keys = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'F', 'Bb', 'Eb', 'Ab']
    minor_keys = [key.lower() for key in major_keys]

    key_parts = key.split()
    if len(key_parts) != 2 or key_parts[1] not in ['major', 'minor']:
        raise ValueError(f"Invalid key format: {key}")

    root, mode = key_parts
    if mode == 'major':
        value = major_keys.index(root)
    else:
        value = minor_keys.index(root.lower()) + 12

    # 조성을 벡터로 인코딩
    encoded = torch.tensor(value).float() / 23  # Normalize to [0, 1]

    return encoded.unsqueeze(0).unsqueeze(0)  # (1, 1)


def transform_to_image(tensor):
    """
    모델이 출력한 텐서를 PIL 이미지 객체로 변환

    Parameter
    ---------
    tensor : torch.Tensor
        변환할 텐서. Shape: (1, 1, height, width)

    Returns
    -------
    PIL.Image.Image
        변환된 PIL 이미지 객체
    """

    # 텐서를 [0, 1] 범위로 정규화
    tensor = (tensor + 1) / 2.0
    tensor = tensor.clamp(0, 1)

    # 텐서를 PIL 이미지로 변환
    transform = transforms.ToPILImage()
    image = transform(tensor.squeeze(0))

    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a music score")
    parser.add_argument('--style', type=str, required=True, help='Music style')
    parser.add_argument('--tempo', type=str, required=True, help='Tempo')
    parser.add_argument('--key', type=str, required=True, help='Musical key')
    args = parser.parse_args()

    generate_score(args.style, args.tempo, args.key)
