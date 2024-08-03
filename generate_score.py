import torch
import argparse
from model import ScoreGenerator
from PIL import Image


def generate_score(style, tempo, key):
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


def encode_style(style):
    # 스타일을 벡터로 인코딩
    pass


def encode_tempo(tempo):
    # 템포를 벡터로 인코딩
    pass


def encode_key(key):
    # 조성을 벡터로 인코딩
    pass


def transform_to_image(tensor):
    # 텐서를 PIL 이미지로 변환
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a music score")
    parser.add_argument('--style', type=str, required=True, help='Music style')
    parser.add_argument('--tempo', type=str, required=True, help='Tempo')
    parser.add_argument('--key', type=str, required=True, help='Musical key')
    args = parser.parse_args()

    generate_score(args.style, args.tempo, args.key)