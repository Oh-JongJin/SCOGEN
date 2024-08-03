# SCOGEN

An innovative project that uses generative AI to create musical score images based on user input and musical characteristics.



## Project Overview

The SCOGEN aims to create a deep learning model capable of generating sheet music images. By understanding musical elements such as key signatures, time signatures, note placements, and overall musical feel, our AI can produce unique and coherent musical scores tailored to user preferences.



## Key Features

- Generate musical score images from user-defined parameters 

- Support various musical styles, genres, and instruments 
- Understand and implement musical theory in generated scores
- Produce high-quality, printable sheet music images



## Required Libraries

- **Python**: 3.10.6
- **CUDA**: 11.8
- **PyTorch**: 2.2.0

- **Music21**: 9.1.0
- **Pretty_midi**: 0.2.10
- **Librosa**: 0.10.2.post1
- **Pillow (PIL)**: 9.5.0
- **NumPy**: 1.23.5
- **Matplotlib**: 3.8.3



## Installation

1. Clone this repository:
   
```bash
git clone https://github.com/Oh-JongJin/SCOGEN.git
```

2. Install the required libraries:

  ```bash
  pip install -r requirements.txt
  ```



## Usage

(...)



## Model Training

1. 데이터셋 구축
   - 다양한 장르, 스타일, 악기의 악보 이미지 수집
   - 각 악보에 대한 메타 데이터(장조/단조, 템포, 장르 등)를 함께 저장
2. 모델 학습
   - 수집한 악보 이미지와 메타 데이터를 사용하여 모델 학습
   - 모델이 악보의 구조, 음표 배치, 음악 이론 등을 학습하도록 함
3. UI
   - 사용자가 원하는 음악의 특성(장르, 템포, 조성 등)을 입력할 수 있는 인터페이스 작성



#### TODO

```
1. 악보의 구조적 일관성 유지
2. 음악 이론 규칙 준수
3. 사용자 의도와 일치하는 악보 생성
4. 고품질의 이미지 생성
```



## License

This project is distributed under the MIT License. 

See `LICENSE` file for more information. 

