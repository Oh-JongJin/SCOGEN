# SCOGEN
```
      _/_/_/        _/_/_/        _/_/         _/_/_/       _/_/_/_/       _/      _/         _/_/
   _/            _/            _/    _/     _/             _/             _/_/    _/         _/  _/
    _/_/        _/            _/    _/     _/  _/_/       _/_/_/         _/  _/  _/         _/    _/
       _/      _/            _/    _/     _/    _/       _/             _/    _/_/       _/_/      _/
_/_/_/          _/_/_/        _/_/         _/_/_/       _/_/_/_/       _/      _/     _/_/_/
                                                                                      _/_/
```
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
- **Tensorflow**
- **keras**
- **scikit-learn**
- **Music21**: 9.1.0
- **numPy**
- **Matplotlib**



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

To generate a music score using SCOGEN, follow these steps: 

1. Ensure you have installed all required dependencies as mentioned in the Installation section. 



2. Run the main script with your desired parameters:

```bash
python lstm_gen.py
```





## Model Training

1. Dataset Construction
   - Gather sheet music images across various genres, styles, and instruments.
   - Annotate each sheet music image with metadata such as key signature, tempo, genre, etc.
2. Model Selection
   - Utilize generative models like GANs (Generative Adversarial Networks) or VAEs (Variational Autoencoders) for image generation.
   - Consider using recent models like Stable Diffusion for text-to-image generation capabilities.
3. UI
   - Develop an interface where users can input desired music characteristics (e.g., genre, tempo, key signature).



#### TODO

1. **Maintaining Structural Consistency**: Ensure the generated sheet music maintains a consistent structure.
2. **Adhering to Music Theory**: Ensure compliance with music theory rules.
3. **Matching User Intent**: Generate sheet music that aligns with user specifications.
4. **High-Quality Image Generation**: Produce high-quality, visually appealing sheet music images.


