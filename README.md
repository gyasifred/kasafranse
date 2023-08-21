# Kasafranse: Twi to French Translation Library

[![GitHub Stars](https://img.shields.io/github/stars/gyasifred/kasafranse)](https://github.com/gyasifred/kasafranse/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/gyasifred/kasafranse)](https://github.com/gyasifred/kasafranse/network/members)
[![License](https://img.shields.io/github/license/gyasifred/kasafranse)](https://github.com/gyasifred/kasafranse/blob/main/LICENSE)

Welcome to Kasafranse, a Python library for translating Twi text to French. 

## Features

- Direct Translation: Translate Twi to French using pre-trained OPUS-MT model.
- Pivot Translation: Cascading translation using two OPUS-MT models.
- Text Preprocessing: Normalize text by removing accents, handling punctuation.
- Integration: Hugging Face Transformers and Google Cloud Translation API.

## Installation

```bash
pip install git+https://github.com/gyasifred/kasafranse
```
## Usage
### Direct Translation
```
from kasafranse.translation import OpusDirectTranslate

translator = OpusDirectTranslate(opus_model="opus-mt-twi-en")
translated_text = translator.translate("Example Twi sentence.")
print(translated_text)
```
