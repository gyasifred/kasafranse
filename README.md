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
<pre>
```python
from kasafranse.translation import OpusDirectTranslate

translator = OpusDirectTranslate(opus_model="opus-mt-twi-en")
translated_text = translator.translate("Example Twi sentence.")
print(translated_text)
```
</pre>
### Pivot Translation
<pre>
```
from kasafranse.translation import OpusPivotTranslate

translator = OpusPivotTranslate(opus_model_1="opus-mt-twi-en", opus_model_2="opus-mt-en-fr")
translated_text = translator.translate("Example Twi sentence.")
print(translated_text)
```
</pre>

## Contributions
1 - Fork repository.
1 - Create branch for changes.
3 - Implement and test.
4 - Create pull request to main branch.

## License
This project is licensed under the MIT License. See LICENSE for details.

Contact
For questions and collaboration, email **gyasifred@gmail.com**.
