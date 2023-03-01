from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
from datasets import Dataset
from kasafranse.preprocessing import Preprocessing
import warnings
warnings.simplefilter('ignore')

preprocessor = Preprocessing()


class BuildDataset:

    ''' Build Hugging Face Dataset
    Args:
    param src_train: Path to the source language training data
    param targ_train: Path to the target language training data
    param src_eval: Path to the source language validation data
    param targ_eval: Path to the target language validation training data
    param src_lang_key: Key for the source language
    param targ_lang_key: key for  the target language
    '''

    def __init__(self, src_train, targ_train, src_eval, targ_eval, src_lang_key, targ_lang_key):
        self.train1 = src_train
        self.train2 = targ_train
        self.val1 = src_eval
        self.val2 = targ_eval
        self.lang1 = src_lang_key
        self.lang2 = targ_lang_key

    def build(self):
        translation = []
        for i, j in zip(self.train1, self.train2):
            translation.append({self.lang1: i, self.lang2: j})

        train = {"translation": translation}
        train = pd.DataFrame(train)

        translation = []
        for i, j in zip(self.val1, self.val2):
            translation.append({self.lang1: i, self.lang2: j})

        val = {"translation": translation}
        val = pd.DataFrame(val)

        train_dataset = Dataset.from_pandas(train)

        val_dataset = Dataset.from_pandas(val)

        return train_dataset, val_dataset


class OpusDirectTranslate:
    '''Translate from source language to target language
    Args:
    param opus_model: Path to the pre-trained OPUS-MT
    param text: The text to be translated

    '''

    def __init__(self, opus_model):
        self.opus_model = opus_model
        self.tokenizer = MarianTokenizer.from_pretrained(self.opus_model)
        self.model = MarianMTModel.from_pretrained(self.opus_model)

    def translate(self, text):
        translated = self.model.generate(
            **self.tokenizer(text.strip(), return_tensors="pt", padding=True))
        translated = [self.tokenizer.decode(
            t, skip_special_tokens=True) for t in translated]
        translated = str(translated)[1:-1][1:-1].replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")\
            .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")\
            .replace(" ) ", ") ").replace(" , ", ", ")
        return translated


class OpusPivotTranslate:
    '''Translate with a cascading of two OPUS-MT model 
    Args:
    param opus_model_1: Provide the path to the first OPUS-MT model
    param opus_model_1: Provide the path to the second OPUS-MT model
    param text: The text to be translated
    '''

    def __init__(self, opus_model_1, opus_model_2):
        self.mt_model_1 = opus_model_1
        self.mt_model_2 = opus_model_2
        self.tokenizer_1 = MarianTokenizer.from_pretrained(self.mt_model_1)
        self.model_1 = MarianMTModel.from_pretrained(self.mt_model_1)
        self.tokenizer_2 = MarianTokenizer.from_pretrained(self.mt_model_2)
        self.model_2 = MarianMTModel.from_pretrained(self.mt_model_2)

    def translate(self, text):

        translated = self.model_1.generate(
            **self.tokenizer_1(text, return_tensors="pt", padding=True))
        translated = [self.tokenizer_1.decode(
            t, skip_special_tokens=True) for t in translated]
        translated = str(translated)[1:-1][1:-1]
        translated = self.model_2.generate(
            **self.tokenizer_2(translated, return_tensors="pt", padding=True))
        translated = [self.tokenizer_2.decode(
            t, skip_special_tokens=True) for t in translated]
        translated = str(translated)[1:-1][1:-1].replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")\
            .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")\
            .replace(" ) ", ") ").replace(" , ", ", ")
        return translated
