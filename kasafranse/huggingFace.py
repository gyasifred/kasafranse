from kasafranse.preprocessing import Preprocessing
from transformers import MarianMTModel, MarianTokenizer
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import pandas as pd
from datasets import Dataset
import sacrebleu
from datasets import load_dataset

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
    param opus_mt_transformer: Path to the pre-trained OPUS-MT
    param file: Path to the document to be translated
    param output: Specify the ouput directory for the final ouput
    param to_console: specify if you want the output printed to the console
    '''

    def __init__(self):
        pass

    def translate(self, opus_model, file, to_console=False, output="translate.txt"):
        tokenizer = MarianTokenizer.from_pretrained(opus_model)
        model = MarianMTModel.from_pretrained(opus_model)

        if to_console == True:
            # Open test file and read lines
            f = open(file, "r")
            src_text = f.readlines()
            f.close()
            for i in src_text:
                print(f'Source: {i}')
                translated = model.generate(
                    **tokenizer(i, return_tensors="pt", padding=True))
                translated = [tokenizer.decode(
                    t, skip_special_tokens=True) for t in translated]
                translated = str(translated)[1:-1][1:-1]
                print(f'Target: {translated}')
                print()

        else:
            # Open test file and read lines
            f = open(file, "r")
            src_text = f.readlines()
            f.close()
            lines = []
            for i in src_text:
                translated = model.generate(
                    **tokenizer(i, return_tensors="pt", padding=True))
                translated = [tokenizer.decode(
                    t, skip_special_tokens=True) for t in translated]
                lines.append(str(translated)[1:-1][1:-1])
            return preprocessor.writeTotxt(output, lines)


class OpusPivotTranslate:
    '''Translate with a cascading of two OPUS-MT model 
    Args:
    param translator_1: Provide the path to the first OPUS-MT model
    param translator_2: Provide the path to the second OPUS-MT model
    param file: Path to the document to be translated
    param output: Specify the ouput directory for the final ouput
    param to_console: specify if you want the output printed to the console
    '''

    def __init__(self):
        pass

    def translate(self, opus_model_1, opus_model_2,file, to_console=False, output="translate.txt"):
        tokenizer_1 = MarianTokenizer.from_pretrained(opus_model_1)
        model_1 = MarianMTModel.from_pretrained(opus_model_1)
        tokenizer_2 = MarianTokenizer.from_pretrained(opus_model_2)
        model_2 = MarianMTModel.from_pretrained(opus_model_2)

        if to_console == True:
            # Open test file and read lines
            f = open(file, "r")
            src_text = f.readlines()
            f.close()
            for i in src_text:
                print(f'Source: {i}')
                translated = model_1.generate(
                    **tokenizer_1(i, return_tensors="pt", padding=True))
                translated = [tokenizer_1.decode(
                    t, skip_special_tokens=True) for t in translated]
                translated = str(translated)[1:-1][1:-1]
                print(f'Pivot: {translated}')
                translated = model_2.generate(
                    **tokenizer_2(translated, return_tensors="pt", padding=True))
                translated = [tokenizer_2.decode(
                    t, skip_special_tokens=True) for t in translated]
                translated = str(translated)[1:-1][1:-1]
                print(f'Target: {translated}')
                print()

        else:
            # Open test file and read lines
            f = open(file, "r")
            src_text = f.readlines()
            f.close()
            lines = []
            for i in src_text:
                translated = model_1.generate(
                    **tokenizer_1(i, return_tensors="pt", padding=True))
                translated = [tokenizer_1.decode(
                    t, skip_special_tokens=True) for t in translated]
                translated = str(translated)[1:-1][1:-1]

                translated = model_2.generate(
                    **tokenizer_2(translated, return_tensors="pt", padding=True))
                translated = [tokenizer_2.decode(
                    t, skip_special_tokens=True) for t in translated]
                lines.append(str(translated)[1:-1][1:-1])
            return preprocessor.writeTotxt(output, lines)
