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


class Translate:
    '''Translate from source language to target language
    Args:
    param opus_mt_transformer: Path to the pre-trained OPUS-MT
    param file: Path to the document to be translated
    param output: Specify the ouput directory for the final ouput
    param to_console: specify if you want the output printed to the console
    '''

    def __init__(self, opus_mt_transformer):
        self.model_name = opus_mt_transformer

    def translate(self, file, to_console=False, output="translate.txt"):
        tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        model = MarianMTModel.from_pretrained(self.model_name)

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


class OpusPivot:
    '''Translate with a cascading of two OPUS-MT model 
    Args:
    param translator_1: Provide the path to the first OPUS-MT model
    param translator_2: Provide the path to the second OPUS-MT model
    param file: Path to the document to be translated
    param output: Specify the ouput directory for the final ouput
    param to_console: specify if you want the output printed to the console
    '''

    def __init__(self, translator_1, translator_2):
        self.translator_1 = translator_1
        self.translator_2 = translator_2

    def translate(self, file, to_console=False, output="translate.txt"):
        tokenizer_1 = MarianTokenizer.from_pretrained(self.translator_1)
        model_1 = MarianMTModel.from_pretrained(self.translator_1)
        tokenizer_2 = MarianTokenizer.from_pretrained(self.translator_2)
        model_2 = MarianMTModel.from_pretrained(self.translator_2)

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


class Bleu():
    def __init__(self, translator, tokenizer):
        self.translator = translator
        self.tokenizer = tokenizer

    def get_bleuscore(self, testfile, referencefile, smothingfunction=None):
        if type(testfile) == str and type(referencefile) == str:
            # Open test file and read lines
            f = open(testfile, "r")
            hypothesis = f.readlines()
            f.close()
            # open refernce file and read lines
            f = open(referencefile, "r")
            reference = f.readlines()
            f.close()
        elif type(testfile) == list and type(referencefile) == list:
            hypothesis = testfile
            reference = referencefile
        else:
            print(f'File must be txt or python list')

        # check the length of our input sentence
        length = len(hypothesis)
        bleu_total = 0
        weights = (0.58, 0, 0, 0)
        for i in range(length):
            hypothesis[i] = hypothesis[i]
            reference[i] = reference[i]
            groundtruth = reference[i].lower().replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")\
                .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")\
                .replace(" ) ", ") ").replace(" , ", ", ").split()
            groundtruth = [groundtruth]

            translated_text = self.translator.generate(
                **self.tokenizer(hypothesis[i], return_tensors="pt", padding=True))
            translated = [self.tokenizer.decode(
                t, skip_special_tokens=True) for t in translated_text]
            candidate = str(translated)[1:-1][1:-1].replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")\
                .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")\
                .replace(" ) ", ") ").replace(" , ", ", ")
            candidate = candidate.lower().split()
            bleu = sentence_bleu(
                groundtruth, candidate, weights, smoothing_function=smothingfunction,
                auto_reweigh=True)
            bleu_total += bleu

        return f'BLEU SCORE: {bleu_total/length:.2f}'


class sacrebleu():
    '''Estimate the SacreBLEU score
    Args:
    translator: Path to the Opus-mt transformer
    tokenizer: Path to the opus-mt transformer tokenizer
    '''

    def __init__(self, translator, tokenizer):
        self.translator = translator
        self.tokenizer = tokenizer

    def get_bleuscore(self, testfile, referencefile):

        # Open test file and read translate
        preds = []
        with open(testfile) as pred:
            for line in pred:
                line = line.strip()
                line = self.translator.generate(
                    **self.tokenizer(line, return_tensors="pt", padding=True))
                translated = [self.tokenizer.decode(
                    t, skip_special_tokens=True) for t in line]
                candidate = str(translated)[1:-1][1:-1].replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")\
                    .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")\
                    .replace(" ) ", ") ").replace(" , ", ", ")
                preds.append(candidate.lower())

        refs = []
        with open(referencefile) as test:
            for line in test:
                line = line.strip().replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")\
                    .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")\
                    .replace(" ) ", ") ").replace(" , ", ", ")
                refs.append(line.lower())
        refs = [refs]
        bleu = sacrebleu.corpus_bleu(preds, refs, smooth_method="add-k",
                                     force=False,
                                     lowercase=True,
                                     tokenize="intl",
                                     use_effective_order=True)
        return f'BLEU SCORE: {bleu.score:.2f}'
