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
    '''
        Args:
        train1:
        train2:
        val1:
        val2:
        lang1:
        lang2:
        example
    '''

    def __init__(self, train1, train2, val1, val2, lang1, lang2):
        self.train1 = train1
        self.train2 = train2
        self.val1 = val1
        self.val2 = val2
        self.lang1 = lang1
        self.lang2 = lang2

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

    def __init__(self, fine_turned_model, file, to_console=False, output="translate.txt"):
        self.model_name = fine_turned_model

    def translate(self, file, to_console=False, dir=None, output="translate.txt"):
        tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        model = MarianMTModel.from_pretrained(self.model_name)

        if to_console == True:
            # Open test file and read lines
            f = open(file, "r")
            src_text = f.readlines()
            f.close()
            for i in src_text:
                translated = model.generate(
                    **tokenizer(i, return_tensors="pt", padding=True))
                translated = [tokenizer.decode(
                    t, skip_special_tokens=True) for t in translated]
                translated = str(translated)[1:-1][1:-1]
                print(translated)

        else:
            # Open test file and read lines
            f = open(self.file, "r")
            src_text = f.readlines()
            f.close()
            lines = []
            for i in src_text:
                translated = model.generate(
                    **tokenizer(i, return_tensors="pt", padding=True))
                translated = [tokenizer.decode(
                    t, skip_special_tokens=True) for t in translated]
                lines.append(str(translated)[1:-1][1:-1])
            return preprocessor.writeTotxt(f'{dir}/{output}', lines)


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


class fineturnedsacrebleu():
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


class Pivot:
    def __init__(self, translator_1, translator_2):
        self.translator_1 = translator_1
        self.translator_2 = translator_2

    def translate(self, file, to_console=False, dir=None, output="translate.txt"):
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
                translated = model_1.generate(
                    **tokenizer_1(i, return_tensors="pt", padding=True))
                translated = [tokenizer_1.decode(
                    t, skip_special_tokens=True) for t in translated]
                translated = str(translated)[1:-1][1:-1]
                print(f'Pivot Translation: {translated}')
                translated = model_2.generate(
                    **tokenizer_2(translated, return_tensors="pt", padding=True))
                translated = [tokenizer_2.decode(
                    t, skip_special_tokens=True) for t in translated]
                translated = str(translated)[1:-1][1:-1]
                print(translated)

        else:
            # Open test file and read lines
            f = open(self.file, "r")
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
            return preprocessor.writeTotxt(f'{dir}/{output}', lines)
