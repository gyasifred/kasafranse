from kasafranse.preprocessing import Preprocessing
from transformers import MarianMTModel, MarianTokenizer
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import pandas as pd
from datasets import Dataset
from datasets import load_dataset

preprocessor = Preprocessing()


class BuildDataset:
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
        val = pd.DataFrame(train)

        train_dataset = Dataset.from_pandas(train)

        val_dataset = Dataset.from_pandas(val)

        return train_dataset, val_dataset


class Translate:
    def __init__(self, fine_turned_model, file, to_console=False, output="translate.txt"):
        self.model_name = fine_turned_model
        self.file = file
        self.to_console = to_console
        self.output = output

    def translate(self):
        tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        model = MarianMTModel.from_pretrained(self.model_name)

        if self.to_console == True:
            # Open test file and read lines
            f = open(self.file, "r")
            src_text = f.readlines()
            f.close()
            for i in src_text:
                translated = model.generate(
                    **tokenizer(i, return_tensors="pt", padding=True))
                translated = [tokenizer.decode(
                    t, skip_special_tokens=True) for t in translated]
                print(str(translated)[1:-1])

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
                lines.append(str(translated)[1:-1])
            return preprocessor.writeTotxt(self.output, lines)


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
            candidate = str(translated)[1:-1]
            # print("Translated Text: ", candidate)
            # print("Ground Truth: ", reference[i])
            candidate = candidate.replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")\
                .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")\
                .replace(" ) ", ") ").replace(" , ", ", ").split()
            bleu = sentence_bleu(
                groundtruth, candidate, weights, smoothing_function=smothingfunction, auto_reweigh=True)
            bleu_total += bleu

        return f'BLEU SCORE: {bleu_total/length}'
