import nltk
import numpy as np
import tensorflow as tf
import tensorflow_text
from nltk.translate.bleu_score import sentence_bleu
import sacrebleu


class TransformerBleu():
    def __init__(self, translator):
        self.translator = translator

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
        bleu_total = np.array([0., 0., 0.])
        weights = [(1./2, 1./2), (1./3, 1./3, 1./3),
                   (1./4, 1./4, 1./4, 1./4)]
        for i in range(length):
            hypothesis[i] = hypothesis[i]
            reference[i] = reference[i]
            groundtruth = reference[i].lower().replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")\
                .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")\
                .replace(" ) ", ") ").replace(" , ", ", ").split()
            groundtruth = [groundtruth]
            translated_text = self.translator(
                hypothesis[i]).numpy().decode("utf-8")
            print("Translated Text: ", translated_text)
            print("Ground Truth: ", reference[i])
            candidate = translated_text.lower().replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")\
                .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")\
                .replace(" ) ", ") ").replace(" , ", ", ").split()
            bleu = np.array(sentence_bleu(
                groundtruth, candidate, weights, smoothing_function=smothingfunction))
            bleu_total += bleu

        return f'2-GRAMS: {bleu_total[0]/length}', f'3-GRAMS: {bleu_total[1]/length}', f'4-GRAMS: {bleu_total[2]/length}'


class Seq2seqBleu():
    def __init__(self, translator):
        self.translator = translator

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
        bleu_total = np.array([0., 0., 0.])
        weights = [(1./2, 1./2), (1./3, 1./3, 1./3),
                   (1./4, 1./4, 1./4, 1./4)]
        for i in range(length):
            hypothesis[i] = hypothesis[i]
            reference[i] = reference[i]
            groundtruth = reference[i].lower().replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")\
                .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")\
                .replace(" ) ", ") ").replace(" , ", ", ").split(" ")
            groundtruth = [groundtruth]
            input = tf.constant([hypothesis[1]])
            result = self.translator(input)
            translated_text = result['text'][0].numpy().decode("utf-8")

            #print("Translated Text: ", translated_text)
            #print("Ground Truth: ", reference[i])
            candidate = translated_text.lower().replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")\
                .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")\
                .replace(" ) ", ") ").replace(" , ", ", ").split()
            bleu = np.array(sentence_bleu(
                groundtruth, candidate, weights, smoothing_function=smothingfunction))
            bleu_total += bleu

        return f'2-GRAMS: {bleu_total[0]/length}', f'3-GRAMS: {bleu_total[1]/length}', f'4-GRAMS: {bleu_total[2]/length}'


class GoogleBleu():
    def __init__(self, translator):
        self.translator = translator

    def get_bleuscore(self, testfile, referencefile, src_lang, dest_lang, smothingfunction=None):
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
        bleu_total = np.array([0., 0., 0.])
        weights = [(1./2, 1./2), (1./3, 1./3, 1./3),
                   (1./4, 1./4, 1./4, 1./4)]
        for i in range(length):
            hypothesis[i] = hypothesis[i]
            reference[i] = reference[i]
            groundtruth = reference[i].lower().replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")\
                .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")\
                .replace(" ) ", ") ").replace(" , ", ", ").split()
            groundtruth = [groundtruth]
            translated_text = self.translator.evaluate(
                hypothesis[i], src_key=src_lang, dest_key=dest_lang)
            print("Translated Text: ", translated_text)
            print("Ground Truth: ", reference[i])
            candidate = translated_text.lower().replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")\
                .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")\
                .replace(" ) ", ") ").replace(" , ", ", ").split()
            bleu = np.array(sentence_bleu(
                groundtruth, candidate, weights, smoothing_function=smothingfunction))
            bleu_total += bleu

        return f'2-GRAMS: {bleu_total[0]/length}', f'3-GRAMS: {bleu_total[1]/length}', f'4-GRAMS: {bleu_total[2]/length}'


class BleuScore():
    def __init__(self, translator):
        self.translator = translator

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
            translated_text = self.translator(
                hypothesis[i]).numpy().decode("utf-8")
            # print("Translated Text: ", translated_text)
            # print("Ground Truth: ", reference[i])
            candidate = translated_text.lower().replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")\
                .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")\
                .replace(" ) ", ") ").replace(" , ", ", ").split()
            bleu = sentence_bleu(
                groundtruth, candidate, weights, smoothing_function=smothingfunction, auto_reweigh=True)
            bleu_total += bleu

        return f'BLEU SCORE: {bleu_total/length}'


class GoogleAPIBleu():
    def __init__(self, translator):
        self.translator = translator

    def get_bleuscore(self, testfile, referencefile, src_lang, dest_lang, smothingfunction=None):
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
            translated_text = self.translator.evaluate(
                hypothesis[i], src_key=src_lang, dest_key=dest_lang)
            # print("Translated Text: ", translated_text)
            # print("Ground Truth: ", reference[i])
            candidate = translated_text.lower().replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")\
                .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")\
                .replace(" ) ", ") ").replace(" , ", ", ").split()
            bleu = sentence_bleu(
                groundtruth, candidate, weights, smoothing_function=smothingfunction, auto_reweigh=True)
            bleu_total += bleu

        return f'BLEU SCORE: {bleu_total/length}'


class TurnedModelBleu():
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
            candidate = str(translated)[1:-1][1:-1]
            # print("Translated Text: ", candidate)
            # print("Ground Truth: ", reference[i])
            candidate = candidate.lower().replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")\
                .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")\
                .replace(" ) ", ") ").replace(" , ", ", ").split()
            bleu = sentence_bleu(
                groundtruth, candidate, weights, smoothing_function=smothingfunction, auto_reweigh=True)
            bleu_total += bleu

        return f'BLEU SCORE: {bleu_total/length}'


class Seq2seqModelBleu():
    def __init__(self, translator):
        self.translator = translator

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
                .replace(" ) ", ") ").replace(" , ", ", ").split(" ")
            groundtruth = [groundtruth]
            input = tf.constant([hypothesis[1]])
            result = self.translator(input)
            translated_text = result['text'][0].numpy().decode("utf-8")

            #print("Translated Text: ", translated_text)
            #print("Ground Truth: ", reference[i])
            candidate = translated_text.lower().replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")\
                .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")\
                .replace(" ) ", ") ").replace(" , ", ", ").split()
            bleu = sentence_bleu(
                groundtruth, candidate, weights, smoothing_function=smothingfunction, auto_reweigh=True)
            bleu_total += bleu

        return f'BLEU SCORE: {bleu_total/length}'


class TransformerSacredBleu():
    def __init__(self, translator):
        self.translator = translator

    def get_bleuscore(self, testfile, referencefile):
        # Open test file and read translate
        translator = self.translator
        preds = []
        with open(testfile) as pred:
            for line in pred:
                line = line.strip()
                line =translator(line).numpy().decode("utf-8")
                candidate = line.lower().replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")\
                    .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")\
                    .replace(" ) ", ") ").replace(" , ", ", ")
                preds.append(candidate)

        refs = []
        with open(referencefile) as test:
            for line in test:
                line = line.strip()
                refs.append(line)

        refs = [refs]
        bleu = sacrebleu.corpus_bleu(preds, refs)
        return f'BLEU SCORE: {bleu.score}'


class GoogleAPISacredBleu():
    def __init__(self, translator):
        self.translator = translator

    def get_bleuscore(self, testfile, referencefile, src_lang, dest_lang):
        translator = self.translator
        # Open test file and read translate
        preds = []
        with open(testfile) as pred:
            for line in pred:
                line = line.strip()
                line = translator.evaluate(
                    line, src_key=src_lang, dest_key=dest_lang)
                candidate = line.lower().replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")\
                    .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")\
                    .replace(" ) ", ") ").replace(" , ", ", ")
                preds.append(candidate)

        refs = []
        with open(referencefile) as test:
            for line in test:
                line = line.strip()
                refs.append(line)

        refs = [refs]
        bleu = sacrebleu.corpus_bleu(preds, refs)
        return f'BLEU SCORE: {bleu.score}'
