import nltk
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import sacrebleu


class BLEU():
    '''This class help estimate the BLEU Score '''

    def __init__(self):
        pass

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

            candidate = hypothesis[i].lower().replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")\
                .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")\
                .replace(" ) ", ") ").replace(" , ", ", ").split()
            # print("Translated Text: ", candidate)
            # print("Ground Truth: ", groundtruth)
            bleu = np.array(sentence_bleu(
                groundtruth, candidate, weights, auto_reweigh=True, smoothing_function=smothingfunction))
            bleu_total += bleu

        return f'2-GRAMS: {bleu_total[0]/length:.2f}', f'3-GRAMS: {bleu_total[1]/length:.2f}', f'4-GRAMS: {bleu_total[2]/length:.2f}'


class BleuScore():
    '''Estimate BLEU score as instantiated by Azunre et al. (2021)'''
    def __init__(self):
        pass

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
            print(f'File must be text file or python list')

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

            candidate = hypothesis[i].lower().replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")\
                .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")\
                .replace(" ) ", ") ").replace(" , ", ", ").split()
            # print("Translated Text: ",candidate)
            # print("Ground Truth: ",groundtruth)
            bleu = sentence_bleu(
                groundtruth, candidate, weights, smoothing_function=smothingfunction,
                auto_reweigh=True)
            bleu_total += bleu

        return f'BLEU SCORE: {bleu_total/length:.2f}'


class Sacrebleu():
    '''Estimate SacreBleu score'''
    def __init__(self) -> None:
        pass

    def get_bleuscore(self, testfile, referencefile):

        # Open test file and read translate
        preds = []
        with open(testfile) as pred:
            for line in pred:
                line = line.strip().replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")\
                    .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")\
                    .replace(" ) ", ") ").replace(" , ", ", ")
                preds.append(line.lower())

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
