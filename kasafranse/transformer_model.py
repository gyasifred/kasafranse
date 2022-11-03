from kasafranse.preprocessing import Preprocessing
from kasafranse.transformer_layers import Encoder, Decoder
import tensorflow as tf
import tensorflow_text
import nltk
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import sacrebleu
# instatiate Preprocessing
preprocessor = Preprocessing()

# Build full Transformer


# Build full Transformer
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        self.decoder = Decoder(
            num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        # (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(inp, training, enc_padding_mask)
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        # (batch_size, tar_seq_len, target_vocab_size)
        final_output = self.final_layer(dec_output)
        return final_output, attention_weights


class Translate:
    def __init__(self, translator_path, file, output_name="translate.txt", to_console=False):
        self.translator = translator_path
        self.file = file
        self.ouput = output_name
        self.to_console = to_console

    def translate(self):
        evaluate = tf.saved_model.load(self.translator)
        if self.to_console == True:
            with open(self.file) as txt_file:
                for line in txt_file:
                    translated_text = evaluate(
                        line).numpy().decode("utf-8")
                    translated_text = translated_text.replace(" ' ", "'").replace(" .", ".")\
                        .replace(" ?", "?").replace(" !", "!")\
                        .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ")\
                        .replace(" ( ", " (").replace(" ) ", ") ").replace(" , ", ", ")
                    print(translated_text)
        else:
            lines = []
            with open(self.file) as txt_file:
                for line in txt_file:
                    translated_text = evaluate(
                        line).numpy().decode("utf-8")
                    translated_text = translated_text.replace(" ' ", "'").replace(" .", ".")\
                        .replace(" ?", "?").replace(" !", "!")\
                        .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ")\
                        .replace(" ( ", " (").replace(" ) ", ") ").replace(" , ", ", ")
                    lines.append(translated_text)
            return preprocessor.writeTotxt(self.ouput, lines)


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
                line = translator(line).numpy().decode("utf-8")
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
        bleu = sacrebleu.corpus_bleu(preds, refs, smooth_method="add-k",
                                     force=False,
                                     lowercase=True,
                                     tokenize="intl",
                                     use_effective_order=True)
        return f'BLEU SCORE: {bleu.score}'
