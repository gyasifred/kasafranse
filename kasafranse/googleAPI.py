import nltk
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import sacrebleu

# Use google API for bidirectional pivot translation of Twi and French
# pivot language = English
# import libraries
from googletrans import Translator, constants
# instantiate a translator object
# initiate translator object
translator = Translator()
# Add Akan to the language supported by this package
# Note the googletrans package has not  been updated to capture the new additions by google since May 2022
# from https://translate.google.com/?sl=en&tl=ak&op=translate , the key and value for Twi is 'ak':'akan'
constants.LANGUAGES['ak'] = 'akan'


class GooglePivot:
    '''Perfom Pivot Translation of source language to target language using Google Translate API
    Args:
    param sentences: Text to be translated
    param src_key: Source Language key as specified by the google translate API
    param dest_key: Target Language key as specified by the google translate API
    param pivot_key: PivotLanguage key as specified by the google translate API
    '''

    def __init__(self):
        pass

    def evaluate(self, sentences, src_key, dest_key, pivot_key="en"):
        eng_text = translator.translate(
            sentences, src=src_key, dest=pivot_key).text
        output = translator.translate(eng_text, dest=dest_key).text

        return output


class GoogleDirect:
    '''Perfom Direct Translation of source language to target language using Google Translate API
    Args:
    param sentences: Text to be translated
    param src_key: Source Language key as specified by the google translate API
    param dest_key: Target Language key as specified by the google translate API
    '''

    def __init__(self):
        pass

    def evaluate(self, sentences, src_key, dest_key):

        return translator.translate(sentences, src=src_key, dest=dest_key).text
    
    def __str__(self) -> str:
        return f'Translation: {self.evaluate}'