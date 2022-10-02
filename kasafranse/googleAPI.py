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
    def __init__(self):
        pass

    def evaluate(self, sentences, src_key, dest_key):
        eng_text = translator.translate(sentences, src=src_key, dest='en').text
        print(eng_text)
        output = translator.translate(eng_text, dest=dest_key).text

        return output


class GoogleDirect:
    def __init__(self):
        pass

    def evaluate(self, sentences, src_key, dest_key):

        return translator.translate(sentences, src=src_key, dest=dest_key).text
