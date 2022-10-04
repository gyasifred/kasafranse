import argparse
from kasafranse.bleu import Seq2seqModelBleu
from nltk.translate.bleu_score import SmoothingFunction
from kasafranse.seq2seq_tokenizer_processing import ProcessTokenizer
import tensorflow as tf
import tensorflow_text

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='ESTIMATE BLEU SCORE')
    parser.add_argument("translator_path",
                        help="Provide the path to the Translator", type=str)
    parser.add_argument("test_file",
                        help="Provide the file to be translated", type=str)
    parser.add_argument("reference",
                        help="Provide the reference file", type=str)
    parser.add_argument(
        "src_tokenizer", help="Provide the path to input tokenizer.", type=str)
    parser.add_argument(
        "target_tokenizer", help="Provide path to target tokenizer", type=str)

    args = parser.parse_args()
    # Load Tokenizers
    # instaintiate ProcessTokenizer class
    tokenizer_preprocess = ProcessTokenizer()

    # reload input text processor
    input_text_processor = tokenizer_preprocess.loadtokenizer(
        args.src_tokenizer)

    # reload target text processor
    output_text_processor = tokenizer_preprocess.loadtokenizer(
        args.target_tokenizer)

    reloaded = tf.saved_model.load(args.translator_path)
    smooth = SmoothingFunction()
    bleu = Seq2seqModelBleu(reloaded)
    print(bleu.get_bleuscore(args.test_file, args.reference, smooth.method7))
