import argparse
import tensorflow as tf
import tensorflow_text
from kasafranse.seq2seq_model import Translate
from kasafranse.seq2seq_tokenizer_processing import ProcessTokenizer


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Translate Sentence with SEQ2SEQ MODEL with Attention')
    parser.add_argument("translator_path",
                        help="Provide the path to the Translator", type=str)
    parser.add_argument(
        "src_tokenizer", help="Provide the path to input tokenizer.", type=str)
    parser.add_argument(
        "target_tokenizer", help="Provide path to target tokenizer", type=str)

    parser.add_argument("file",
                        help="Provide the file to be translated. This file must be a txt file", type=str)
    parser.add_argument("--output_name",
                        help="Pass the directory and name of the final output", default="translate.txt", type=str)

    parser.add_argument("--to_console",
                        help="Specify whether to print the ouput to console or output to txt file. The default\
                             False will save the output as txt to the curent directory", default=False, type=bool)

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

    evaluate = Translate(args.translator_path, args.file,
                         args.output_name, args.to_console,)
    evaluate.translate()
