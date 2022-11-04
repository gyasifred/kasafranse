import argparse
import tensorflow as tf
import tensorflow_text
from kasafranse.transformer_model import Translate
import warnings
warnings.simplefilter('ignore')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Translate Sentence with Transformer translator')
    parser.add_argument("translator_path",
                        help="Provide the path to the Translator", type=str)
    parser.add_argument("file",
                        help="Provide the file to be translated. This file must be a txt file", type=str)
    parser.add_argument("--output_name",
                        help="Pass the directory and name of the final output", default="translate.txt", type=str)

    parser.add_argument("--to_console",
                        help="Specify whether to print the ouput to console or output to txt file. The default\
                             False will save the output as txt to the curent directory", default=False, type=bool)

    args = parser.parse_args()

    evaluate = Translate(args.translator_path, args.file,args.output_name, args.to_console,)
    evaluate.translate()
