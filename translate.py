import argparse
from transformers import MarianMTModel, MarianTokenizer
from kasafranse.hugging_face_utils import Translate
import warnings
warnings.simplefilter('ignore')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Translate Sentence with Fine-turned Hugging face transformer')
    parser.add_argument("translator_path",
                        help="Provide the path to the Translator", type=str)
    parser.add_argument("file",
                        help="Provide the file to be translated. This file must be a txt file", type=str)
    parser.add_argument("--output_name",
                        help="Pass the name for the final output file", default="translate.txt", type=str)
    parser.add_argument("--dir",
                        help="Pass the directory for the final output file", default=None, type=str)

    parser.add_argument("--to_console",
                        help="Specify whether to print the ouput to console or output to txt file. The default\
                             False will save the output as txt to the curent directory", default=False, type=bool)

    args = parser.parse_args()

    model_name = args.translator_path
    evaluate = Translate(model_name)
    evaluate.translate(args.file,
                       args.to_console, args.dir, args.output_name)
