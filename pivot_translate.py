import argparse
from transformers import MarianMTModel, MarianTokenizer
from kasafranse.hugging_face_utils import Pivot
import warnings
warnings.simplefilter('ignore')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Translate Sentence with Fine-turned Hugging face transformer')
    parser.add_argument("first_translator_path",
                        help="Provide the path to the First Translator", type=str)
    parser.add_argument("second_translator_path",
                        help="Provide the path to the Second Translator", type=str)
    parser.add_argument("file",
                        help="Provide the file to be translated. This file must be a txt file", type=str)
    parser.add_argument("--output_name",
                        help="Pass the path and name for the final output file", default="translate.txt", type=str)
    parser.add_argument("--to_console",
                        help="Specify whether to print the ouput to console or output to txt file. The default\
                             False will save the output as txt to the curent directory", default=False, type=bool)

    args = parser.parse_args()

    model_1 = args.first_translator_path
    model_2 = args.second_translator_path
    evaluate = Pivot(model_1,model_2)
    evaluate.translate(args.file,
                       args.to_console, args.output_name)
