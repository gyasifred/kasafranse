import argparse
from kasafranse.googleAPI import GoogleDirect
from kasafranse.preprocessing import Preprocessing
import warnings
warnings.simplefilter('ignore')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Translate Text with Google Translate API')
    parser.add_argument("file",
                        help="Provide the file to be translated. This file must be a txt file", type=str)
    parser.add_argument("src_key",
                        help="Provide the google translate API key for the source language", type=str)
    parser.add_argument("dest_key",
                        help="Provide the google translate API key for the destination language", type=str)
    parser.add_argument("--output_name",
                        help="Pass path and name of the final output file", default="translate.txt", type=str)
    
    parser.add_argument("--to_console",
                        help="Specify whether to print the ouput to console or output to txt file. The default\
                             False will save the output as txt to the curent directory", default=False, type=bool)

    args = parser.parse_args()

    translator = GoogleDirect()
    preprocessor = Preprocessing()

    if args.to_console == True:
        with open(args.file) as txt_file:
            for line in txt_file:
                translated_text = translator.evaluate(
                    line, args.src_key, args.dest_key)
                print(translated_text)
    else:
        lines = []
        with open(args.file) as txt_file:
            for line in txt_file:
                translated_text = translator.evaluate(
                    line, args.src_key, args.dest_key)
                lines.append(translated_text)

        preprocessor.writeTotxt(args.output_name, lines)