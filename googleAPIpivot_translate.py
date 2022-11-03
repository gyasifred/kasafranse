import argparse
from kasafranse.googleAPI import GooglePivot
from kasafranse.preprocessing import Preprocessing


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Pivot Translations with Google Translate API, pivot language :English')
    parser.add_argument("file",
                        help="Provide the file to be translated. This file must be a txt file", type=str)
    parser.add_argument("src_key",
                        help="Provide the google translate API key for the source language", type=str)
    parser.add_argument("dest_key",
                        help="Provide the google translate API key for the destination language", type=str)
    parser.add_argument("--output_name",
                        help="Pass the name of the final output", default="translate.txt", type=str)
    parser.add_argument("--output_dir",
                        help="Pass the directory to save the file to", type=str)

    parser.add_argument("--to_console",
                        help="Specify whether to print the ouput to console or output to txt file. The default\
                             False will save the output as txt to the curent directory", default=False, type=bool)

    args = parser.parse_args()

    translator = GooglePivot()
    preprocessor = Preprocessing()

    if args.to_console == True:
        with open(args.file) as txt_file:
            for line in txt_file:
                translated_text = translator.evaluate(
                    line, args.src_key, args.dest_key)
                #print(translated_text)
    else:
        lines = []
        with open(args.file) as txt_file:
            for line in txt_file:
                translated_text = translator.evaluate(
                    line, args.src_key, args.dest_key)
                lines.append(translated_text)

        preprocessor.writeTotxt(f'{args.model_dir}/{args.output_name}', lines)


    