import argparse
from kasafranse.googleAPI import GoogleDirect, GooglePivot,GoogleAPIBleu
from nltk.translate.bleu_score import SmoothingFunction

import warnings
warnings.simplefilter('ignore')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='ESTIMATE BLEU SCORE')
    parser.add_argument("test_file",
                        help="Provide the file to be translated", type=str)
    parser.add_argument("reference",
                        help="Provide the reference file", type=str)
    parser.add_argument("src_key",
                        help="Provide the google translate API key for the source language", type=str)
    parser.add_argument("dest_key",
                        help="Provide the google translate API key for the destination language", type=str)
    parser.add_argument("--use_pivot", default=False,
                        help="Specify to use Pivot translate system", type=bool)

    args = parser.parse_args()
    smooth = SmoothingFunction()

    if args.use_pivot == True:
        evaluate = GooglePivot()
        bleu = GoogleAPIBleu(evaluate)
        print(bleu.get_bleuscore(args.test_file,
              args.reference, args.src_key, args.dest_key,smooth.method7))
    else:
        evaluate = GoogleDirect()
        bleu = GoogleAPIBleu(evaluate)
        print(bleu.get_bleuscore(args.test_file,
              args.reference, args.src_key, args.dest_key,smooth.method7))
