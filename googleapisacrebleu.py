import argparse
from kasafranse.bleu import GoogleAPISacredBleu
from kasafranse.googleAPI import GoogleDirect, GooglePivot
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='ESTIMATE BLEU SCORE')
    parser.add_argument("translator_path",
                        help="Provide the path to the Translator", type=str)
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

    if args.use_pivot == True:
        evaluate = GooglePivot()
        bleu = GoogleAPISacredBleu(evaluate)
        print(bleu.get_bleuscore(args.test_file,
              args.reference, args.src_key, args.dest_key))
    else:
        evaluate = GoogleDirect()
        bleu = GoogleAPISacredBleu(evaluate)
        print(bleu.get_bleuscore(args.test_file,
              args.reference, args.src_key, args.dest_key))
