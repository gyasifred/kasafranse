import argparse
from kasafranse.bleu import BLEU, BleuScore
from nltk.translate.bleu_score import SmoothingFunction

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='ESTIMATE BLEU SCORE')
    parser.add_argument("hypothesis",
                        help="Provide the hypothesis file", type=str)
    parser.add_argument("reference",
                        help="Provide the reference file", type=str)
    parser.add_argument("--get_ngrams", default=False,
                        help="State if you want BLEU scores for n-grams", type=bool)

    args = parser.parse_args()

    if args.get_ngrams == True:
        smooth = SmoothingFunction()
        bleu = BLEU()
        print(bleu.get_bleuscore(args.hypothesis, args.reference, smooth.method7))
    else:
        smooth = SmoothingFunction()
        bleu = BleuScore()
        print(bleu.get_bleuscore(args.hypothesis, args.reference, smooth.method7))
