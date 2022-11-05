import argparse
from kasafranse.bleu import Sacrebleu

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='ESTIMATE SacreBLEU SCORE')
    parser.add_argument("hypothesis",
                        help="Provide the hypothesis file", type=str)
    parser.add_argument("reference",
                        help="Provide the reference file", type=str)

    args = parser.parse_args()
    bleu = Sacrebleu()
    print(bleu.get_bleuscore(args.hypothesis, args.reference))
