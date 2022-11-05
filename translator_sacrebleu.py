import argparse
from kasafranse.hugging_face_utils import fineturnedsacrebleu
from transformers import MarianMTModel, MarianTokenizer
import warnings
warnings.simplefilter('ignore')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='ESTIMATE BLEU SCORE')
    parser.add_argument("translator_path",
                        help="Provide the path to the Translator", type=str)
    parser.add_argument("test_file",
                        help="Provide the file to be translated", type=str)
    parser.add_argument("reference",
                        help="Provide the reference file", type=str)

    args = parser.parse_args()
    tokenizer = MarianTokenizer.from_pretrained(args.translator_path)
    model = MarianMTModel.from_pretrained(args.translator_path)
    bleu = fineturnedsacrebleu(model, tokenizer)
    print(bleu.get_bleuscore(args.test_file, args.reference))
