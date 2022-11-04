import argparse
from kasafranse.transformer_model import TransformerSacredBleu
import tensorflow_text
import tensorflow as tf
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

    reloaded = tf.saved_model.load(args.translator_path)
    bleu = TransformerSacredBleu(reloaded)
    print(bleu.get_bleuscore(args.test_file, args.reference))
    
