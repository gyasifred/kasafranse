import argparse
from kasafranse.seq2seq_tokenizer_processing import ProcessTokenizer
# import libraries
import tensorflow as tf
import tensorflow_text as tf_text
import pickle

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="BUILD TOKENIZER FOR ENGLISH FRENCH and AKAN-TWI USING 'tf.keras.layers.TextVectorization'")
    parser.add_argument("twi_data",
                        help="Provide the Path to the TWI texts", type=str)
    parser.add_argument("french_data",
                        help="Provide the Path to the FRENCH texts", type=str)
    parser.add_argument("english_data",
                        help="Provide the Path to the ENGLISH texts", type=str)
    parser.add_argument("vocab_size",
                        help="Provide the maximum vocabulary size", type=int)
    parser.add_argument("--save_dir", default=".",
                        help="Provide the directory to save the tokenizer to", type=str)
    args = parser.parse_args()

    # build TF datasets from input sentences in ALL languages
    # read the raw datasets
    lines_dataset_fr = tf.data.TextLineDataset(args.french_data)
    lines_dataset_tw = tf.data.TextLineDataset(args.twi_data)
    lines_dataset_eng = tf.data.TextLineDataset(args.english_data)

    # Text Vectorization
    # instaintiate ProcessTokenizer class
    tokenizer_preprocess = ProcessTokenizer()

    # set maximum vocaburary size
    max_vocab_size = args.vocab_size

    # Process twi
    twi_tokenizer = tokenizer_preprocess.build_tokenizer(
        lines_dataset_tw, max_vocab_size)

    # Process french
    french_tokenizer = tokenizer_preprocess.build_tokenizer(
        lines_dataset_fr, max_vocab_size)

    # Process english
    english_tokenizer = tokenizer_preprocess.build_tokenizer(
        lines_dataset_eng, max_vocab_size)

    # save Twi tokenizer
    tokenizer_preprocess.savetokenizer(
        f'{args.save_dir}/twi_tokenizer.pkl', twi_tokenizer)

    # save French tokenizer
    tokenizer_preprocess.savetokenizer(
        f'{args.save_dir}/french_tokenizer.pkl', french_tokenizer)

    # save english tokenizer
    tokenizer_preprocess.savetokenizer(
        f'{args.save_dir}/english_tokenizer.pkl', english_tokenizer)
    

    print(f'TOKENIZER PROCESSING  COMPLETE')
