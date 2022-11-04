# This code will build a subword tokenizer from the parallel text corpus
#import dependencies
import argparse
from kasafranse.transformer_tokenizer import CustomTokenizer
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
import tensorflow as tf
import warnings
warnings.simplefilter('ignore')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="BUILD SUBWORD TOKENIZER FOR ENGLISH FRENCH and AKAN-TWI USING TENSORFLOW  'text.BertTokenizer'")
    parser.add_argument("src_data",
                        help="Provide the Path to the source texts", type=str)
    parser.add_argument("targ_data",
                        help="Provide the Path to the target texts", type=str)
    parser.add_argument("vocab_size",
                        help="Provide the maximum vocabulary size", type=int)
    parser.add_argument("--save_dir", default=".",
                        help="Provide the directory to save the tokenizer to", type=str)
    parser.add_argument(
        "--tokenizer_name", type=str, default="translate_frengtwi_converter", help="Name of the tokenizer")

    args = parser.parse_args()

    # build TF datasets from input sentences in ALL languages
    # read the raw datasets
    lines_dataset_src = tf.data.TextLineDataset(args.src_data)
    lines_dataset_targ = tf.data.TextLineDataset(args.targ_data)
    
    # combine languages into single dataset
    combined = tf.data.Dataset.zip(
        (lines_dataset_src, lines_dataset_targ))

    # set tokenizer parameters and add reserved tokens; input files already lower-cased, but
    # lower_case option does NFD normalization, which is needed
    bert_tokenizer_params = dict(lower_case=True)
    reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]
    # main parameter here that could be tuned is vocab size
    bert_vocab_args = dict(
        # The target vocabulary size
        vocab_size=args.vocab_size,
        # Reserved tokens that must be included in the vocabulary
        reserved_tokens=reserved_tokens,
        # Arguments for `text.BertTokenizer`
        bert_tokenizer_params=bert_tokenizer_params,
        # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
        learn_params={},
    )

    # build source language vocab file (takes several mins)
    # this is the bert_vocab module building its vocab file from the raw French sentences
    src_vocab = bert_vocab.bert_vocab_from_dataset(
        lines_dataset_src,
        **bert_vocab_args
    )

    # build target language vocab file
    targ_vocab = bert_vocab.bert_vocab_from_dataset(
        lines_dataset_targ,
        **bert_vocab_args
    )

   

    # function to write the build vocab to file
    # this file will be used to build tokenizer

    def write_vocab_file(filepath, vocab):
        with open(filepath, 'w') as f:
            for token in vocab:
                print(token, file=f)

    # Write French, english,and TWI vocabS to file
    # use to export the tokenizer
    write_vocab_file('src_vocab.txt', src_vocab)
    write_vocab_file('targ_vocab.txt', targ_vocab)
    

    # Instantiate tokenizer class for both TWI, FRENCH, and ENGLISH
    tokenizers = tf.Module()
    tokenizers.src = CustomTokenizer(reserved_tokens, 'src_vocab.txt')
    tokenizers.targ = CustomTokenizer(reserved_tokens, 'targ_vocab.txt')
    
    # Save tokenizer model
    model_name = f'{args.save_dir}/{args.tokenizer_name}'
    tf.saved_model.save(tokenizers, model_name)


    print(f'Tokenizer save sucessfully to {args.save_dir}')