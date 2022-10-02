# This code will build a subword tokenizer from the parallel text corpus
#import dependencies
import argparse
from kasafranse.transformer_tokenizer import CustomTokenizer
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
import tensorflow as tf

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="BUILD SUBWORD TOKENIZER FOR ENGLISH FRENCH and AKAN-TWI USING TENSORFLOW  'text.BertTokenizer'")
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
    parser.add_argument(
        "--tokenizer_name", type=str, default="translate_frengtwi_converter", help="Name of the tokenizer")

    args = parser.parse_args()

    # build TF datasets from input sentences in ALL languages
    # read the raw datasets
    lines_dataset_fr = tf.data.TextLineDataset(args.french_data)
    lines_dataset_tw = tf.data.TextLineDataset(args.twi_data)
    lines_dataset_eng = tf.data.TextLineDataset(args.english_data)

    # combine languages into single dataset
    combined = tf.data.Dataset.zip(
        (lines_dataset_tw, lines_dataset_fr, lines_dataset_eng))

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

    # build French vocab file (takes several mins)
    # this is the bert_vocab module building its vocab file from the raw French sentences
    fr_vocab = bert_vocab.bert_vocab_from_dataset(
        lines_dataset_fr,
        **bert_vocab_args
    )

    # build Twi vocab file
    twi_vocab = bert_vocab.bert_vocab_from_dataset(
        lines_dataset_tw,
        **bert_vocab_args
    )

    # build English vocab file
    eng_vocab = bert_vocab.bert_vocab_from_dataset(
        lines_dataset_eng,
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
    write_vocab_file('fr_vocab.txt', fr_vocab)
    write_vocab_file('twi_vocab.txt', twi_vocab)
    write_vocab_file('eng_vocab.txt', eng_vocab)

    # Instantiate tokenizer class for both TWI, FRENCH, and ENGLISH
    tokenizers = tf.Module()
    tokenizers.fr = CustomTokenizer(reserved_tokens, 'fr_vocab.txt')
    tokenizers.twi = CustomTokenizer(reserved_tokens, 'twi_vocab.txt')
    tokenizers.eng = CustomTokenizer(reserved_tokens, 'eng_vocab.txt')

    # Save tokenizer model
    model_name = f'{args.save_dir}/{args.tokenizer_name}'
    tf.saved_model.save(tokenizers, model_name)

    # # Verify tokenizer model can be reloaded
    # tokenizers = tf.saved_model.load(model_name)
    # print(tokenizers.fr.get_vocab_size().numpy())
    # print(tokenizers.twi.get_vocab_size().numpy())
    # print(tokenizers.eng.get_vocab_size().numpy())

    # # Verify tokenizer works on test french sentence
    # tokens = tokenizers.fr.tokenize(['je suis étudiant'])
    # text_tokens = tokenizers.fr.lookup(tokens)
    # print(text_tokens)

    # # Remove token markers to get original sentence
    # round_trip = tokenizers.fr.detokenize(tokens)
    # print(round_trip.numpy()[0].decode('utf-8'))

    # # Verify if tokenizer work on test Twi sentence
    # tokens = tokenizers.twi.tokenize(['Obiara ani gyee n akokoduru no ho .'])
    # text_tokens = tokenizers.twi.lookup(tokens)
    # print(text_tokens)

    # # Remove token markers to get original sentence
    # round_trip = tokenizers.twi.detokenize(tokens)
    # print(round_trip.numpy()[0].decode('utf-8'))

    # # Verify if tokenizer work on test English sentence
    # tokens = tokenizers.eng.tokenize(['Patience is key for happiness'])
    # text_tokens = tokenizers.eng.lookup(tokens)
    # print(text_tokens)

    # # Remove token markers to get original sentence
    # round_trip = tokenizers.eng.detokenize(tokens)
    # print(round_trip.numpy()[0].decode('utf-8'))
    
    print("Tokenizer processing complete")