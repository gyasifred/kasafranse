import argparse
from kasafranse.seq2seq_train_utils import TrainTranslator, MaskedLoss, ProcessBatch
from kasafranse.seq2seq_tokenizer_processing import ProcessTokenizer
from kasafranse.preprocessing import Preprocessing
from kasafranse.seq2seq_model import Translator
import tensorflow as tf
import tensorflow_text

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Training Neural Machine Translation with Attention: Tensorflow')
    parser.add_argument("src_train_data",
                        help="Provide the Path to the Input language Training  Data", type=str)
    parser.add_argument("target_train_data",
                        help="Provide the Path to the Target Language Training  Data", type=str)
    parser.add_argument("vocab_size",
                        help="Provide the maximum vocabulary size", type=int)
    parser.add_argument(
        "--batch_size", type=int, default=50, help="Give the batch size for training data")
    parser.add_argument(
        "--d_model", type=int, default=1024, help="Give the embedding dimension")
    parser.add_argument(
        "--units", type=int, default=1024, help="Give the rnn units")
    parser.add_argument(
        "--epoch", type=int, default=20, help="Enter the number of training epochs")
    parser.add_argument('--save_dir', type=str,
                        help="Provide the path to save to the fineturned model")
    parser.add_argument("model_name", type=str,
                        help="Namefor saving the trained model")

    args = parser.parse_args()

    # build  Tokenizers
    # build TF datasets from input sentences in ALL languages
    lines_src = tf.data.TextLineDataset(args.src_train_data)
    lines_targ = tf.data.TextLineDataset(args.target_train_data)

    # Text Vectorization
    # instaintiate ProcessTokenizer class
    tokenizer_preprocess = ProcessTokenizer()

    # set maximum vocaburary size
    max_vocab_size = args.vocab_size

    # Processor for source language
    input_text_processor = tokenizer_preprocess.build_tokenizer(
        lines_src, max_vocab_size)

    output_text_processor = tokenizer_preprocess.build_tokenizer(
        lines_targ, max_vocab_size)

    # import preprocessing class
    # Create an instance of tft preprocessing class
    preprocessor = Preprocessing()

    # Read raw parallel dataset
    inp, targ = preprocessor.read_parallel_dataset(
        filepath_1=args.src_train_data,
        filepath_2=args.target_train_data
    )

    # create training batches
    BUFFER_SIZE = len(inp)
    BATCH_SIZE = args.batch_size

    # Instiante Process batch class
    processbatches = ProcessBatch()

    # train batches
    trained_dataset = processbatches.make_batches(
        inp, targ, BUFFER_SIZE, BATCH_SIZE)

    # set Hyerperameters
    embedding_dim = args.d_model
    units = args.units

    # train a model
    train_translator = TrainTranslator(
        embedding_dim, units,
        input_text_processor=input_text_processor,
        output_text_processor=output_text_processor,
        use_tf_function=False)

    # Configure the loss and optimizer
    train_translator.compile(
        optimizer=tf.optimizers.Adam(),
        loss=MaskedLoss(),
    )

    train_translator.use_tf_function = True

    train_translator.fit(trained_dataset, epochs=args.epoch)

    # instantiate a translator
    translate = Translator(
        encoder=train_translator.encoder,
        decoder=train_translator.decoder,
        input_text_processor=input_text_processor,
        output_text_processor=output_text_processor,
    )

    # Save model
    if args.save_dir:
        tf.saved_model.save(translate, f'{args.save_dir}/{args.model_name}',
                            signatures={'serving_default': translate.__call__})
    else:
        tf.saved_model.save(translate, args.model_name,
                            signatures={'serving_default': translate.__call__})

    print('TRAINING COMPLETE')
    print(f'TRANSLATOR SAVE SUCESSFULLY TO DIRECTORY {args.save_dir}')
