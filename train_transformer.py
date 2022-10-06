import argparse
from kasafranse.transformer_model import Transformer, Translator
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
from kasafranse.transformer_tokenizer import CustomTokenizer
from kasafranse.transformer_layers import create_masks
from kasafranse.transformer_utils import CustomSchedule, loss_function, accuracy_function,\
    ProcessBatch, write_vocab_file
import tensorflow as tf
import tensorflow_text
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training of Transformer Language Model for Machine Translation: Tensorflow')
    parser.add_argument("src_train_data",
                        help="Provide the Path to the Input language Training  Data", type=str)
    parser.add_argument("target_train_data",
                        help="Provide the Path to the Target Language Training  Data", type=str)

    parser.add_argument("tokenizer",
                        help="Provide the Path to the Tokenizer", type=str)
    parser.add_argument(
        "--buffer_size", type=int, default=30000, help="Enter the buffer size for creating batches of data")
    parser.add_argument(
        "--max_input_length", type=int, default=50, help="Enter the maximum length for the source language")
    parser.add_argument(
        "--batch_size", type=int, default=50, help="Give the batch size for training data")
    parser.add_argument(
        "--num_layers", type=int, default=6, help="State the number of encoder and decoder layers")
    parser.add_argument(
        "--d_model", type=int, default=512, help="Give the embedding dimension")
    parser.add_argument(
        "--dff", type=int, default=2048, help="Give the output dimension")
    parser.add_argument(
        "--heads", type=int, default=8, help="Give the number of multihead attentions")
    parser.add_argument(
        "--dropout", type=int, default=0.1, help="dropout rate")
    parser.add_argument(
        "--epoch", type=int, default=100, help="Enter the number of training epochs")
    parser.add_argument('--save_dir', type=str,
                        help="Provide the path to save to the fineturned model")
    parser.add_argument("model_name", type=str,
                        help="Namefor saving the trained model")

    args = parser.parse_args()

    # build tf datasets
    src_train_data = tf.data.TextLineDataset(args.src_train_data)
    targ_train_data = tf.data.TextLineDataset(args.target_train_data)

    # combine languages into single dataset
    trained_combined = tf.data.Dataset.zip((src_train_data, targ_train_data))

    # Build Tokenizer
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
    src_vocab = bert_vocab.bert_vocab_from_dataset(
        src_train_data,
        **bert_vocab_args
    )
    targ_vocab = bert_vocab.bert_vocab_from_dataset(
        targ_train_data,
        **bert_vocab_args
    )

    # Write source and target vocabS to file
    # use to export the tokenizer
    write_vocab_file('src_vocab.txt', src_vocab)
    write_vocab_file('targ_vocab.txt', targ_vocab)

    # Instantiate tokenizer class
    tokenizers = tf.Module()
    tokenizers.src = CustomTokenizer(reserved_tokens, 'src_vocab.txt')
    tokenizers.targ = CustomTokenizer(reserved_tokens, 'targ_vocab.txt')

    # create batches of data for training
    # set MAX_TOKENS
    MAX_TOKENS = args.max_input_length
    # create training batches
    BUFFER_SIZE = args.buffer_size
    BATCH_SIZE = args.batch_size

    # Instantiate ProcessBatch class
    batch_processor = ProcessBatch(
        tokenizers.src, tokenizers.targ, MAX_TOKENS)

    # Create training set batches.
    train_batches = batch_processor.make_batches(
        trained_combined, BUFFER_SIZE, BATCH_SIZE)

    # Set hyperparameters for  Transformer
    num_layers = args.num_layers
    d_model = args.d_model
    dff = args.dff
    num_heads = args.heads
    dropout_rate = args.dropout

    # Instantiate  Transformer
    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=tokenizers.src.get_vocab_size().numpy(),
        target_vocab_size=tokenizers.targ.get_vocab_size().numpy(),
        rate=dropout_rate)

    # Instantiate learning rate and set optimizer
    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    # Set up training checkpoints
    checkpoint_path = "./checkpoints/train"
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, checkpoint_path, max_to_keep=5)
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    # Choose number of training epochs
    EPOCHS = args.epoch

    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]
    # The @tf.function trace-compiles train_step into a TF graph for faster
    # execution. The function specializes to the precise shape of the argument
    # tensors. To avoid re-tracing due to the variable sequence lengths or variable
    # batch sizes (the last batch is smaller), use input_signature to specify
    # more generic shapes.

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            inp, tar_inp)
        with tf.GradientTape() as tape:
            predictions, _ = transformer(
                inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
            loss = loss_function(tar_real, predictions)
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, transformer.trainable_variables))
        train_loss(loss)
        train_accuracy(accuracy_function(tar_real, predictions))

    # Run training! Each epoch takes several mins with GPU
    for epoch in range(EPOCHS):
        start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    # inp -> twi, tar -> french
    for (batch, (inp, tar)) in enumerate(train_batches):
        train_step(inp, tar)

        if batch % 50 == 0:
            print(
                f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

    print(
        f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

    print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

    # translator
    class Translator(tf.Module):
        def __init__(self, tokenizers, transformer):
            self.tokenizers = tokenizers
            self.transformer = transformer

        def __call__(self, sentence, max_length=MAX_TOKENS):
            # The input sentence is English, hence adding the `[START]` and `[END]` tokens.
            sentence = tf.convert_to_tensor([sentence])
            sentence = self.tokenizers.src.tokenize(sentence)
            # trim sentence greater than Max_tokens
            sentence = sentence[:, :self.max_length]
            sentence = sentence.to_tensor()

            encoder_input = sentence

            # As the output language is TWI, initialize the output with the
            # English `[START]` token.
            start_end = self.tokenizers.targ.tokenize([''])[0]
            start = start_end[0][tf.newaxis]
            end = start_end[1][tf.newaxis]

            # `tf.TensorArray` is required here (instead of a Python list), so that the
            # dynamic-loop can be traced by `tf.function`.
            output_array = tf.TensorArray(
                dtype=tf.int64, size=0, dynamic_size=True)
            output_array = output_array.write(0, start)
            output = tf.transpose(output_array.stack())

            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                encoder_input, output)

            for i in tf.range(max_length):
                output = tf.transpose(output_array.stack())

                enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                    encoder_input, output)

                predictions, attention_weights = self.transformer(
                    encoder_input, output, False, enc_padding_mask, combined_mask, dec_padding_mask)

                # Select the last token from the `seq_len` dimension.
                # Shape `(batch_size, 1, vocab_size)`.
                predictions = predictions[:, -1:, :]

                predicted_id = tf.argmax(predictions, axis=-1)

                # Concatenate the `predicted_id` to the output which is given to the
                # decoder as its input.
                output_array = output_array.write(i+1, predicted_id[0])

                if predicted_id == end:
                    break

            output = tf.transpose(output_array.stack())
            # The output shape is `(1, tokens)`.
            text = self.tokenizers.targ.detokenize(output)[0]  # Shape: `()`.

            tokens = self.tokenizers.targ.lookup(output)[0]
            _, attention_weights = self.transformer(
                encoder_input, output[:, :-1], False, enc_padding_mask, combined_mask, dec_padding_mask)

            return text, tokens, attention_weights

    # Create an instance of this Translator class
    translator = Translator(tokenizers, transformer)

    # class to export translator
    class ExportTranslator(tf.Module):
        def __init__(self, translator):
            self.translator = translator

        @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
        def __call__(self, sentence):
            (result,
             tokens,
             attention_weights) = self.translator(sentence, max_length=MAX_TOKENS)

            return result

    # SAVE MODEL
    translator = ExportTranslator(translator)
    if args.save_dir:

        tf.saved_model.save(
            translator, export_dir=f'{args.save_dir}/{args.model_name}')
    else:
        tf.saved_model.save(translator, export_dir=args.model_name)
