import argparse
from kasafranse.transformer_model import Transformer, Translator,\
    ExportTranslator
from kasafranse.transformer_utils import CustomSchedule, loss_function, accuracy_function,\
    ProcessBatch
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

    # build tf datasets from traning and validation sentences in both languages
    src_train_data = tf.data.TextLineDataset(args.src_train_data)
    targ_train_data = tf.data.TextLineDataset(args.target_train_data)

    # combine languages into single dataset
    trained_combined = tf.data.Dataset.zip((src_train_data, targ_train_data))

    # import tokenizer
    model_named = args.tokenizer
    tokenizers = tf.saved_model.load(model_named)

    # set input and output processors
    input_processor = tokenizers.eng
    output_processor = tokenizers.twi

    # set MAX_TOKENS
    MAX_TOKENS = args.max_input_length
    # create training batches
    BUFFER_SIZE = args.buffer_size
    BATCH_SIZE = args.batch_size

    # Instantiate ProcessBatch class
    batch_processor = ProcessBatch(
        input_processor, output_processor, MAX_TOKENS)

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
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_attention_heads=num_heads,
        dff=dff,
        input_vocab_size=input_processor.get_vocab_size().numpy(),
        target_vocab_size=output_processor.get_vocab_size().numpy(),
        dropout_rate=dropout_rate)

    # Instantiate learning rate and set optimizer
    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

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
        (
            tf.TensorSpec(shape=(None, None), dtype=tf.int64),
            tf.TensorSpec(shape=(None, None), dtype=tf.int64)),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]

    # The `@tf.function` trace-compiles train_step into a TF graph for faster
    # execution. The function specializes to the precise shape of the argument
    # tensors. To avoid re-tracing due to the variable sequence lengths or variable
    # batch sizes (the last batch is smaller), use input_signature to specify
    # more generic shapes.

    @tf.function(input_signature=train_step_signature)
    def train_step(inputs, labels):
        (inp, tar_inp) = inputs
        tar_real = labels

        with tf.GradientTape() as tape:
            predictions, _ = transformer([inp, tar_inp],
                                         training=True)
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

    # Create an instance of this Translator class
    translator = Translator(transformer, input_processor=input_processor,
                            output_processor=output_processor, max_length=MAX_TOKENS)

    # SAVE MODEL
    translator = ExportTranslator(translator)
    if args.save_dir:

        tf.saved_model.save(
            translator, export_dir=f'{args.save_dir}/{args.model_name}')
    else:
        tf.saved_model.save(translator, export_dir=args.model_name)
