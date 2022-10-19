import argparse
import logging
import time
import numpy as np
import tensorflow as tf

import tensorflow_text
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training of Transformer Language Model for Machine Translation: Tensorflow')
    parser.add_argument("src_train_data",
                        help="Provide the Path to the Input language Training  Data", type=str)
    parser.add_argument("target_train_data",
                        help="Provide the Path to the Target Language Training  Data", type=str)
    parser.add_argument(
        "src_val_data", help="Provide the Path to the Input language validation Data", type=str)
    parser.add_argument(
        "target_val_data", help="Provide the Path to the Target Language validation Data", type=str)
    parser.add_argument("tokenizer",
                        help="Provide the Path to the Tokenizer", type=str)
    parser.add_argument(
        "--buffer_size", type=int, default=20000, help="Enter the buffer size for creating batches of data")
    parser.add_argument(
        "--max_input_length", type=int, default=50, help="Enter the maximum length for the source language")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Give the batch size for training data")
    parser.add_argument(
        "--epoch", type=int, default=100, help="Enter the number of training epochs")
    parser.add_argument('--save_dir', type=str,
                        help="Provide the path to save to the trained model")
    parser.add_argument("model_name", type=str,
                        help="Trined model name")

    args = parser.parse_args()

    # build tf datasets from traning and validation sentences in both languages
    src_train_data = tf.data.TextLineDataset(args.src_train_data)
    targ_train_data = tf.data.TextLineDataset(args.target_train_data)
    val_dataset_src = tf.data.TextLineDataset(args.src_val_data)
    val_dataset_targ = tf.data.TextLineDataset(args.target_val_data)

    # combine languages into single dataset
    trained_combined = tf.data.Dataset.zip((src_train_data, targ_train_data))
    val_combined = tf.data.Dataset.zip((val_dataset_src, val_dataset_targ))

    # import tokenizer
    model_named = args.tokenizer
    tokenizers = tf.saved_model.load(model_named)

    # Process traning and validation training batches
    MAX_TOKENS = args.max_input_length

    def prepare_batch(l1, l2):
        l1 = tokenizers.src.tokenize(l1)      # Output is ragged.
        l1 = l1[:, :MAX_TOKENS]    # Trim to MAX_TOKENS.
        l1 = l1.to_tensor()  # Convert to 0-padded dense Tensor

        l2 = tokenizers.targ.tokenize(l2)
        l2 = l2[:, :(MAX_TOKENS+1)]
        l2_inputs = l2[:, :-1].to_tensor()  # Drop the [END] tokens
        l2_labels = l2[:, 1:].to_tensor()   # Drop the [START] tokens

        return (l1, l2_inputs), l2_labels

    BUFFER_SIZE = args.buffer_size
    BATCH_SIZE = args.batch_size

    def make_batches(ds):
        return (
            ds
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .map(prepare_batch, tf.data.AUTOTUNE)
            .prefetch(buffer_size=tf.data.AUTOTUNE))

    # Create training and validation set batches.
    train_batches = make_batches(trained_combined)
    val_batches = make_batches(val_combined)

    # define transformer components
    # define positional encoding

    def positional_encoding(length, depth):
        depth = depth/2

        positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
        depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

        angle_rates = 1 / (10000**depths)         # (1, depth)
        angle_rads = positions * angle_rates      # (pos, depth)

        pos_encoding = np.concatenate(
            [np.sin(angle_rads), np.cos(angle_rads)],
            axis=-1)

        return tf.cast(pos_encoding, dtype=tf.float32)

    # define positional embedding
    class PositionalEmbedding(tf.keras.layers.Layer):
        def __init__(self, vocab_size, d_model):
            super().__init__()
            self.d_model = d_model
            self.embedding = tf.keras.layers.Embedding(
                vocab_size, d_model, mask_zero=True)
            self.pos_encoding = positional_encoding(length=2048, depth=d_model)

        def compute_mask(self, *args, **kwargs):
            return self.embedding.compute_mask(*args, **kwargs)

        def call(self, x):
            length = tf.shape(x)[1]
            x = self.embedding(x)
            # This factor sets the relative scale of the embedding and positonal_encoding.
            x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            x = x + self.pos_encoding[tf.newaxis, :length, :]
            return x

        class BaseAttention(tf.keras.layers.Layer):
            def __init__(self, **kwargs):
                super().__init__()
                self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
                self.layernorm = tf.keras.layers.LayerNormalization()
                self.add = tf.keras.layers.Add()
        
        class CrossAttention(BaseAttention):
            def call(self, x, context):
                attn_output, attn_scores = self.mha(
                    query=x,
                    key=context,
                    value=context,
                    return_attention_scores=True)
            
                # Cache the attention scores for plotting later.
                self.last_attn_scores = attn_scores

                x = self.add([x, attn_output])
                x = self.layernorm(x)

                return x

    # SAVE MODEL
    translator = ExportTranslator(translator)
    if args.save_dir:

        tf.saved_model.save(
            translator, export_dir=f'{args.save_dir}/{args.model_name}')
    else:
        tf.saved_model.save(translator, export_dir=args.model_name)
