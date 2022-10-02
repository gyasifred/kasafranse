import tensorflow as tf
import numpy as np

# Calculate positional encoding (needed for Transformer model to indicate word spacing


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


# positional embedding

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
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x

# Create Pointwise feedforward network


def point_wise_feed_forward_network(
    d_model,  # Input/output dimensionality.
    dff  # Inner-layer dimensionality.
):

    return tf.keras.Sequential([
        # Shape `(batch_size, seq_len, dff)`.
        tf.keras.layers.Dense(dff, activation='relu'),
        # Shape `(batch_size, seq_len, d_model)`.
        tf.keras.layers.Dense(d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *,
                 d_model,  # Input/output dimensionality.
                 num_attention_heads,
                 dff,  # Inner-layer dimensionality.
                 dropout_rate=0.1
                 ):
        super().__init__()

        # Multi-head self-attention.
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_attention_heads,
            # Size of each attention head for query Q and key K.
            key_dim=d_model,
            dropout=dropout_rate,
        )
        # Point-wise feed-forward network.
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        # Layer normalization.
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropout for the point-wise feed-forward network.
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):

        # A boolean mask.
        if mask is not None:
            mask1 = mask[:, :, None]
            mask2 = mask[:, None, :]
            attention_mask = mask1 & mask2
        else:
            attention_mask = None

        # Multi-head self-attention output (`tf.keras.layers.MultiHeadAttention `).
        attn_output = self.mha(
            query=x,  # Query Q tensor.
            value=x,  # Value V tensor.
            key=x,  # Key K tensor.
            # A boolean mask that prevents attention to certain positions.
            attention_mask=attention_mask,
            # A boolean indicating whether the layer should behave in training mode.
            training=training,
        )

        # Multi-head self-attention output after layer normalization and a residual/skip connection.
        # Shape `(batch_size, input_seq_len, d_model)`
        out1 = self.layernorm1(x + attn_output)

        # Point-wise feed-forward network output.
        # Shape `(batch_size, input_seq_len, d_model)`
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout1(ffn_output, training=training)
        # Point-wise feed-forward network output after layer normalization and a residual skip connection.
        # Shape `(batch_size, input_seq_len, d_model)`.
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class Encoder(tf.keras.layers.Layer):
    def __init__(self,
                 *,
                 num_layers,
                 d_model,  # Input/output dimensionality.
                 num_attention_heads,
                 dff,  # Inner-layer dimensionality.
                 input_vocab_size,  # Input (Portuguese) vocabulary size.
                 dropout_rate=0.1
                 ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # Embeddings + Positional encoding
        self.pos_embedding = PositionalEmbedding(input_vocab_size, d_model)

        # Encoder layers.
        self.enc_layers = [
            EncoderLayer(
                d_model=d_model,
                num_attention_heads=num_attention_heads,
                dff=dff,
                dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        # Dropout.
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    # Masking.
    def compute_mask(self, x, previous_mask=None):
        return self.pos_embedding.compute_mask(x, previous_mask)

    def call(self, x, training):

        seq_len = tf.shape(x)[1]

        # Sum up embeddings and positional encoding.
        mask = self.compute_mask(x)
        # Shape `(batch_size, input_seq_len, d_model)`.
        x = self.pos_embedding(x)
        # Add dropout.
        x = self.dropout(x, training=training)

        # N encoder layers.
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # Shape `(batch_size, input_seq_len, d_model)`.


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,
                 *,
                 d_model,  # Input/output dimensionality.
                 num_attention_heads,
                 dff,  # Inner-layer dimensionality.
                 dropout_rate=0.1
                 ):
        super().__init__()

        # Masked multi-head self-attention.
        self.mha_masked = tf.keras.layers.MultiHeadAttention(
            num_heads=num_attention_heads,
            # Size of each attention head for query Q and key K.
            key_dim=d_model,
            dropout=dropout_rate
        )
        # Multi-head cross-attention.
        self.mha_cross = tf.keras.layers.MultiHeadAttention(
            num_heads=num_attention_heads,
            # Size of each attention head for query Q and key K.
            key_dim=d_model,
            dropout=dropout_rate
        )

        # Point-wise feed-forward network.
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        # Layer normalization.
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropout for the point-wise feed-forward network.
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, mask, enc_output, enc_mask, training):
        # The encoder output shape is `(batch_size, input_seq_len, d_model)`.

        # A boolean mask.
        self_attention_mask = None
        if mask is not None:
            mask1 = mask[:, :, None]
            mask2 = mask[:, None, :]
            self_attention_mask = mask1 & mask2

        # Masked multi-head self-attention output (`tf.keras.layers.MultiHeadAttention`).
        attn_masked, attn_weights_masked = self.mha_masked(
            query=x,
            value=x,
            key=x,
            # A boolean mask that prevents attention to certain positions.
            attention_mask=self_attention_mask,
            # A boolean to indicate whether to apply a causal mask to prevent tokens from attending to future tokens.
            use_causal_mask=True,
            # Shape `(batch_size, target_seq_len, d_model)`.
            return_attention_scores=True,
            # A boolean indicating whether the layer should behave in training mode.
            training=training
        )

        # Masked multi-head self-attention output after layer normalization and a residual/skip connection.
        out1 = self.layernorm1(attn_masked + x)

        # A boolean mask.
        attention_mask = None
        if mask is not None and enc_mask is not None:
            mask1 = mask[:, :, None]
            mask2 = enc_mask[:, None, :]
            attention_mask = mask1 & mask2

        # Multi-head cross-attention output (`tf.keras.layers.MultiHeadAttention `).
        attn_cross, attn_weights_cross = self.mha_cross(
            query=out1,
            value=enc_output,
            key=enc_output,
            # A boolean mask that prevents attention to certain positions.
            attention_mask=attention_mask,
            # Shape `(batch_size, target_seq_len, d_model)`.
            return_attention_scores=True,
            # A boolean indicating whether the layer should behave in training mode.
            training=training
        )

        # Multi-head cross-attention output after layer normalization and a residual/skip connection.
        # (batch_size, target_seq_len, d_model)
        out2 = self.layernorm2(attn_cross + out1)

        # Point-wise feed-forward network output.
        # Shape `(batch_size, target_seq_len, d_model)`.
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout1(ffn_output, training=training)
        # Shape `(batch_size, target_seq_len, d_model)`.
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_masked, attn_weights_cross


class Decoder(tf.keras.layers.Layer):
    def __init__(self,
                 *,
                 num_layers,
                 d_model,  # Input/output dimensionality.
                 num_attention_heads,
                 dff,  # Inner-layer dimensionality.
                 target_vocab_size,
                 dropout_rate=0.1
                 ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(target_vocab_size, d_model)

        self.dec_layers = [
            DecoderLayer(
                d_model=d_model,
                num_attention_heads=num_attention_heads,
                dff=dff,
                dropout_rate=dropout_rate)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, enc_mask, training):
        attention_weights = {}

        mask = self.pos_embedding.compute_mask(x)
        # Shape: `(batch_size, target_seq_len, d_model)`.
        x = self.pos_embedding(x)

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](
                x, mask, enc_output, enc_mask, training)

            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        # The shape of x is `(batch_size, target_seq_len, d_model)`.
        return x, attention_weights
