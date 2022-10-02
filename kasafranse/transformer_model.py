from kasafranse.preprocessing import Preprocessing
from kasafranse.transformer_layers import Encoder, Decoder
import tensorflow as tf
import tensorflow_text

# instatiate Preprocessing
preprocessor = Preprocessing()

# Build full Transformer


class Transformer(tf.keras.Model):
    def __init__(self,
                 *,
                 num_layers,  # Number of decoder layers.
                 d_model,  # Input/output dimensionality.
                 num_attention_heads,
                 dff,  # Inner-layer dimensionality.
                 input_vocab_size,  # Input vocabulary size.
                 target_vocab_size,  # Target vocabulary size.
                 dropout_rate=0.1
                 ):
        super().__init__()
        # The encoder.
        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            num_attention_heads=num_attention_heads,
            dff=dff,
            input_vocab_size=input_vocab_size,
            dropout_rate=dropout_rate
        )

        # The decoder.
        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_attention_heads=num_attention_heads,
            dff=dff,
            target_vocab_size=target_vocab_size,
            dropout_rate=dropout_rate
        )

        # The final linear layer.
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, training):
        # Keras models prefer if you pass all your inputs in the first argument.
        inp, tar = inputs

        # The encoder output.
        # `(batch_size, inp_seq_len, d_model)`
        enc_output = self.encoder(inp, training)
        enc_mask = self.encoder.compute_mask(inp)

        # The decoder output.
        dec_output, attention_weights = self.decoder(
            tar, enc_output, enc_mask, training)  # `(batch_size, tar_seq_len, d_model)`

        # The final linear layer output.
        # Shape `(batch_size, tar_seq_len, target_vocab_size)`.
        final_output = self.final_layer(dec_output)

        # Return the final output and the attention weights.
        return final_output, attention_weights


class Translator(tf.Module):
    def __init__(self,transformer, input_processor, output_processor, max_length):
        self.input_tokenizer = input_processor
        self.output_tokenizer = output_processor
        self.transformer = transformer
        self.max_length = max_length

    def __call__(self, sentence):
        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]

        sentence = self.input_tokenizer.tokenize(sentence).to_tensor()

        encoder_input = sentence

        start_end = self.output_tokenizer.tokenize([''])[0]
        start = start_end[0][tf.newaxis]
        end = start_end[1][tf.newaxis]

        # `tf.TensorArray` is required here (instead of a Python list), so that the
        # dynamic-loop can be traced by `tf.function`.
        output_array = tf.TensorArray(
            dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        for i in tf.range(self.max_length):
            output = tf.transpose(output_array.stack())
            predictions, _ = self.transformer(
                [encoder_input, output], training=False)

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
        text = self.output_tokenizer.detokenize(output)[0]  # Shape: `()`.

        tokens = self.output_tokenizer.lookup(output)[0]

        # `tf.function` prevents us from using the attention_weights that were
        # calculated on the last iteration of the loop.
        # Therefore, recalculate them outside the loop.
        _, attention_weights = self.transformer(
            [encoder_input, output[:, :-1]], training=False)

        return text, tokens, attention_weights


class ExportTranslator(tf.Module):
    def __init__(self, translator):
        self.translator = translator

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def __call__(self, sentence):
        (result,
         tokens,
         attention_weights) = self.translator(sentence)

        return result


# FUNCTION TO PRINT TRANSLATIONS


def print_translation(sentence, tokens, ground_truth):
    print(f'{"Input:":15s}: {sentence}')
    print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
    print(f'{"Ground truth":15s}: {ground_truth}')


class Translate:
    def __init__(self, translator_path, file, output_name="translate.txt", to_console=False):
        self.translator = translator_path
        self.file = file
        self.ouput = output_name
        self.to_console = to_console

    def translate(self):
        evaluate = tf.saved_model.load(self.translator)
        if self.to_console == True:
            with open(self.file) as txt_file:
                for line in txt_file:
                    translated_text = evaluate(
                        line).numpy().decode("utf-8")
                    translated_text = translated_text.replace(" ' ", "'").replace(" .", ".")\
                        .replace(" ?", "?").replace(" !", "!")\
                        .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ")\
                        .replace(" ( ", " (").replace(" ) ", ") ").replace(" , ", ", ")
                    print(translated_text)
        else:
            lines = []
            with open(self.file) as txt_file:
                for line in txt_file:
                    translated_text = evaluate(
                        line).numpy().decode("utf-8")
                    translated_text = translated_text.replace(" ' ", "'").replace(" .", ".")\
                        .replace(" ?", "?").replace(" !", "!")\
                        .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ")\
                        .replace(" ( ", " (").replace(" ) ", ") ").replace(" , ", ", ")
                    lines.append(translated_text)
            return preprocessor.writeTotxt(self.ouput, lines)
