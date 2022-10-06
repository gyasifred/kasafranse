from kasafranse.preprocessing import Preprocessing
from kasafranse.transformer_layers import Encoder, Decoder
import tensorflow as tf
import tensorflow_text

# instatiate Preprocessing
preprocessor = Preprocessing()

# Build full Transformer


# Build full Transformer
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        self.decoder = Decoder(
            num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        # (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(inp, training, enc_padding_mask)
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        # (batch_size, tar_seq_len, target_vocab_size)
        final_output = self.final_layer(dec_output)
        return final_output, attention_weights


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
