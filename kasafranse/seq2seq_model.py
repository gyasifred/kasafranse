import tensorflow as tf
from kasafranse.seq2seq_layer import DecoderInput
from kasafranse.preprocessing import Preprocessing
import numpy as np

preprocessor = Preprocessing()


class Translator(tf.Module):

    def __init__(self, encoder, decoder, input_text_processor,
                 output_text_processor):
        self.encoder = encoder
        self.decoder = decoder
        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor

        self.output_token_string_from_index = (
            tf.keras.layers.StringLookup(
                vocabulary=output_text_processor.get_vocabulary(),
                mask_token='',
                invert=True))

        # The output should never generate padding, unknown, or start.
        index_from_string = tf.keras.layers.StringLookup(
            vocabulary=output_text_processor.get_vocabulary(), mask_token='')
        token_mask_ids = index_from_string(['', '[UNK]', '[START]']).numpy()

        token_mask = np.zeros(
            [index_from_string.vocabulary_size()], dtype=np.bool)
        token_mask[np.array(token_mask_ids)] = True
        self.token_mask = token_mask

        self.start_token = index_from_string(tf.constant('[START]'))
        self.end_token = index_from_string(tf.constant('[END]'))

    def tokens_to_text(self, result_tokens):
        result_text_tokens = self.output_token_string_from_index(
            result_tokens)
        result_text = tf.strings.reduce_join(result_text_tokens,
                                             axis=1, separator=' ')
        result_text = tf.strings.strip(result_text)
        return result_text

    def sample(self, logits, temperature):
        token_mask = self.token_mask[tf.newaxis, tf.newaxis, :]
        logits = tf.where(self.token_mask, -np.inf, logits)

        if temperature == 0.0:
            new_tokens = tf.argmax(logits, axis=-1)
        else:
            logits = tf.squeeze(logits, axis=1)
            new_tokens = tf.random.categorical(logits/temperature,
                                               num_samples=1)

        return new_tokens

    def translate(self,
                  input_text, *,
                  max_length=50,
                  return_attention=True,
                  temperature=1.0):
        batch_size = tf.shape(input_text)[0]
        input_tokens = self.input_text_processor(input_text)
        enc_output, enc_state = self.encoder(input_tokens)

        dec_state = enc_state
        new_tokens = tf.fill([batch_size, 1], self.start_token)

        result_tokens = []
        attention = []
        done = tf.zeros([batch_size, 1], dtype=tf.bool)

        for _ in range(max_length):
            dec_input = DecoderInput(new_tokens=new_tokens,
                                     enc_output=enc_output,
                                     mask=(input_tokens != 0))

            dec_result, dec_state = self.decoder(dec_input, state=dec_state)

            attention.append(dec_result.attention_weights)

            new_tokens = self.sample(dec_result.logits, temperature)

            # If a sequence produces an `end_token`, set it `done`
            done = done | (new_tokens == self.end_token)
            # Once a sequence is done it only produces 0-padding.
            new_tokens = tf.where(done, tf.constant(
                0, dtype=tf.int64), new_tokens)

            # Collect the generated tokens
            result_tokens.append(new_tokens)

            if tf.executing_eagerly() and tf.reduce_all(done):
                break

        # Convert the list of generates token ids to a list of strings.
        result_tokens = tf.concat(result_tokens, axis=-1)
        result_text = self.tokens_to_text(result_tokens)

        if return_attention:
            attention_stack = tf.concat(attention, axis=1)
            return {'text': result_text, 'attention': attention_stack}
        else:
            return {'text': result_text}

    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None])])
    def __call__(self, input_text):
        return self.translate(input_text)


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
                    input = tf.constant([line])
                    result = evaluate(input)
                    translated_text = result['text'][0].numpy().decode("utf-8")
                    translated_text = translated_text.replace(" ' ", "'").replace(" .", ".")\
                        .replace(" ?", "?").replace(" !", "!")\
                        .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ")\
                        .replace(" ( ", " (").replace(" ) ", ") ").replace(" , ", ", ")
                    print(translated_text)
        else:
            lines = []
            with open(self.file) as txt_file:
                for line in txt_file:
                    input = tf.constant([line])
                    result = evaluate(input)
                    translated_text = result['text'][0].numpy().decode("utf-8")
                    translated_text = translated_text.replace(" ' ", "'").replace(" .", ".")\
                        .replace(" ?", "?").replace(" !", "!")\
                        .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ")\
                        .replace(" ( ", " (").replace(" ) ", ") ").replace(" , ", ", ")
                    lines.append(translated_text)
            return preprocessor.writeTotxt(self.ouput, lines)
