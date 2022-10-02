import tensorflow as tf
import tensorflow_text as tf_text
import pickle


class ProcessTokenizer:
    def __init__(self) -> None:
        pass

    def tf_start_and_end_tokens(self, text):
        # Strip whitespace.
        text = tf.strings.strip(text)

        text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
        return text

    def build_tokenizer(self, ds, max_vocab_size):
        # Process twi as input
        tmp = tf.keras.layers.TextVectorization(
            standardize=self.tf_start_and_end_tokens,
            max_tokens=max_vocab_size)
        tmp.adapt(ds)
        return tmp

    def savetokenizer(self, filepath, tokenizer):
        return pickle.dump({'config': tokenizer.get_config(),
                            'vocabulary': tokenizer.get_vocabulary(),
                            'weights': tokenizer.get_weights()},
                           open(filepath, "wb"))

    def loadtokenizer(self, filepath):
        tmp = pickle.load(open(filepath, "rb"))
        temp = tf.keras.layers.TextVectorization.from_config(tmp['config'])
        # You have to call `adapt` with some dummy data (BUG in Keras)
        temp.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
        temp.set_weights(tmp['weights'])
        temp.set_vocabulary(tmp['vocabulary'])
        return temp
