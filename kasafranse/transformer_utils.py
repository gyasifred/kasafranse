import tensorflow as tf


class ProcessBatch:
    def __init__(self, input_processor, output_processor, max_tokens):
        self.input_processor = input_processor
        self.output_processor = output_processor
        self.max_tokens = max_tokens

    def prepare_batch(self, l1, l2):
        l1 = self.input_processor.tokenize(l1)      # Output is ragged.
        l1 = l1[:, :self.max_tokens]    # Trim to MAX_TOKENS.
        l1 = l1.to_tensor()  # Convert to 0-padded dense Tensor

        l2 = self.output_processor.tokenize(l2)
        l2 = l2[:, :(self.max_tokens+1)]
        l2 = l2[:, :-1].to_tensor()  # Drop the [END] tokens
        return l1, l2

    def make_batches(self, ds, BUFFER_SIZE, BATCH_SIZE):
        return (
            ds
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .map(self.prepare_batch, tf.data.AUTOTUNE)
            .prefetch(buffer_size=tf.data.AUTOTUNE))

    def writebatch(self, batches):
        # Write the test batch to file.
        # This will come in handy later if you prefer to translate from txt file.
        # and estimate the translator [BLEU](https://aclanthology.org/P02-1040.pdf) score.
        # The translator will run to error for any for any sentence with the shape greater
        # than the MAX_TOKENS used for training.
        # It adivisable to use the trimmed sentences for testting to avoid such occurrance
        lang_1 = []
        lang_2 = []

        for input_lang_batches, output_lang_batches in batches:
            for i in self.input_processor.detokenize(input_lang_batches):
                lang_1.append(i.numpy().decode("utf-8"))

            for j in self.output_processor.detokenize(output_lang_batches):
                lang_2.append(j.numpy().decode("utf-8"))
        return lang_1, lang_2


# Set learning rate schedule

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


def write_vocab_file(filepath, vocab):
    with open(filepath, 'w') as f:
        for token in vocab:
            print(token, file=f)
