import tensorflow as tf
from kasafranse.seq2seq_layer import Encoder, Decoder, DecoderInput


class MaskedLoss(tf.keras.losses.Loss):
    def __init__(self):
        self.name = 'masked_loss'
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

    def __call__(self, y_true, y_pred):

        # Calculate the loss for each item in the batch.
        loss = self.loss(y_true, y_pred)

        # Mask off the losses on padding.
        mask = tf.cast(y_true != 0, tf.float32)
        loss *= mask

        # Return the total.
        return tf.reduce_sum(loss)


class TrainTranslator(tf.keras.Model):
    def __init__(self, embedding_dim, units,
                 input_text_processor,
                 output_text_processor,
                 use_tf_function=True):
        super().__init__()
        # Build the encoder and decoder
        encoder = Encoder(input_text_processor.vocabulary_size(),
                          embedding_dim, units)
        decoder = Decoder(output_text_processor.vocabulary_size(),
                          embedding_dim, units)

        self.encoder = encoder
        self.decoder = decoder
        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor
        self.use_tf_function = use_tf_function

    def train_step(self, inputs):
        # .shape_checker = ShapeChecker()
        if self.use_tf_function:
            return self._tf_train_step(inputs)
        else:
            return self._train_step(inputs)

    # Implement preprocessing step to:
    # Receive a batch of input_text, target_text from the tf.data.Dataset.
    # Convert those raw text inputs to token-embeddings and masks.
    def _preprocess(self, input_text, target_text):
        # Convert the text to token IDs
        input_tokens = self.input_text_processor(input_text)
        target_tokens = self.output_text_processor(target_text)
        # Convert IDs to masks.
        input_mask = input_tokens != 0

        target_mask = target_tokens != 0
        return input_tokens, input_mask, target_tokens, target_mask

    # the function The _train_step:
    # Run the encoder on the input_tokens to get the encoder_output and encoder_state.
    # Initialize the decoder state and loss.
    # Loop over the target_tokens:
    #   Run the decoder one step at a time.
    #   Calculate the loss for each step.
    # Accumulate the average loss.
    # Calculate the gradient of the loss and use the optimizer to apply updates to the model's trainable_variables.

    def _train_step(self, inputs):
        input_text, target_text = inputs

        (input_tokens, input_mask,
         target_tokens, target_mask) = self._preprocess(input_text, target_text)

        max_target_length = tf.shape(target_tokens)[1]

        with tf.GradientTape() as tape:
            # Encode the input
            enc_output, enc_state = self.encoder(input_tokens)

            # Initialize the decoder's state to the encoder's final state.
            # This only works if the encoder and decoder have the same number of
            # units.
            dec_state = enc_state
            loss = tf.constant(0.0)

            for t in tf.range(max_target_length-1):
                # Pass in two tokens from the target sequence:
                # 1. The current input to the decoder.
                # 2. The target for the decoder's next prediction.
                new_tokens = target_tokens[:, t:t+2]
                step_loss, dec_state = self._loop_step(new_tokens, input_mask,
                                                       enc_output, dec_state)
                loss = loss + step_loss

            # Average the loss over all non padding tokens.
            average_loss = loss / \
                tf.reduce_sum(tf.cast(target_mask, tf.float32))

        # Apply an optimization step
        variables = self.trainable_variables
        gradients = tape.gradient(average_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        # Return a dict mapping metric names to current value
        return {'batch_loss': average_loss}

    # The _loop_step method, added below, executes the decoder and calculates the incremental loss and new decoder state (dec_state).

    def _loop_step(self, new_tokens, input_mask, enc_output, dec_state):
        input_token, target_token = new_tokens[:, 0:1], new_tokens[:, 1:2]

        # Run the decoder one step.
        decoder_input = DecoderInput(new_tokens=input_token,
                                     enc_output=enc_output,
                                     mask=input_mask)

        dec_result, dec_state = self.decoder(decoder_input, state=dec_state)

        # `self.loss` returns the total for non-padded tokens
        y = target_token
        y_pred = dec_result.logits
        step_loss = self.loss(y, y_pred)

        return step_loss, dec_state

    @tf.function(input_signature=[[tf.TensorSpec(dtype=tf.string, shape=[None]),
                                   tf.TensorSpec(dtype=tf.string, shape=[None])]])
    def _tf_train_step(self, inputs):
        return self._train_step(inputs)


class BatchLogs(tf.keras.callbacks.Callback):
    def __init__(self, key):
        self.key = key
        self.logs = []

    def on_train_batch_end(self, n, logs):
        self.logs.append(logs[self.key])


batch_loss = BatchLogs('batch_loss')


# class ProcessBatch:
#     def __init__(self) -> None:
#         pass

#     def make_batches(self, inp, targ, BUFFER_SIZE, BATCH_SIZE):
#         return tf.data.Dataset\
#             .from_tensor_slices((inp, targ))\
#             .shuffle(BUFFER_SIZE)\
#             .batch(BATCH_SIZE)

#     def writebatch(self, batches):
#         lang_1 = []
#         lang_2 = []

#         for input_lang_batches, output_lang_batches in batches:
#             for line in input_lang_batches.numpy():
#                 lang_1.append(line.decode("utf-8"))

#             for line in output_lang_batches.numpy():
#                 lang_2.append(line.decode("utf-8"))
#         return lang_1, lang_2


class ProcessBatch:
    def __init__(self, input_processor, output_processor, max_tokens):
        self.input_processor = input_processor
        self.output_processor = output_processor
        self.max_tokens = max_tokens

    def prepare_batch(self, l1, l2):
        l1 = self.input_processor(l1)      # Output is ragged.
        l1 = l1[:, :self.max_tokens]    # Trim to MAX_TOKENS.
        l1 = l1.to_tensor()  # Convert to 0-padded dense Tensor

        l2 = self.output_processor(l2)
        l2 = l2[:, :(self.max_tokens+1)]
        l2 = l2.to_tensor() 
        return l1, l2

    def make_batches(self, ds, BUFFER_SIZE, BATCH_SIZE):
        return (
            ds
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .map(self.prepare_batch, tf.data.AUTOTUNE)
            .prefetch(buffer_size=tf.data.AUTOTUNE))

    def writebatch(self, batches):
        lang_1 = []
        lang_2 = []

        for input_lang_batches, output_lang_batches in batches:
            for line in input_lang_batches.numpy():
                lang_1.append(line.decode("utf-8"))

            for line in output_lang_batches.numpy():
                lang_2.append(line.decode("utf-8"))
        return lang_1, lang_2
