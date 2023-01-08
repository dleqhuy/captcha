import tensorflow as tf
from tensorflow.keras import layers

class CTCDecoder():
    def __init__(self, vocab_path):
        with open(vocab_path) as file:
            vocab = [line.rstrip() for line in file]
        self.char_to_num = layers.StringLookup(vocabulary=vocab, mask_token=None,)

        self.num_to_char = layers.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True
        )

    def detokenize(self, x):
        x = tf.RaggedTensor.from_sparse(x)
        x = tf.ragged.map_flat_values(self.num_to_char, x)
        strings = tf.strings.reduce_join(x, axis=1)
        return strings


class CTCGreedyDecoder(CTCDecoder):
    def __init__(self, table_path, merge_repeated=True):
        super().__init__(table_path)
        self.merge_repeated = merge_repeated
        
    def call(self, inputs):
        input_shape = tf.shape(inputs)
        sequence_length = tf.fill([input_shape[0]], input_shape[1])
        decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(
            tf.transpose(inputs, perm=[1, 0, 2]), 
            sequence_length,
            self.merge_repeated)
        strings = self.detokenize(decoded[0])
        labels = tf.cast(decoded[0], tf.int32)
        loss = tf.nn.ctc_loss(
            labels=labels,
            logits=inputs,
            label_length=None,
            logit_length=sequence_length,
            logits_time_major=False,
            blank_index=-1)
        probability = tf.math.exp(-loss)
        return strings, probability


class CTCBeamSearchDecoder(CTCDecoder):
    def __init__(self, table_path, beam_width=100, top_paths=1):
        super().__init__(table_path)
        self.beam_width = beam_width
        self.top_paths = top_paths
        
    def call(self, inputs):
        input_shape = tf.shape(inputs)
        decoded, log_probability = tf.nn.ctc_beam_search_decoder(
            tf.transpose(inputs, perm=[1, 0, 2]), 
            tf.fill([input_shape[0]], input_shape[1]),
            self.beam_width, 
            self.top_paths)
        strings = []
        for i in range(self.top_paths):
            strings.append(self.detokenize(decoded[i]))
        strings = tf.concat(strings, 1)
        probability = tf.math.exp(log_probability)
        return strings, probability