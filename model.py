from tensorflow import keras
import tensorflow as tf


def stack_dict(inputs, fun=tf.stack):
    values = []
    for key in sorted(inputs.keys()):
        values.append(tf.cast(inputs[key], tf.float32))

    return fun(values, axis=-1)


class MyModel(tf.keras.Model):
    def __init__(self):
        # Create all the internal layers in init.
        super().__init__(self)

        self.normalizer = tf.keras.layers.Normalization(axis=-1)

        self.seq = tf.keras.Sequential([
            self.normalizer,
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

    def adapt(self, inputs):
        # Stach the inputs and `adapt` the normalization layer.
        inputs = stack_dict(inputs)
        self.normalizer.adapt(inputs)

    def call(self, inputs):
        # Stack the inputs
        inputs = stack_dict(inputs)
        # Run them through all the layers.
        result = self.seq(inputs)

        return result
