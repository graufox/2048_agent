import tensorflow as tf
from tensorflow.keras import layers


class Conv2DStack(layers.Layer):
    def __init__(
        self,
        filters=32,
        kernel_size=(3, 3),
        activation=tf.nn.leaky_relu,
        padding="same",
        dropout_rate=0.2,
    ):
        super().__init__()
        self.conv_layer = layers.Conv2D(
            filters=filters, kernel_size=kernel_size, activation=activation, padding=padding
        )
        self.bn_layer = layers.BatchNormalization()
        self.dropout_layer = layers.SpatialDropout2D(dropout_rate)

    def call(self, inputs, training=False):
        x = self.conv_layer(inputs)
        x_bn = self.bn_layer(x, training=training)
        x_do = self.dropout_layer(x_bn, training=training)
        return x_do


class Conv3DStack(layers.Layer):
    def __init__(
        self,
        filters=32,
        kernel_size=(3, 3, 3),
        activation=tf.nn.leaky_relu,
        padding="same",
        dropout_rate=0.2,
    ):
        super().__init__()
        self.conv_layer = layers.Conv3D(
            filters=filters, kernel_size=kernel_size, activation=activation, padding=padding
        )
        self.bn_layer = layers.BatchNormalization()
        self.dropout_layer = layers.SpatialDropout3D(dropout_rate)

    def call(self, inputs, training=False):
        x = self.conv_layer(inputs)
        x_bn = self.bn_layer(x, training=training)
        x_do = self.dropout_layer(x_bn, training=training)
        return x_do
