import tensorflow as tf
from tensorflow.keras import layers


class DenseStack(layers.Layer):
    def __init__(
        self,
        units=1024,
        activation=tf.nn.leaky_relu,
        dropout_rate=0.2,
    ):
        super().__init__()
        self.dense_layer = layers.Dense(
            units=units, activation=activation
        )
        self.bn_layer = layers.BatchNormalization()
        self.dropout_layer = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        x = self.dense_layer(inputs)
        x_bn = self.bn_layer(x, training=training)
        x_do = self.dropout_layer(x_bn, training=training)
        return x_do


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


class ReinforcementAgent(tf.keras.models.Model):
    def __init__(
        self
    ):
        super().__init__()
        self.preproc = Conv2DStack(kernel_size=(1, 1), dropout_rate=0.)
        self.conv = Conv2DStack(kernel_size=(3, 3), dropout_rate=0.5)
        self.flatten = layers.Flatten()
        self.dense = DenseStack(units=1024, dropout_rate=0.5)
        self.compute_unmasked_logQ = DenseStack(units=4, dropout_rate=0.)

    def call(self, inputs, training=False):
        observation, available_moves = inputs
        x = self.preproc(observation)
        x = self.conv(x, training=training)
        x = self.flatten(x)
        x = self.dense(x, training=training)
        unmasked_logQ = self.compute_unmasked_logQ(x)
        Q = tf.math.exp(unmasked_logQ) * available_moves
        self.add_loss(tf.reduce_max(Q)**2)
        return Q

    @tf.function
    def train_step(self, x, targetQ):
        with tf.GradientTape() as tape:
            Q = self(x, training=True)
            loss_value = self.compiled_loss(targetQ, Q)
        grads = tape.gradient(loss_value, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return loss_value
