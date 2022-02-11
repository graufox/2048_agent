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
        self,
        conv_filters=128,
        conv_dropout=0.2,
        dense_units=1024,
        dense_dropout=0.5,
        kernel_size=(3, 3)
    ):
        super().__init__()

        self.preproc = Conv2DStack(
            filters=conv_filters,
            kernel_size=(1, 1),
            dropout_rate=0.
        )
        self.conv = Conv2DStack(
            filters=conv_filters,
            kernel_size=kernel_size,
            dropout_rate=conv_dropout
        )
        self.flatten = layers.Flatten()
        self.dense = DenseStack(units=dense_units, dropout_rate=dense_dropout)
        self.compute_unmasked_logQ = DenseStack(units=4, dropout_rate=0.)

    def call(self, inputs, training=False):
        observation, available_moves = inputs
        x = self.preproc(observation)
        x = self.conv(x, training=training)
        x = self.flatten(x)
        x = self.dense(x, training=training)
        unmasked_logQ = self.compute_unmasked_logQ(x)
        Q = tf.math.exp(unmasked_logQ) * available_moves
        self.add_loss(1e-1 * tf.reduce_max(Q)**2)
        # TODO: loss definition
        # # loss = tf.reduce_sum(-log_pickedQ * reward_)
        # # loss += tf.reduce_sum(tf.abs(Qout - nextQ))
        return Q

    @tf.function
    def train_step(self, x, picked_action, reward, targetQ):
        with tf.GradientTape() as tape:
            Q = self(x, training=True)
            loss_value = self.compiled_loss(targetQ, Q)
            pickedQ = tf.gather(targetQ, picked_action, batch_dims=1)
            loss_value -= tf.math.log(pickedQ + 1e-12) * reward
        grads = tape.gradient(loss_value, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return loss_value
