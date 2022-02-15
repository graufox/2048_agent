import tensorflow as tf
from icecream import ic
from tensorflow.keras import layers
from tensorflow.keras import constraints


class DenseStack(layers.Layer):
    def __init__(
        self,
        units=1024,
        activation=tf.nn.leaky_relu,
        dropout_rate=0.2,
    ):
        super().__init__()
        self.dense_layer = layers.Dense(
            units=units,
            activation=activation,
            kernel_constraint=constraints.MaxNorm(2.0),
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
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            padding=padding,
            kernel_constraint=constraints.MaxNorm(2.0, axis=[0, 1, 2]),
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
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            padding=padding,
            kernel_constraint=constraints.MaxNorm(2.0, axis=[0, 1, 2]),
        )
        self.bn_layer = layers.BatchNormalization()
        self.dropout_layer = layers.SpatialDropout3D(dropout_rate)

    def call(self, inputs, training=False):
        x = self.conv_layer(inputs)
        x_bn = self.bn_layer(x, training=training)
        x_do = self.dropout_layer(x_bn, training=training)
        return x_do


class ConvModel(tf.keras.models.Model):
    def __init__(
        self,
        conv_filters=64,
        conv_dropout=0.2,
        num_conv_stacks=3,
        dense_units=1024,
        dense_dropout=0.5,
        output_units=4,
        kernel_size=(3, 3),
        output_activation=None,
        board_size=4,
        board_depth=16,
    ):
        super().__init__()

        self.preproc = Conv2DStack(
            filters=conv_filters,
            kernel_size=(1, 1),
            dropout_rate=0.,
            padding='same'
        )
        self.convs = []
        for i in range(num_conv_stacks):
            self.convs.append(
                Conv2DStack(
                    filters=conv_filters,
                    kernel_size=(3, 3),
                    dropout_rate=conv_dropout,
                    padding='same'
                )
            )
        self.conv_flatten = layers.Flatten()
        self.dense = DenseStack(
            units=dense_units,
            dropout_rate=dense_dropout,
        )
        self.output_layer = layers.Dense(
            units=output_units,
            activation=output_activation,
        )

    def call(self, inputs, training=False):
        x = self.preproc(inputs)
        for conv in self.convs:
            x = x + conv(x, training=training)
        x = self.conv_flatten(x)
        x = self.dense(x, training=training)
        output = self.output_layer(x)
        return output


class ReinforcementAgent(tf.keras.models.Model):
    """
    Deep Q-Network Reinforcement Learning Agent

    Deep Q-Network Reinforcement Learning Agent, implemented in TensorFlow 2.
        The network is based on a rotationally-symmetrized deep residual convo-
        lution network. The output are one of 4 actions, which are associated
        with up, right, down, and left respectfully. The output of the network
        is the estimated Q-value for the actions at the given state.
    """

    def __init__(
        self,
        conv_filters=128,
        conv_dropout=0.2,
        num_conv_stacks=3,
        dense_units=1024,
        output_units=4,
        dense_dropout=0.5,
        kernel_size=(3, 3),
    ):
        super().__init__()

        self.base_model = ConvModel(
            conv_filters=conv_filters,
            conv_dropout=conv_dropout,
            num_conv_stacks=num_conv_stacks,
            dense_units=dense_units,
            dense_dropout=dense_dropout,
            kernel_size=kernel_size,
            output_activation=None,
        )

    def call(self, inputs, training=False):

        observation, available_moves = inputs

        obs_0 = observation
        obs_90 = tf.image.rot90(observation, k=1)
        obs_180 = tf.image.rot90(observation, k=2)
        obs_270 = tf.image.rot90(observation, k=3)

        logQ_0 = self.base_model(obs_0, training=training)
        logQ_90 = self.base_model(obs_90, training=training)
        logQ_180 = self.base_model(obs_180, training=training)
        logQ_270 = self.base_model(obs_270, training=training)

        logQ = tf.reduce_mean(
            [
                logQ_0,
                tf.gather(logQ_90, [3, 0, 1, 2], axis=1),
                tf.gather(logQ_180, [2, 3, 0, 1], axis=1),
                tf.gather(logQ_270, [1, 2, 3, 0], axis=1),
            ],
            axis=0,
        )
        Q = tf.nn.softplus(logQ) * available_moves
        action = tf.argmax(Q, axis=1)
        return Q, action

    @tf.function
    def train_step(self, x, reward, targetQ):
        with tf.GradientTape() as tape:
            Q, action = self(x, training=True)
            loss_value = tf.reduce_mean((targetQ - Q) ** 2)
            selected_Q = tf.gather(Q, action)
            loss_value -= tf.math.log(selected_Q) * reward
        grads = tape.gradient(loss_value, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return loss_value
