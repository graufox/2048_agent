import tensorflow as tf
from icecream import ic
from tensorflow.keras import layers
from tensorflow.keras import constraints


class DenseStack(layers.Layer):
    """Dense layer, followed by batch normalization and dropout."""

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
        )
        self.norm_layer = layers.LayerNormalization()
        self.dropout_layer = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        x = self.dense_layer(inputs)
        x = self.norm_layer(x, training=training)
        x = self.dropout_layer(x, training=training)
        return x


class Conv2DStack(layers.Layer):
    """2D convolution layer, followed by batch normalization and spatial dropout."""

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
        )
        self.norm_layer = layers.LayerNormalization()
        self.dropout_layer = layers.SpatialDropout2D(dropout_rate)

    def call(self, inputs, training=False):
        x = self.conv_layer(inputs)
        x = self.norm_layer(x, training=training)
        x = self.dropout_layer(x, training=training)
        return x


class Conv3DStack(layers.Layer):
    """3D convolution layer, followed by batch normalization and spatial dropout."""

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
        )
        self.norm_layer = layers.LayerNormalization()
        self.dropout_layer = layers.SpatialDropout3D(dropout_rate)

    def call(self, inputs, training=False):
        x = self.conv_layer(inputs)
        x = self.norm_layer(x, training=training)
        x = self.dropout_layer(x, training=training)
        return x


class ConvModel(tf.keras.models.Model):
    def __init__(
        self,
        conv_filters=64,
        conv_dropout=0.2,
        num_conv_stacks=3,
        dense_units=(1024,),
        dense_dropout=0.5,
        output_units=4,
        kernel_size=(3, 3),
        output_activation=None,
        use_preprocessing=True,
    ):
        super().__init__()

        self.preproc = None
        if use_preprocessing:
            self.preproc = Conv2DStack(
                filters=conv_filters,
                kernel_size=(1, 1),
                dropout_rate=0.0,
                padding="same",
            )
        self.convs = []
        for i in range(num_conv_stacks):
            self.convs.append(
                Conv2DStack(
                    filters=conv_filters,
                    kernel_size=kernel_size,
                    dropout_rate=conv_dropout,
                    padding="same",
                )
            )
        self.conv_flatten = layers.Flatten()
        self.dense_layers = []
        for units in dense_units:
            self.dense_layers.append(
                DenseStack(
                    units=units,
                    dropout_rate=dense_dropout,
                )
            )
        self.output_layer = layers.Dense(
            units=output_units,
            activation=output_activation,
        )

    def call(self, x, training=False):
        if self.preproc is not None:
            x = self.preproc(x, training=training)
        for conv in self.convs:
            x = x + conv(x, training=training)
        x = self.conv_flatten(x)
        for layer in self.dense_layers:
            x = layer(x, training=training)
        output = self.output_layer(x)
        return output


class Conv1DModel(tf.keras.models.Model):
    def __init__(
        self,
        conv_filters=64,
        conv_dropout=0.2,
        num_conv_stacks=3,
        dense_units=(1024,),
        dense_dropout=0.5,
        output_units=4,
        kernel_size=(3, 3),
        output_activation=None,
        use_preprocessing=True,
    ):
        super().__init__()

        self.preproc = None
        if use_preprocessing:
            self.preproc = Conv2DStack(
                filters=conv_filters,
                kernel_size=(1, 1),
                dropout_rate=0.0,
                padding="same",
            )
        self.convs = []
        for i in range(num_conv_stacks):
            self.convs_x.append(
                Conv1DStack(
                    filters=conv_filters,
                    kernel_size=kernel_size,
                    dropout_rate=conv_dropout,
                    padding="same",
                )
            )
            self.convs_y.append(
                Conv1DStack(
                    filters=conv_filters,
                    kernel_size=kernel_size,
                    dropout_rate=conv_dropout,
                    padding="same",
                )
            )
        self.conv_flatten = layers.Flatten()
        self.dense_layers = []
        for units in dense_units:
            self.dense_layers.append(
                DenseStack(
                    units=units,
                    dropout_rate=dense_dropout,
                )
            )
        self.output_layer = layers.Dense(
            units=output_units,
            activation=output_activation,
        )

    def call(self, x, training=False):
        if self.preproc is not None:
            x = self.preproc(x, training=training)
        for conv in self.convs:
            x = x + conv(x, training=training)
        x = self.conv_flatten(x)
        for layer in self.dense_layers:
            x = layer(x, training=training)
        output = self.output_layer(x)
        return output


class ReinforcementAgent(tf.keras.models.Model):
    """
    Deep Q-Network Reinforcement Learning Agent

    Deep Q-Network Reinforcement Learning Agent, implemented in TensorFlow 2.
        The network is based on a deep residual convolution network. The output
        are one of 4 actions, which are associated with up, right, down, and
        left respectfully. The output of the network is the estimated Q-value
        for the actions at the given state.
    """

    def __init__(
        self,
        conv_filters=128,
        conv_dropout=0.2,
        num_conv_stacks=3,
        dense_units=(1024,),
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
            output_units=output_units,
            output_activation=None,
        )

    @tf.function
    def call(self, inputs, training=False):
        observation, available_moves = inputs
        logQ = self.base_model(observation, training=training)
        Q = tf.nn.softplus(logQ)
        Q_masked = Q * available_moves
        action = tf.argmax(Q_masked, axis=1)
        return Q, action

    @tf.function
    def train_step(self, x, targetQ):
        with tf.GradientTape() as tape:
            Q, _ = self(x, training=True)
            loss_value = tf.keras.losses.Huber()(targetQ, Q)
            grads = tape.gradient(loss_value, self.trainable_weights)
            grads = [
                (None if gradient is None else tf.clip_by_norm(gradient, 1.0))
                for gradient in grads
            ]
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return loss_value


class LinearReinforcementAgent(tf.keras.models.Model):
    """
    Deep Q-Network Reinforcement Learning Agent

    Deep Q-Network Reinforcement Learning Agent, implemented in TensorFlow 2.
        The network is based on a deep residual convolution network. The output
        are one of 4 actions, which are associated with up, right, down, and
        left respectfully. The output of the network is the estimated Q-value
        for the actions at the given state.
    """

    def __init__(
        self,
        conv_filters=128,
        conv_dropout=0.2,
        num_conv_stacks=3,
        dense_units=(1024,),
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
            output_units=output_units,
            output_activation=None,
        )

    @tf.function
    def call(self, inputs, training=False):
        observation, available_moves = inputs
        logQ = self.base_model(observation, training=training)
        Q = tf.nn.softplus(logQ)
        Q_masked = Q * available_moves
        action = tf.argmax(Q_masked, axis=1)
        return Q, action

    @tf.function
    def train_step(self, x, targetQ):
        with tf.GradientTape() as tape:
            Q, _ = self(x, training=True)
            loss_value = tf.keras.losses.Huber()(targetQ, Q)
            grads = tape.gradient(loss_value, self.trainable_weights)
            grads = [
                (None if gradient is None else tf.clip_by_norm(gradient, 1.0))
                for gradient in grads
            ]
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return loss_value


class RotationalReinforcementAgent(tf.keras.models.Model):
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
        use_preprocessing=True,
    ):
        super().__init__()

        self.base_model = ConvModel(
            conv_filters=conv_filters,
            conv_dropout=conv_dropout,
            num_conv_stacks=num_conv_stacks,
            dense_units=dense_units,
            dense_dropout=dense_dropout,
            kernel_size=kernel_size,
            output_units=output_units,
            output_activation=None,
            use_preprocessing=use_preprocessing,
        )

    @tf.function
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
        Q = tf.nn.leaky_relu(logQ, alpha=0.1)
        Q_masked = Q * available_moves
        action = tf.argmax(Q_masked, axis=1)
        return Q, action

    @tf.function
    def train_step(self, x, targetQ):
        with tf.GradientTape() as tape:
            Q, _ = self(x, training=True)
            loss_value = tf.keras.losses.Huber()(targetQ, Q)
            grads = tape.gradient(loss_value, self.trainable_weights)
            grads = [
                (None if gradient is None else tf.clip_by_norm(gradient, 0.3))
                for gradient in grads
            ]
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return loss_value
