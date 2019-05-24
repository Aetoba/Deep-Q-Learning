# import os
import tensorflow as tf

class Q_cnn:
    """
    An estimator network for the Q-table.
    """

    def __init__(self, learning_rate, n_actions, name='Q'):
        self.state_size = [105, 80, 4]

        with tf.variable_scope(name):
            # Input layer (n, 84, 84, 4)
            self.input_layer = tf.placeholder(tf.float32, [None, *self.state_size], name="input_layer")
            # Target for training (n, 1)
            self.target = tf.placeholder(tf.float32, [None], name="target")
            # Selects an action
            self.action = tf.placeholder(tf.float32, [None, n_actions], name="action")

            # Normalize inputs
            self.normed = tf.map_fn(lambda x: x/255, self.input_layer)

            # Convolutional layer #1 (21, 21, 16)
            self.conv1 = tf.layers.conv2d(
                inputs=self.normed,
                filters=16,
                kernel_size=[8, 8],
                strides=[4, 4],
                padding="same",
                activation=tf.nn.relu)

            # Convolutional layer #2 (11, 11, 32)
            self.conv2 = tf.layers.conv2d(
                inputs=self.conv1,
                filters=32,
                kernel_size=[4, 4],
                strides=[2, 2],
                padding="same",
                activation=tf.nn.relu)

            # Flatten layer #1 (3872)
            self.flatten1 = tf.layers.flatten(
                inputs=self.conv2
            )         

            # Fully connected layer #1 (256)
            self.dense1 = tf.layers.dense(
                inputs=self.flatten1,
                units=256,
                activation=tf.nn.relu
            )

            # Fully connected layer #2 (n_actions)
            self.dense2 = tf.layers.dense(
                inputs=self.dense1,
                units=n_actions
            )

            self.output = self.dense2
            self.chosen = tf.boolean_mask(self.output, self.action)
            self.loss = tf.losses.mean_squared_error(self.chosen, self.target)
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss)
            # self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
