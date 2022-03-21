import tensorflow as tf
from tensorflow.keras import layers


class ConvBlock(layers.Layer):

    def __init__(self, filters, kernel_size, strides, **kwargs_conv2d):
        super(ConvBlock, self).__init__(**kwargs_conv2d)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        kwargs_conv2d["filters"] = filters
        kwargs_conv2d["kernel_size"] = kernel_size
        kwargs_conv2d["strides"] = strides
        kwargs_conv2d["padding"] = 'same'

        block = [
            layers.Conv2D(**kwargs_conv2d),
            ChannelNorm(),
            layers.ReLU()]

        self.block = tf.keras.Sequential(layers=block)

    def call(self, inputs, **kwargs):
        return self.block(inputs, **kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
        })
        return config


class UpConvBlock(layers.Layer):

    def __init__(self, filters, kernel_size, strides, **kwargs_conv2d):
        super(UpConvBlock, self).__init__(**kwargs_conv2d)
        kwargs_conv2d["filters"] = filters
        kwargs_conv2d["kernel_size"] = kernel_size
        kwargs_conv2d["strides"] = strides
        kwargs_conv2d["padding"] = 'same'

        block = [
            layers.Conv2DTranspose(**kwargs_conv2d),
            ChannelNorm(),
            layers.ReLU()]

        self.block = tf.keras.Sequential(layers=block)

    def call(self, inputs, **kwargs):
        return self.block(inputs, **kwargs)


class ResidualBlock(layers.Layer):
    """Implement a residual block."""

    def __init__(self, filters, kernel_size, name=None, activation="relu", **kwargs_conv2d):
        """
        Instantiate layer.

        :param filters: int, number of filters, passed to the conv layers.
        :param kernel_size: int, kernel_size, passed to the conv layers.
        :param name: str, name of the layer.
        :param activation: function or string, resolved with keras.
        :param kwargs_conv2d: Additional arguments to be passed directly to Conv2D. E.g. 'padding'.
        """

        super(ResidualBlock, self).__init__()
        kwargs_conv2d["filters"] = filters
        kwargs_conv2d["kernel_size"] = kernel_size
        kwargs_conv2d["padding"] = 'same'

        block = [
            layers.Conv2D(**kwargs_conv2d),
            ChannelNorm(),
            layers.Activation(activation),
            layers.Conv2D(**kwargs_conv2d),
            ChannelNorm()]

        self.block = tf.keras.Sequential(name=name, layers=block)

    def call(self, inputs, **kwargs):
        return inputs + self.block(inputs, **kwargs)


# experimental
class ChannelNorm(tf.keras.layers.Layer):
    """
    Implement ChannelNorm
    based on https://github.com/tensorflow/compression/blob/d722a35e899326f340d4f76bac5e637b52dc547a/models/hific/archs.py#L215
    """

    def __init__(self,
                 epsilon: float = 1e-3,
                 center: bool = True,
                 scale: bool = True,
                 beta_initializer="zeros",
                 gamma_initializer="ones",
                 **kwargs):
        """
        Instantiate layer.

        :param epsilon:  For stability when normalizing.
        :param center: Whether to create and use a {beta}.
        :param scale: Whether to create and use a {gamma}.
        :param beta_initializer: Initializer for beta.
        :param gamma_initializer: Initializer for gamma.
        :param kwargs:
        """
        super(ChannelNorm, self).__init__(**kwargs)

        self.axis = -1
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)

    def build(self, input_shape):
        self._add_gamma_weight(input_shape)
        self._add_beta_weight(input_shape)
        self.built = True
        super().build(input_shape)

    def call(self, inputs, modulation=None):
        mean, variance = self._get_moments(inputs)
        # inputs = tf.Print(inputs, [mean, variance, self.beta, self.gamma], "NORM")
        return tf.nn.batch_normalization(
            inputs, mean, variance, self.beta, self.gamma, self.epsilon,
            name="normalize")

    def _get_moments(self, inputs):
        # Like tf.nn.moments but unbiased sample std. deviation.
        # Reduce over channels only.
        mean = tf.reduce_mean(inputs, [self.axis], keepdims=True, name="mean")
        variance = tf.reduce_sum(
            tf.math.squared_difference(inputs, tf.stop_gradient(mean)),
            [self.axis], keepdims=True, name="variance_sum")
        # Divide by N-1
        inputs_shape = tf.shape(inputs)
        counts = tf.reduce_prod([inputs_shape[ax] for ax in [self.axis]])
        variance /= (tf.cast(counts, tf.float32) - 1)
        return mean, variance

    def _add_gamma_weight(self, input_shape):
        dim = input_shape[self.axis]
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                name="gamma",
                initializer=self.gamma_initializer)
        else:
            self.gamma = None

    def _add_beta_weight(self, input_shape):
        dim = input_shape[self.axis]
        shape = (dim,)

        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                name="beta",
                initializer=self.beta_initializer)
        else:
            self.beta = None

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': self.beta_initializer,
            'gamma_initializer': self.gamma_initializer
        })
        return config