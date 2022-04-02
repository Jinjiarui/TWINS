import tensorflow as tf


class FMLayer(tf.keras.layers.Layer):
    def __init__(self, n, k, w1=None, v=None, point_wise=True):
        super(FMLayer, self).__init__()
        self.point_wise = point_wise
        self.W0 = tf.Variable(tf.random_normal([1]), name='W0')
        self.W1 = w1  # 前两项线性层
        self.V = v  # 交互矩阵
        if self.W1 is None:
            self.W1 = MyEmbeddingLayer(tf.Variable(tf.random_normal([n, 1])))
        if self.V is None:
            self.V = MyEmbeddingLayer(tf.Variable(tf.random_normal([n, k])))

    def call(self, inputs, **kwargs):
        # inputs:(bs,n),n is the num of fields
        linear_part = tf.reduce_sum(self.W1(inputs), axis=-2) + self.W0  # (bs,1)
        x = self.V(inputs)  # (bs,n,k)
        interaction_part_1 = tf.pow(tf.reduce_sum(x, -2), 2)  # (bs,k)
        interaction_part_2 = tf.reduce_sum(tf.pow(x, 2), -2)
        product_part = interaction_part_1 - interaction_part_2
        if self.point_wise:
            product_part = tf.reduce_sum(product_part, -1, keepdims=True)  # (bs,1)
        output = linear_part + 0.5 * product_part
        return output


class PredictMLP(tf.keras.layers.Layer):
    def __init__(self, prediction_hidden_width, keep_prob):
        super(PredictMLP, self).__init__()
        self.drop_out = tf.keras.layers.Dropout(rate=1 - keep_prob)
        self.activate = tf.keras.layers.LeakyReLU()
        self.prediction_mlp = tf.keras.Sequential()
        for width in prediction_hidden_width:
            self.prediction_mlp.add(tf.keras.layers.Dense(units=width, use_bias=True))
            self.prediction_mlp.add(self.drop_out)
            self.prediction_mlp.add(self.activate)

    def call(self, inputs, **kwargs):
        return self.prediction_mlp(inputs)


class MyEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, variable):
        super(MyEmbeddingLayer, self).__init__()
        self.embedding = variable

    def call(self, inputs, **kwargs):
        if not (isinstance(inputs, list) or isinstance(inputs, tuple)):
            return tf.nn.embedding_lookup(self.embedding, inputs)
        else:
            inputs_id, inputs_value = inputs
            return tf.nn.embedding_lookup(self.embedding, inputs_id) * tf.expand_dims(inputs_value, -1)


class NARM(tf.keras.layers.Layer):
    def __init__(self, n_hidden, dropout):
        super(NARM, self).__init__()
        self.n_hidden = n_hidden
        self.A1 = tf.keras.layers.Dense(units=n_hidden, use_bias=False, name='A1')
        self.A2 = tf.keras.layers.Dense(units=n_hidden, use_bias=False, name='A2')
        self.v = tf.keras.layers.Dense(units=1, use_bias=False, name='v')
        self.GRU_g = tf.keras.layers.GRU(units=n_hidden, return_sequences=True, dropout=dropout)
        self.GRU_l = tf.keras.layers.GRU(units=n_hidden, return_sequences=True, dropout=dropout)

    def call(self, inputs, **kwargs):
        mask = kwargs['mask']
        h_g = self.GRU_g(inputs, mask=mask)  # (bs,max_len,hidden)
        h_l = self.GRU_l(inputs, mask=mask)
        h_l_1 = self.A1(h_l)
        h_l_2 = self.A2(h_g)
        c_l = tf.map_fn(lambda i: tf.reduce_sum(
            self.v(tf.sigmoid(h_l_1[:, :i + 1] + h_l_2[:, i:i + 1])) * h_l[:, :i + 1], axis=-2),
                        tf.range(tf.shape(inputs)[1]), dtype=inputs.dtype)
        c_l = tf.transpose(c_l, [1, 0, 2])
        c_t = tf.concat([h_g, c_l], axis=-1)
        return c_t


class ESMM(tf.keras.layers.Layer):
    def __init__(self, n_hidden, dropout):
        super(ESMM, self).__init__()
        self.n_hidden = n_hidden
        self.dropout = tf.keras.layers.Dropout(rate=dropout)
        self.A1 = tf.keras.layers.Dense(units=n_hidden, use_bias=True)
        self.A2 = tf.keras.layers.Dense(units=n_hidden, use_bias=True)

    def call(self, inputs, **kwargs):
        e1 = self.dropout(self.A1(inputs))
        e2 = self.dropout(self.A2(inputs))
        return e1 * e2


class DeepFM(tf.keras.layers.Layer):
    def __init__(self, w1, v, prediction_hidden_width, keep_prob):
        super(DeepFM, self).__init__()
        self.fm = FMLayer(0, 0, w1, v, point_wise=True)
        self.deep = PredictMLP(prediction_hidden_width, keep_prob=keep_prob)
        self.V = v

    def call(self, inputs, **kwargs):
        embed_inputs = tf.reduce_sum(self.V(inputs), axis=-2)
        return self.fm(inputs) + self.deep(embed_inputs)


class DIN(tf.keras.layers.Layer):
    def __init__(self, dropout):
        super(DIN, self).__init__()
        self.activation_layer = tf.keras.Sequential()
        self.activation_layer.add(tf.keras.layers.Dense(36))
        self.activation_layer.add(tf.keras.layers.LeakyReLU())
        self.activation_layer.add(tf.keras.layers.Dropout(dropout))
        self.activation_layer.add(tf.keras.layers.Dense(1))

    def call(self, inputs, **kwargs):
        behaviors, candidate = inputs
        candidate = tf.tile(tf.expand_dims(candidate, -2), [1, tf.shape(behaviors)[-2], 1])
        activation_weight = self.activation_layer(tf.concat([behaviors, candidate, behaviors * candidate], -1))
        behaviors *= activation_weight
        return behaviors


class AUGRUCell(tf.contrib.rnn.BasicRNNCell):
    def __init__(self, hidden_size, **kwargs):
        super(AUGRUCell, self).__init__(num_units=hidden_size, **kwargs)
        self.dense_layer = {}
        self.hidden_size = hidden_size

    @property
    def output_size(self):
        return self._num_units

    def get_dense_name(self):
        return ['xu', 'hu', 'xr', 'hr', 'xg', 'hg']

    def build(self, input_shape):
        dense_layer_name = self.get_dense_name()
        for i in dense_layer_name:
            if i[0] == 'x':
                self.dense_layer[i] = tf.layers.Dense(units=self.hidden_size,
                                                      use_bias=True,
                                                      kernel_initializer='random_normal', name=i)
            else:
                self.dense_layer[i] = tf.layers.Dense(units=self.hidden_size,
                                                      use_bias=False,
                                                      kernel_initializer='random_normal', name=i)
        self.built = True

    def call(self, inputs, h):
        inputs, attention = inputs[:, :-1], inputs[:, -1:]
        u = tf.sigmoid(self.dense_layer['xu'](inputs) + self.dense_layer['hu'](h))
        u = u * attention
        r = tf.sigmoid(self.dense_layer['xr'](inputs) + self.dense_layer['hr'](h))
        g = tf.tanh(self.dense_layer['xg'](inputs) + r * self.dense_layer['hg'](h))
        h = (1 - u) * h + u * g
        return h, h


class DIEN(tf.keras.layers.Layer):
    def __init__(self, hidden_size, dropout):
        super(DIEN, self).__init__()
        self.gru = tf.keras.layers.GRU(hidden_size, dropout=dropout, return_sequences=True)
        self.W = tf.keras.layers.Dense(hidden_size)
        cell = AUGRUCell(hidden_size)
        self.augru = tf.keras.layers.RNN(cell, name="RNN", trainable=True, return_sequences=True)

    def call(self, inputs, **kwargs):
        mask = kwargs['mask']
        behaviors, candidate = inputs
        behaviors = self.gru(behaviors, mask=mask)
        activation_weight = self.W(candidate)
        activation_weight = tf.tile(tf.expand_dims(activation_weight, -2), [1, tf.shape(behaviors)[-2], 1])
        activation_weight = tf.exp(tf.reduce_sum(activation_weight * behaviors, -1))  # (bs,seq)
        activation_weight = tf.where(mask, activation_weight, tf.zeros_like(activation_weight))
        activation_weight = activation_weight / (tf.reduce_sum(activation_weight, axis=-1, keep_dims=True) + 1e-8)
        activation_weight = tf.expand_dims(activation_weight, -1)
        behaviors = tf.concat([behaviors, activation_weight], -1)
        behaviors = self.augru(behaviors, mask=mask)
        return behaviors


class PNN(tf.keras.layers.Layer):
    def __init__(self, w1, v):
        super(PNN, self).__init__()
        self.fm = FMLayer(0, 0, w1, v, point_wise=False)
        self.V = v

    def call(self, inputs, **kwargs):
        embed_inputs = tf.reduce_sum(self.V(inputs), axis=-2)
        fm_inputs = self.fm(inputs)
        embed_sqrt_mean = tf.sqrt(tf.reduce_sum(embed_inputs ** 2, axis=-1, keepdims=True) + 1e-8)
        fm_sqrt_mean = tf.sqrt(tf.reduce_sum(fm_inputs ** 2, axis=-1, keepdims=True) + 1e-8)
        fm_inputs = fm_inputs / fm_sqrt_mean * embed_sqrt_mean
        outputs = tf.concat([fm_inputs, embed_inputs], axis=-1)
        # outputs = tf.reshape(outputs, [-1, 2 * outputs.shape[-1]])
        return outputs


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    layer = DIEN(3, 0.5)
    h = tf.random_normal(shape=(4, 14, 5))
    c = tf.random_normal(shape=(4, 5))
    mask = tf.greater_equal(h[:, :, 0], -10000000)
    y = layer([h, c], mask=mask)
    print(y)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run([y, mask]))
