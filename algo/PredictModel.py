import tensorflow as tf

from BackModels import FMLayer, MyEmbeddingLayer, PredictMLP, NARM, ESMM, DeepFM, PNN, DIN, DIEN


class PredictModel(tf.keras.layers.Layer):
    def __init__(self, embedding_size, max_features, decay_step, lr, l2_reg,
                 prediction_hidden_width, keep_prob=0.5, model_name='FM'):
        super(PredictModel, self).__init__()
        self.l2_reg = l2_reg
        self.lr = lr
        self.prediction_hidden_width = prediction_hidden_width
        self.model_name = model_name

        self.keep_prob = keep_prob
        self.embedding_size = embedding_size
        self.max_features = max_features
        self.W1 = MyEmbeddingLayer(tf.Variable(tf.random_normal([max_features, 1], stddev=1e-3)))  # 前两项线性层
        self.V = MyEmbeddingLayer(tf.Variable(tf.random_normal([max_features, embedding_size], stddev=1e-3)))  # 交互矩阵

        self.global_step = tf.Variable(0, trainable=False)
        cov_learning_rate = tf.train.exponential_decay(self.lr, self.global_step, decay_step, 0.7)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=cov_learning_rate)
        self.back_model = self.get_model()

    def get_model(self):
        back_model = tf.keras.Sequential()
        if self.model_name == 'FM':
            back_model.add(FMLayer(0, 0, self.W1, self.V, point_wise=True))
        elif self.model_name == 'FM2':
            back_model.add(FMLayer(0, 0, self.W1, self.V, point_wise=False))
            back_model.add(PredictMLP(self.prediction_hidden_width, self.keep_prob))
            back_model.add(tf.keras.layers.Dense(1, use_bias=True))
        elif self.model_name == 'DeepFM':
            self.prediction_hidden_width.append(1)
            back_model.add(DeepFM(self.W1, self.V, self.prediction_hidden_width, self.keep_prob))
        elif self.model_name == 'PNN':
            back_model.add(PNN(self.W1, self.V))
            back_model.add(PredictMLP(self.prediction_hidden_width, self.keep_prob))
            back_model.add(tf.keras.layers.Dense(1, use_bias=True))
        return back_model

    def optimize_loss(self, y, y_, training):
        y_ = tf.sigmoid(y_)
        base_loss = tf.losses.log_loss(labels=y, predictions=y_)
        base_loss = tf.reduce_mean(base_loss)
        loss = base_loss
        for v in tf.trainable_variables():
            loss += self.l2_reg * tf.nn.l2_loss(v)
        tf.summary.scalar('base_loss', base_loss)
        tf.summary.scalar('loss', loss)

        threshold = 0.5
        one_click = tf.ones_like(y)
        zero_click = tf.zeros_like(y)
        eval_metric_ops = {
            "auc": tf.metrics.auc(y, y_),
            "acc": tf.metrics.accuracy(y, tf.where(y_ >= threshold, one_click, zero_click)),
        }
        if not training:
            return base_loss, loss, eval_metric_ops, y_, y
        gvs, v = zip(*self.optimizer.compute_gradients(loss))
        gvs, _ = tf.clip_by_global_norm(gvs, 5.0)
        gvs = zip(gvs, v)
        train_op = self.optimizer.apply_gradients(gvs, global_step=self.global_step)
        return base_loss, loss, eval_metric_ops, y_, y, train_op

    def call(self, x, y, training=True, **kwargs):
        if training:
            tf.keras.backend.set_learning_phase(1)
        else:
            tf.keras.backend.set_learning_phase(0)
        x = [tf.reshape(i, (-1, 2 * i.shape[-1])) for i in x]
        y_ = tf.squeeze(self.back_model(x), -1)
        return self.optimize_loss(y, y_, training)


class PredictModelRNN(PredictModel):
    def __init__(self, embedding_size, hidden_size, max_features, decay_step, lr, l2_reg,
                 prediction_hidden_width, keep_prob=0.5, model_name='TRNN', two_side=False):
        super(PredictModelRNN, self).__init__(embedding_size, max_features, decay_step, lr, l2_reg,
                                              prediction_hidden_width, keep_prob, model_name)
        self.two_side = two_side
        self.pnn = PNN(self.W1, self.V)
        self.rnn = self.get_rnn(hidden_size, keep_prob)

    def get_model(self):
        back_model = tf.keras.Sequential()
        back_model.add(PredictMLP(self.prediction_hidden_width, self.keep_prob))
        back_model.add(tf.keras.layers.Dense(1, use_bias=True))
        return back_model

    def get_rnn(self, hidden_size, keep_prob):
        if self.model_name == 'NARM':
            return NARM(hidden_size, 1 - keep_prob)
        elif self.model_name == 'ESMM':
            return ESMM(hidden_size, 1 - keep_prob)
        elif self.model_name == 'DIN':
            return DIN(1 - keep_prob)
        elif self.model_name == 'DIEN':
            return DIEN(hidden_size, 1 - keep_prob)
        return tf.keras.layers.LSTM(hidden_size, dropout=1 - keep_prob, return_sequences=True)

    def one_side_forward(self, x):
        x, mask = x
        x_user, x_log = [self.pnn(i) for i in x]

        if self.model_name in ['DIN', 'DIEN']:
            x_log = self.rnn([x_log, x_user[:, 1]], mask=mask)  # (bs,max_len,hidden_dim)
        else:
            x_log = self.rnn(x_log, mask=mask)  # (bs,max_len,hidden_dim)
        mask = tf.cast(tf.expand_dims(mask, axis=-1), dtype=tf.float32)  # (bs,max_len,1)
        x_log = tf.reduce_sum(x_log * mask, axis=1) / (tf.reduce_sum(mask, axis=-2) + 1e-8)  # (bs,hidden_dim)
        x_user = tf.reshape(x_user, [-1, 2 * x_user.shape[-1]])
        x = tf.concat([x_user, x_log], -1)
        return x

    def two_side_forward(self, x):
        x, mask = x
        friend_mask, user_log_mask, anchor_log_mask = mask
        x = [self.pnn(i) for i in x]
        user_anchor, real_friend, user_log, anchor_log = x
        if self.model_name in ['DIN', 'DIEN']:
            real_friend = self.rnn([real_friend, user_anchor[:, 1]], mask=friend_mask)  # (bs,max_len,hidden_dim)
        else:
            real_friend = self.rnn(real_friend, mask=friend_mask)  # (bs,max_len,hidden_dim)
        mask = [tf.cast(tf.expand_dims(_, -1), tf.float32) for _ in
                (friend_mask, user_log_mask, anchor_log_mask)]
        seq = [real_friend, user_log, anchor_log]
        seq = [tf.reduce_sum(mask[_] * seq[_], -2) / (tf.reduce_sum(mask[_], -2) + 1e-8) for _ in range(len(seq))]
        seq = tf.concat(seq, -1)
        x_user = tf.reshape(user_anchor, [-1, 2 * user_anchor.shape[-1]])
        return tf.concat([seq, x_user], -1)

    def call(self, x, y, training=True, **kwargs):
        if training:
            tf.keras.backend.set_learning_phase(1)
        else:
            tf.keras.backend.set_learning_phase(0)
        if self.two_side:
            x = self.two_side_forward(x)
        else:
            x = self.one_side_forward(x)
        y_ = tf.squeeze(self.back_model(x), -1)
        return self.optimize_loss(y, y_, training)


def get_mask_softmax(a, mask):
    # (bs,max_len,1)
    a = tf.exp(a)
    a = tf.where(tf.expand_dims(mask, -1), a, tf.zeros_like(a))
    return a / (tf.reduce_sum(a, axis=-2, keepdims=True) + 1e-8)


def cosine_similarity(x1, x2):
    norm_1 = tf.sqrt(tf.reduce_sum(tf.square(x1), axis=-1) + 1e-8)
    norm_2 = tf.sqrt(tf.reduce_sum(tf.square(x2), axis=-1) + 1e-8)

    product = tf.reduce_sum(x1 * x2, axis=-1)
    cosine = product / (norm_1 * norm_2)
    return cosine


class PredictModelTwins(PredictModelRNN):
    def __init__(self, embedding_size, hidden_size, max_features, decay_step, lr, l2_reg,
                 prediction_hidden_width, keep_prob=0.5, interact_mode=0, search_len=10):
        super(PredictModelTwins, self).__init__(embedding_size, hidden_size, max_features, decay_step, lr, l2_reg,
                                                prediction_hidden_width, keep_prob, 'Twins')
        self.search_len = search_len
        self.interact_mode = interact_mode
        if interact_mode == 0:
            self.attention_item_1 = tf.keras.layers.Dense(1, use_bias=True)
            self.attention_item_2 = tf.keras.layers.Dense(1, use_bias=True)
            self.interact_fun = self.interact_fun0
        else:
            self.attention_item = tf.keras.layers.Dense(1, use_bias=True)
            self.interact_fun = self.interact_fun1
        self.attention_anchor = tf.keras.layers.Dense(1, use_bias=True)
        self.A1 = tf.keras.layers.Dense(hidden_size, use_bias=False)
        self.hidden_size = hidden_size

    def get_top_k(self, user_anchor_info, item_seq, mask):
        category_sim = cosine_similarity(user_anchor_info, item_seq)
        category_sim = tf.where(mask, category_sim, tf.fill(tf.shape(category_sim), -2.0))
        top_item_idx = tf.nn.top_k(category_sim, self.search_len, False).indices  # (bs,item_num)
        top_item_idx = tf.nn.top_k(top_item_idx, tf.shape(top_item_idx)[1], False).values
        top_item_idx = tf.reverse(top_item_idx, axis=[-1])
        a1 = tf.expand_dims(tf.range(tf.shape(top_item_idx)[0]), -1)
        a1 = tf.tile(a1, [1, self.search_len])
        top_item_idx = tf.stack([a1, top_item_idx], axis=-1)
        item_seq = tf.gather_nd(item_seq, top_item_idx)
        mask = tf.gather_nd(mask, top_item_idx)
        return item_seq, mask

    def interact_fun0(self, user_anchor_info, user_log, anchor_log, user_log_mask, anchor_log_mask):
        user_anchor_info = tf.repeat(tf.expand_dims(user_anchor_info, -2), tf.shape(user_log)[1], -2)
        user_log = self.rnn(user_log, mask=user_log_mask)  # (bs,max_len,hidden)
        anchor_log = self.rnn(anchor_log, mask=anchor_log_mask)
        a_item_1 = self.attention_item_1(
            tf.concat([user_anchor_info[:, 0] * user_anchor_info[:, 1] * user_log,
                       user_anchor_info[:, 0], user_anchor_info[:, 1], user_log], axis=-1))  # (bs,max_len,1)
        a_item_2 = self.attention_item_2(
            tf.concat([user_anchor_info[:, 0] * user_anchor_info[:, 1] * anchor_log,
                       user_anchor_info[:, 0], user_anchor_info[:, 1], anchor_log], axis=-1))

        a_item_1 = get_mask_softmax(a_item_1, user_log_mask)
        a_item_2 = get_mask_softmax(a_item_2, anchor_log_mask)
        user_log = tf.reduce_sum(user_log * a_item_1, -2)
        anchor_log = tf.reduce_sum(anchor_log * a_item_2, -2)  # (bs,hidden)
        return tf.concat([user_log, anchor_log], -1)

    def interact_fun1(self, user_anchor_info, user_log, anchor_log, user_log_mask, anchor_log_mask):
        user_anchor_copy = tf.repeat(tf.expand_dims(user_anchor_info, -2), tf.shape(user_log)[1], -2)

        user_log, user_log_mask = self.get_top_k(user_anchor_copy[:, 1], user_log, user_log_mask)
        anchor_log, anchor_log_mask = self.get_top_k(user_anchor_copy[:, 0], anchor_log, anchor_log_mask)
        user_log = self.rnn(user_log, mask=user_log_mask)  # (bs,max_len,hidden)
        anchor_log = self.rnn(anchor_log, mask=anchor_log_mask)
        user_log = tf.tile(user_log, [1, self.search_len, 1])
        user_log_mask = tf.tile(user_log_mask, [1, self.search_len])

        anchor_log = tf.repeat(tf.expand_dims(anchor_log, -2), self.search_len, -2)
        anchor_log = tf.reshape(anchor_log, (-1, self.search_len ** 2, anchor_log.shape[-1]))
        anchor_log_mask = tf.repeat(tf.expand_dims(anchor_log_mask, -2), self.search_len, -1)
        anchor_log_mask = tf.reshape(anchor_log_mask, (-1, self.search_len ** 2))

        user_anchor_copy = tf.repeat(tf.expand_dims(user_anchor_info, -2), tf.shape(user_log)[1], -2)
        mask = tf.logical_and(user_log_mask, anchor_log_mask)
        a_item = self.attention_item(
            tf.concat([user_anchor_copy[:, 0], user_anchor_copy[:, 1], user_log, anchor_log], -1))
        a_item = get_mask_softmax(a_item, mask)
        return tf.reduce_sum(user_log * anchor_log * a_item, -2)

    def call(self, x, y, training=True, **kwargs):
        x, mask = x
        friend_mask, user_log_mask, anchor_log_mask = mask
        x = [self.pnn(i) for i in x]
        x = [self.A1(i) for i in x]
        user_anchor, real_friend, user_log, anchor_log = x
        # (bs,2,emb),(bs,max_friend,emb),(bs,max_user_log,emb),(bs,max_anchor_len,emb)
        log_info = self.interact_fun(user_anchor, user_log, anchor_log, user_log_mask, anchor_log_mask)

        user_anchor_copy = tf.repeat(tf.expand_dims(user_anchor, -2), tf.shape(real_friend)[1], -2)
        a_anchor = self.attention_anchor(
            tf.concat([user_anchor_copy[:, 0], user_anchor_copy[:, 1], real_friend,
                       user_anchor_copy[:, 0] * user_anchor_copy[:, 1] * real_friend], axis=-1))
        a_anchor = get_mask_softmax(a_anchor, friend_mask)
        real_friend = tf.reduce_sum(real_friend * a_anchor, -2)
        user_anchor = tf.reshape(user_anchor, [-1, 2 * self.hidden_size])
        y_ = self.back_model(tf.concat([user_anchor, log_info, real_friend], axis=-1))
        # y_ = self.back_model(tf.concat([user_anchor, user_log, anchor_log], axis=-1))
        # y_ = self.back_model(tf.concat([user_anchor], axis=-1))
        # y_ = self.back_model(tf.concat([user_anchor, real_friend], axis=-1))
        y_ = tf.squeeze(y_, -1)
        return self.optimize_loss(y, y_, training)
