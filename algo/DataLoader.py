import numpy as np
import tensorflow as tf


def deal(batch_dis, batch_con, start):
    con_ids = tf.range(batch_con.shape[-1], dtype=tf.int32) + start
    con_ids = tf.expand_dims(con_ids, 0)
    con_ids = tf.expand_dims(con_ids, 0)  # (1,1,con)
    batch_ids = tf.concat([
        tf.cast(batch_dis, tf.int32),
        tf.tile(con_ids, [tf.shape(batch_dis)[0], tf.shape(batch_dis)[1], 1])
    ], axis=-1)
    batch_features = tf.concat([
        tf.ones_like(batch_dis, dtype=tf.float32),
        batch_con
    ], axis=-1)  # (bs,?,dis+con)
    return [batch_ids, batch_features]


class DataSet:
    def __init__(self, user_item, batch_size):
        self.user_info = user_item['user_info']
        self.user_friend_label = user_item['user_friend_label']
        self.mask = user_item['mask']
        train_valid_test = [np.where(self.mask == _)[0] for _ in [0, 1, 2]]
        for i in train_valid_test:
            np.random.shuffle(i)
        train, valid, test = train_valid_test
        self.mode_mask = {'train': train, 'valid': valid, 'test': test}
        self.batch_size = batch_size
        self.discrete_len = user_item['fields_len'][0]
        self.continuous_len = np.shape(self.user_info)[1] - self.discrete_len
        self.max_discrete_id = np.sum(user_item['fields_num'][:self.discrete_len])
        self.fields_num_sum = int(self.max_discrete_id + self.continuous_len)
        if self.continuous_len != 0:
            self.user_info[:, self.discrete_len:] /= np.max(self.user_info[:, self.discrete_len:], axis=0,
                                                            keepdims=True)
        self.mode = 'train'
        self.user_info = tf.convert_to_tensor(self.user_info, dtype=tf.float32)

    def input_fn(self, mode='train', epochs=1, shuffle=True):
        dataset = tf.data.Dataset.from_tensor_slices(self.user_friend_label[self.mode_mask[mode]])
        if shuffle:
            dataset = dataset.shuffle(self.batch_size + 1)
        dataset = dataset.repeat(epochs)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(1)
        iterator = dataset.make_one_shot_iterator()
        batch_x = iterator.get_next()
        batch_x, batch_labels = tf.gather(self.user_info, batch_x[:, :2]), batch_x[:, -1]  # (bs,2,dis+con),(bs,)
        batch_ids, batch_features = deal(batch_x[:, :, :self.discrete_len],
                                         batch_x[:, :, self.discrete_len:],
                                         self.max_discrete_id)  # (bs,2,dis+con)

        return (batch_ids, batch_features), tf.cast(batch_labels, tf.float32)


class RNNDataSet(DataSet):
    def __init__(self, user_item, batch_size, max_len=50, seq_base='anchor'):
        super().__init__(user_item, batch_size)
        self.log = user_item['log'].astype(float)
        self.log_shape = np.shape(self.log)
        self.user_log_id = np.array([np.arange(i[0], i[0] + i[1])[-max_len:] for i in user_item['begin_len']],
                                    dtype=np.ndarray)
        self.discrete_len_log = user_item['fields_len'][1]
        self.continuous_len_log = self.log_shape[1] - self.discrete_len_log
        self.max_discrete_id = np.sum(user_item['fields_num'])
        self.fields_num_sum = int(self.max_discrete_id + self.continuous_len + self.continuous_len_log)
        if self.continuous_len_log != 0:
            self.log[:, self.discrete_len_log:] /= np.max(self.log[:, self.discrete_len_log:], axis=0, keepdims=True)
        if seq_base != 'anchor':
            self.user_friend_label[:, [0, 1]] = self.user_friend_label[:, [1, 0]]
            # will use the seq of the user

    def from_generator(self, mode):
        mode = str(mode, encoding='utf-8')
        for x in self.user_friend_label[self.mode_mask[mode]]:
            log = self.log[self.user_log_id[x[1]]]
            yield x, log[:, :self.discrete_len_log], log[:, self.discrete_len_log:]

    def input_fn(self, mode='train', epochs=1, shuffle=True):
        dataset = tf.data.Dataset.from_generator(self.from_generator, (tf.int32, tf.int32, tf.float32), args=(mode,))
        if shuffle:
            dataset = dataset.shuffle(self.batch_size + 1)
        dataset = dataset.repeat(epochs)
        dataset = dataset.padded_batch(self.batch_size,
                                       padded_shapes=(
                                           tf.TensorShape([3]),  # 表示不补全
                                           tf.TensorShape([None, self.discrete_len_log]),
                                           tf.TensorShape([None, self.log_shape[1] - self.discrete_len_log])
                                       ),
                                       padding_values=(0, self.fields_num_sum, 0.0))
        dataset = dataset.prefetch(1)
        iterator = dataset.make_one_shot_iterator()
        batch_x, batch_log_1, batch_log_2 = iterator.get_next()  # (bs,3),(bs,max_len,dis_log),(bs,max_len,con_log)

        pad_mask = tf.not_equal(batch_log_1[:, :, 0], self.fields_num_sum)
        batch_user, batch_labels = batch_x[:, :2], batch_x[:, -1]
        batch_x = tf.gather(self.user_info, batch_user)  # (bs,2,dis+con)
        batch_ids, batch_features = deal(batch_x[:, :, :self.discrete_len], batch_x[:, :, self.discrete_len:],
                                         self.max_discrete_id)  # (bs,2,dis+con)
        batch_log_ids, batch_log_features = deal(batch_log_1, batch_log_2,
                                                 start=self.max_discrete_id + self.continuous_len)  # (bs,dis_log*2+con_log*2)

        batch_x = ((batch_ids, batch_features), (batch_log_ids, batch_log_features))
        return (batch_x, pad_mask), tf.cast(batch_labels, tf.float32)


class TwinsDataset(RNNDataSet):
    def __init__(self, user_item, batch_size, max_len=50):
        super(TwinsDataset, self).__init__(user_item, batch_size, max_len)
        self.user_positive = self.user_friend_label[:, :2][self.user_friend_label[:, -1] > 0]
        self.user_friend_list = np.concatenate([[0], np.diff(self.user_positive[:, 0])]) != 0
        self.user_friend_list = np.where(self.user_friend_list)[0]
        self.user_friend_list = np.split(self.user_positive, self.user_friend_list)
        user_friend_list_ = list(np.expand_dims(np.arange(len(self.user_log_id)), -1))
        for user_friend in self.user_friend_list:
            user_friend_list_[user_friend[0, 0]] = np.concatenate([
                user_friend_list_[user_friend[0, 0]], user_friend[:max_len, 1]])
        self.user_friend_list = user_friend_list_

    def from_generator(self, mode):
        mode = str(mode, encoding='utf-8')
        for x in self.user_friend_label[self.mode_mask[mode]]:
            log_user = self.log[self.user_log_id[x[0]]]
            log_anchor = self.log[self.user_log_id[x[1]]]
            anchors = self.user_friend_list[x[0]]
            anchors = anchors[anchors != x[1]]
            yield x, anchors, \
                  log_user[:, :self.discrete_len_log], log_user[:, self.discrete_len_log:], \
                  log_anchor[:, :self.discrete_len_log], log_anchor[:, self.discrete_len_log:]

    def input_fn(self, mode='train', epochs=1, shuffle=True):
        dataset = tf.data.Dataset.from_generator(self.from_generator,
                                                 (tf.int32, tf.int32,
                                                  tf.int32, tf.float32, tf.int32, tf.float32), args=(mode,))
        if shuffle:
            dataset = dataset.shuffle(self.batch_size + 1)
        dataset = dataset.repeat(epochs)
        dataset = dataset.padded_batch(self.batch_size,
                                       padded_shapes=(
                                           tf.TensorShape([3]),  # 表示不补全
                                           tf.TensorShape([None]),
                                           tf.TensorShape([None, self.discrete_len_log]),
                                           tf.TensorShape([None, self.log_shape[1] - self.discrete_len_log]),
                                           tf.TensorShape([None, self.discrete_len_log]),
                                           tf.TensorShape([None, self.log_shape[1] - self.discrete_len_log])
                                       ),
                                       padding_values=(
                                           0, self.fields_num_sum, self.fields_num_sum, 0.0, self.fields_num_sum, 0.0))
        dataset = dataset.prefetch(1)
        iterator = dataset.make_one_shot_iterator()
        batch_x, batch_friend, batch_log_user_1, batch_log_user_2, batch_log_anchor_1, batch_log_anchor_2 = iterator.get_next()

        friend_mask = tf.not_equal(batch_friend[:, :], self.fields_num_sum)
        batch_friend = tf.where(friend_mask, batch_friend, tf.zeros_like(batch_friend))
        user_log_mask = tf.not_equal(batch_log_user_1[:, :, 0], self.fields_num_sum)
        anchor_log_mask = tf.not_equal(batch_log_anchor_1[:, :, 0], self.fields_num_sum)

        batch_user, batch_labels = batch_x[:, :2], batch_x[:, -1]
        batch_x = tf.gather(self.user_info, batch_user)  # (bs,2,dis+con)
        batch_friend = tf.gather(self.user_info, batch_friend)  # (bs,max_friend,dis+con)

        batch_x = deal(batch_x[:, :, :self.discrete_len], batch_x[:, :, self.discrete_len:],
                       self.max_discrete_id)
        batch_friend = deal(batch_friend[:, :, :self.discrete_len],
                            batch_friend[:, :, self.discrete_len:],
                            self.max_discrete_id)
        batch_user_log = deal(batch_log_user_1, batch_log_user_2,
                              self.max_discrete_id + self.continuous_len)
        batch_anchor_log = deal(batch_log_anchor_1, batch_log_anchor_2,
                                self.max_discrete_id + self.continuous_len)

        return ((batch_x, batch_friend, batch_user_log, batch_anchor_log),
                (friend_mask, user_log_mask, anchor_log_mask)), \
               tf.cast(batch_labels, tf.float32)


class AnchorRNNDataSet(TwinsDataset):
    def from_generator(self, mode):
        mode = str(mode, encoding='utf-8')
        for x in self.user_friend_label[self.mode_mask[mode]]:
            anchors = self.user_friend_list[x[0]]
            anchors = anchors[anchors != x[1]]
            yield x, anchors

    def input_fn(self, mode='train', epochs=1, shuffle=True):
        dataset = tf.data.Dataset.from_generator(self.from_generator,
                                                 (tf.int32, tf.int32), args=(mode,))
        if shuffle:
            dataset = dataset.shuffle(self.batch_size + 1)
        dataset = dataset.repeat(epochs)
        dataset = dataset.padded_batch(self.batch_size,
                                       padded_shapes=(
                                           tf.TensorShape([3]),  # 表示不补全
                                           tf.TensorShape([None])),
                                       padding_values=(0, self.fields_num_sum))
        dataset = dataset.prefetch(1)
        iterator = dataset.make_one_shot_iterator()
        batch_x, batch_friend = iterator.get_next()
        friend_mask = tf.not_equal(batch_friend[:, :], self.fields_num_sum)
        batch_friend = tf.where(friend_mask, batch_friend, tf.zeros_like(batch_friend))

        batch_user, batch_labels = batch_x[:, :2], batch_x[:, -1]
        batch_x = tf.gather(self.user_info, batch_user)  # (bs,2,dis+con)
        batch_friend = tf.gather(self.user_info, batch_friend)  # (bs,max_friend,dis+con)

        batch_x = deal(batch_x[:, :, :self.discrete_len], batch_x[:, :, self.discrete_len:],
                       self.max_discrete_id)
        batch_friend = deal(batch_friend[:, :, :self.discrete_len], batch_friend[:, :, self.discrete_len:],
                            self.max_discrete_id)
        return ((batch_x, batch_friend), friend_mask), tf.cast(batch_labels, tf.float32)


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    dataset_name = 'citation'
    np_path = './dataset/{}/{}.npz'.format(dataset_name, dataset_name)
    user_item = np.load(np_path)
    dataset = TwinsDataset(user_item, 2000)
    x, y = dataset.input_fn('train', epochs=1)
    x, mask = x
    mask = [tf.reduce_min(tf.reduce_sum(tf.cast(i, dtype=tf.int32), axis=-1)) for i in mask]
    print(x)
    with tf.Session() as sess:
        step = 0
        try:
            while True:
                if step > 3:
                    break
                step += 1
                b_mask = sess.run(mask)
                print(b_mask)
                print('-' * 40)
        except tf.errors.OutOfRangeError:
            print()
            print("print Done!!!")
    print("Done")
