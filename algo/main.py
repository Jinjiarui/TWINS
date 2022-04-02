import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from DataLoader import DataSet, RNNDataSet, TwinsDataset, AnchorRNNDataSet
from utils import load_model


def load_data(args):
    user_item = np.load(os.path.join(args['data_dir'], '{}.npz'.format(args['dataset'])))
    if not args['is_RNN']:
        dataset = DataSet(user_item, args['batch_size'])
    elif args['is_Twins']:
        dataset = TwinsDataset(user_item, args['batch_size'])
    else:
        if args['side'] == 'anchor':
            dataset = AnchorRNNDataSet(user_item, args['batch_size'])
        elif args['side'] == 'two':
            dataset = TwinsDataset(user_item, args['batch_size'])
        else:
            dataset = RNNDataSet(user_item, args['batch_size'], seq_base=args['base'])
    return dataset


def main(args):
    args['data_dir'] = args['data_dir'].format(args['dataset'])
    dataset = load_data(args)
    args.update({'all_features_num': dataset.fields_num_sum + 1})

    model_folder = "./saved_models/"
    model_path = model_folder + str(args["exp_name"]) + "_" + str(args["postfix"])
    if args['base'] == 'user':
        model_path += 'user'
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    train_x, train_label = dataset.input_fn('train', epochs=args['num_epochs'])
    valid_x, valid_label = dataset.input_fn('valid', epochs=None)
    test_x, test_label = dataset.input_fn('test', epochs=1)
    model = load_model(args)
    train_base_loss, train_loss, train_eval_metric_ops, train_y_, train_y, train_op = model(train_x, train_label)
    valid_base_loss, valid_loss, valid_eval_metric_ops, valid_y_, valid_y = model(valid_x, valid_label, training=False)
    test_base_loss, test_loss, test_eval_metric_ops, test_y_, test_y = model(test_x, test_label, training=False)
    saver = tf.train.Saver(max_to_keep=2)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    best_val_auc = 0
    step_up = len(dataset.mode_mask['train']) // args['batch_size']
    print("-----------------Training Start-----------------\n")
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if args['load_model']:
            saver.restore(sess, model_path)
        for epoch in range(args['num_epochs']):
            sess.run(tf.local_variables_initializer())
            for step in tqdm(range(step_up)):
                batch_base_loss, batch_loss, batch_eval, batch_y_, batch_y, _ = sess.run(
                    [train_base_loss, train_loss, train_eval_metric_ops, train_y_, train_y, train_op])
                print('#Epoch:{}, #batch:{}, base_loss:{:.4f} loss:{:.4f} acc:{:.4f} auc:{:.4f}'
                      .format(epoch, step, batch_base_loss, batch_loss, batch_eval['acc'][1], batch_eval['auc'][1]))
            print()
            print("-----------------Validating Start-----------------\n")
            sess.run(tf.local_variables_initializer())
            base_loss, loss = 0, 0
            valid_step = 200
            for _ in tqdm(range(valid_step)):
                batch_base_loss, batch_loss, batch_eval, batch_y_, batch_y = sess.run(
                    [valid_base_loss, valid_loss, valid_eval_metric_ops, valid_y_, valid_y])
                base_loss += batch_base_loss
                loss += batch_loss
            base_loss /= valid_step
            loss /= valid_step
            print('#Validated base_loss:{:.4f} loss:{:.4f} acc:{:.4f} auc:{:.4f}'
                  .format(base_loss, loss, batch_eval['acc'][1], batch_eval['auc'][1]))
            print(batch_y_[:100])
            print(batch_y[:100])
            if batch_eval['auc'][1] > best_val_auc:
                print('Get new better result!')
                best_val_auc = batch_eval['auc'][1]
                saver.save(sess, model_path)
                print("New best result Saved!")

        print("-----------------Testing Start-----------------\n")
        sess.run(tf.local_variables_initializer())
        step = 0
        base_loss, loss = 0, 0
        while True:
            try:
                step += 1
                batch_base_loss, batch_loss, batch_eval, _, _ = sess.run(
                    [test_base_loss, test_loss, test_eval_metric_ops, test_y_, test_y])
                base_loss += batch_base_loss
                loss += batch_loss
                print('#Testing step:{} base_loss:{:.4f} loss:{:.4f} acc:{:.4f} auc:{:.4f}'
                      .format(step, batch_base_loss, batch_loss, batch_eval['acc'][1], batch_eval['auc'][1]), end='\r')
            except tf.errors.OutOfRangeError:
                break
            print()
        print("Average base_loss:{:.4f} loss:{:.4f}".format(base_loss / step, loss / step))


if __name__ == "__main__":
    import argparse
    from configure import get_exp_configure

    parser = argparse.ArgumentParser(description="Twins")
    parser.add_argument(
        "-m", "--model_name", type=str,
        choices=['FM', 'FM2', 'DeepFM', 'PNN', 'DIN', 'DIEN', 'LSTM', 'NARM', 'ESMM', 'Twins'],
        default='FM', help="Model to use"
    )
    parser.add_argument('-d', '--dataset', type=str, choices=["yelp", 'trust', 'citation'], default="yelp",
                        help="Dataset to use")
    parser.add_argument('--base', type=str, choices=['user', 'anchor'], default='anchor', help='sequence choose')
    parser.add_argument('--side', type=str, choices=['anchor', 'item', 'two'], default='anchor', help='side choose')
    parser.add_argument('--data_dir', type=str, default='./dataset/{}')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--load_model', action="store_true", default=False)
    parser.add_argument('--interact_mode', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument("-c", "--cuda", type=str, default="0")
    parser.add_argument("--postfix", type=str, default="",
                        help="a string appended to the file name of the saved model_name")
    parser.add_argument("--rand_seed", type=int, default=-1, help="random seed for torch and numpy")
    args = parser.parse_args().__dict__
    args["exp_name"] = "_".join([args["model_name"], args["dataset"]])
    args.update(get_exp_configure(args))
    if args["cuda"] == "none":
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args["cuda"]
    main(args)
