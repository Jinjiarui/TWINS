from PredictModel import PredictModel, PredictModelRNN, PredictModelTwins


def load_model(args):
    if not args['is_RNN']:
        model = PredictModel(
            embedding_size=args['input_dim'],
            max_features=args['all_features_num'],
            decay_step=args['decay_step'],
            lr=args['lr'],
            l2_reg=args['l2_reg'],
            prediction_hidden_width=args['prediction_hidden_width'],
            keep_prob=args['keep_prob'],
            model_name=args['model_name']
        )
    elif args['is_Twins']:
        model = PredictModelTwins(
            embedding_size=args['input_dim'],
            hidden_size=args['hidden_dim'],
            max_features=args['all_features_num'],
            decay_step=args['decay_step'],
            lr=args['lr'],
            l2_reg=args['l2_reg'],
            prediction_hidden_width=args['prediction_hidden_width'],
            keep_prob=args['keep_prob'],
            search_len=args['search_len'],
            interact_mode=args['interact_mode']
        )
    else:
        model = PredictModelRNN(
            embedding_size=args['input_dim'],
            hidden_size=args['hidden_dim'],
            max_features=args['all_features_num'],
            decay_step=args['decay_step'],
            lr=args['lr'],
            l2_reg=args['l2_reg'],
            prediction_hidden_width=args['prediction_hidden_width'],
            keep_prob=args['keep_prob'],
            model_name=args['model_name'],
            two_side=(args['side'] == 'two')
        )
    return model
