def get_exp_configure(args):
    configure_list = {'yelp': {}, 'citation': {}, 'trust': {}}
    model_list = ['FM', 'FM2', 'DeepFM', 'PNN', 'LSTM', 'NARM', 'ESMM', 'Twins', 'DIN', 'DIEN']
    for model in model_list:
        for dataset in configure_list.keys():
            configure_list[dataset][model] = {
                'input_dim': 48,
                'batch_size': 2000,
                'decay_step': 1000,
                'l2_reg': 1e-4,
                'is_RNN': False,
                'is_Twins': False,
                'prediction_hidden_width': [256, 64, 16],
                'keep_prob': 0.5,
                'search_len': 10
            }
        configure_list['yelp'][model].update({'decay_step': 10000})
        configure_list['trust'][model].update({'decay_step': 260})
    for model in ['LSTM', 'NARM', 'ESMM', 'Twins', 'DIN', 'DIEN']:
        for dataset in configure_list.keys():
            configure_list[dataset][model].update({
                'hidden_dim': 64,
                'is_RNN': True
            })
    for model in ['Twins']:
        for dataset in configure_list.keys():
            configure_list[dataset][model].update({
                'is_Twins': True
            })
    return configure_list[args['dataset']][args['model_name']]
