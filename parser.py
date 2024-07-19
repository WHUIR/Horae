import argparse
import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', nargs='?', default='./saved_model/',
                        help='Store model path.')
    parser.add_argument('--dataset', type=str, default='FHCKM')
    parser.add_argument('--data_path', type=str, default='./dataset/Amazon/')
    parser.add_argument('--local_save_path', type=str, default='./local_data/')
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--lr', type=float,
                        help='Learning rate.')
    parser.add_argument('--lr2', type=float,
                        help='Learning rate.')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--load_model_path', type=str, default='')
    parser.add_argument('--save_flag', type=int, default=1)
    parser.add_argument('--flag_step', type=int, default=10)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--aspects', type=int)
    parser.add_argument('--balance_alpha', type=float)
    parser.add_argument('--stage', type=str, default='pretrain') #trans
    parser.add_argument('--save_step', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=2048)
    parser.add_argument('--freeze', action='store_true', default=True)
    parser.add_argument('--time_scale', type=int, default=60 * 60 * 24)
    parser.add_argument('--triplet_alpha', type=float)
    parser.add_argument('--triplet_beta', type=float)
    parser.add_argument('--center_alpha', type=float)

    return parser.parse_args()


def parser_dic():
    parameter_dict = {
    }
    data_dic = {
        'dataset': 'Scientific',
        "min_seq_len": 1,
        'max_seq_len': 20,
        'data_augmentation': False,
        'data_split': 'loo',
        'pad_mode': 'prefix',
        'ks': [5, 10, 20, 50, 100],
        'neg_count': 'contrastive',
        'test_neg_count': 100,
        'data_path': './dataset/Amazon/',
        'local_save_path': './local_data/',
        'use_text_emb': True,
        'use_id_emb': True,
        'item_drop_ratio': 0.2,
    }

    optimize_dic = {
        'clip_grad_norm': {'max_norm': 5.0, 'norm_type': 2},
        'epoch': 1000,
        'train_batch_size': 1024,
        'test_batch_size': 2048,
        'regs': 0,
        'lr': 0.001,
        'lr2': 0.001,
        'save_step': 2,
        'freeze': False,
    }

    model_dic = {
        "n_layers": 2, #default 2
        "n_heads": 2,
        "hidden_size": 300,
        'embedding_size': 300,
        "inner_size": 256,
        "hidden_dropout_prob": 0.3,
        "attn_dropout_prob": 0.3,
        "hidden_act": 'gelu',
        "layer_norm_eps": 1e-12,
        "initializer_range": 0.02,
        "loss_type": 'CE',

        'encoder_layers': 2, #default 2

        'aspect_cons_tau': 0.1,
        'aspects': 32,
        'aspect_alpha': 0.01,

        'balance_alpha': 10,
        'noise_scale': 0.2,

        'disen_alpha': 1e-2,
        'disen_tau': 1,

        'seq_cons_alpha': 1e-3,
        'seq_cons_tau': 0.07,

        'interest_cons_alpha': 1e-2,
        'item_cons_alpha': 1e-3,

        'triplet_alpha':0.05,
        'triplet_beta':0.05,

        'center_alpha':0.05,

        'n_exps': 8,
        'adaptor_dropout_prob': 0.3,
        'adaptor_layers': [768, 300],

        'caps_layers': 3,

        'moe_dropout': 0.3,
        'stage': 'trans',

        'time_embedding_size':200,
    }

    special_dic = {
    }

    parameter_dict.update(model_dic)
    parameter_dict.update(optimize_dic)
    parameter_dict.update(data_dic)

    return parameter_dict



def get_config():
    args = parse_args()
    args_dic = vars(args)

    parameter_dict = parser_dic()

    for key in args_dic.keys():
        if args_dic[key] is not None:
            parameter_dict[key] = args_dic[key]

    return parameter_dict
