from argparse import ArgumentParser

def set_opt():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='assist12',
                        help='choose from assist12, assist17, EdNet, junyi, Eedi')
    parser.add_argument('--length', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=1010)
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--cv_num', type=int, default=0)
    parser.add_argument('--early_stopp_patience', type=int, default=3)
    parser.add_argument('--model', type=str, default='PSKT')

    opt = parser.parse_args()

    if opt.dataset == 'assist12':
        opt.q_num = 36458
        opt.kc_num = 254
   
    print(opt)
    return opt
