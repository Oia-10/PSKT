import random
import torch
import torch.nn as nn
import numpy as np
import os


def set_seed(seed=1010):
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def get_group(df):
    group = df[
            ['user_id','problem_id','skill_id','correct','time_stamp']].groupby('user_id').apply(lambda r: (
            r['user_id'].values,
            r['problem_id'].values,
            r['skill_id'].values,
            r['correct'].values,
            r['time_stamp'].values,
        ))
    return group
    

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='   .pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.counter = 0

    def save_checkpoint(self, val_loss):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def save_info_opt(dataset, length, batch_size, learning_rate, seed, gpu, embed_dim, cv_num, early_stopp_patience, info_file):
    params_list = (
        'dataset = %s\n' % dataset,
        'length = %s\n' % length,
        'batch_size = %d\n' % batch_size,
        'learning_rate = %f\n' % learning_rate,
        'seed = %d\n' % seed,
        'gpu = %s\n' % gpu,
        'embed_dim = %f\n' % embed_dim,
        'cv_num = %d\n' % cv_num,
        'early_stopp_patience = %f\n' % early_stopp_patience,
    )
    info_file.write('%s %s %s %s %s %s %s %s %s' % params_list)


def save_info_train_valid(cv, epoch, max_auc,
                valid_auc, valid_loss, valid_mse, valid_rmse, valid_mae, valid_acc, valid_r2,
                train_auc, train_loss, train_mse, train_rmse, train_mae, train_acc, train_r2,
                time_end, time_start, info_file):
    print_list = (
        'cv:%-3d' % cv,
        'epoch:%-3d' % epoch,
        'max_auc:%-8.4f' % max_auc,

        'valid_auc:%-8.4f' % valid_auc,
        'valid_loss:%-8.4f' % valid_loss,
        'valid_mse:%-8.4f' % valid_mse,
        'valid_rmse:%-8.4f' % valid_rmse,
        'valid_mae:%-8.4f' % valid_mae,
        'valid_acc:%-8.8f' % valid_acc,
        'valid_r2:%-8.8f' % valid_r2,

        'train_auc:%-8.4f' % train_auc,
        'train_loss:%-8.4f' % train_loss,
        'train_mse:%-8.4f' % train_mse,
        'train_rmse:%-8.4f' % train_rmse,
        'train_mae:%-8.4f' % train_mae,
        'train_acc:%-8.8f' % train_acc,
        'train_r2:%-8.8f' % train_r2,

        'time:%-6.2fs' % (time_end - time_start)
    )

    print('%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s' % print_list)
    info_file.write('%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n' % print_list)


def save_info_test(cv, test_auc, test_loss, test_mse, test_rmse,test_mae, test_acc,  test_r2, info_file):
    print_list_test = (
        'cv:%-3d' % cv,
        'test_auc:%-8.8f' % test_auc,
        'test_loss:%-8.4f' % test_loss,
        'test_mse:%-8.4f' % test_mse,
        'test_rmse:%-8.4f' % test_rmse,
        'test_mae:%-8.4f' % test_mae,
        'test_acc:%-8.8f' % test_acc,
        'test_r2:%-8.4f' % test_r2,
    )

    print('%s %s %s %s %s %s %s %s\n' % print_list_test)
    info_file.write('%s %s %s %s %s %s %s %s\n' % print_list_test)
    
    
    

