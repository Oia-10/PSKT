import time
import datetime
import copy
from torch.utils.data import DataLoader
import pandas as pd
import pickle
import os
from Q0_config import *
from Q1_utils import *
from Q2_trainUtils import *
from Q3_dataload import *
from Q4_model import *

opt = set_opt()
set_seed(opt.seed)

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data_path = './data/'+ opt.dataset+'/' + opt.dataset+'_pro.csv'
test_list_path = './data/'+ opt.dataset+'/test_' + opt.dataset+'_100_split.pkl'
train_valid_list_path = './data/'+ opt.dataset +  '/' +  str(opt.cv_num) + '_train_valid_' + opt.dataset+'_100_split.pkl'

data = pd.read_csv(data_path)

with open(test_list_path, 'rb') as f:
    test_list = pickle.load(f)

with open(train_valid_list_path, 'rb') as f:
    train_list, valid_list = pickle.load(f)

train_df = data.loc[data['user_id'].isin(train_list)]
valid_df = data.loc[data['user_id'].isin(valid_list)]
test_df = data.loc[data['user_id'].isin(test_list)]

train_group = get_group(train_df)
valid_group = get_group(valid_df)
test_group = get_group(test_df)

train_dataset = KTDataset(train_group, max_seq=opt.length)
valid_dataset = KTDataset(valid_group, max_seq=opt.length)
test_dataset = KTDataset(test_group, max_seq=opt.length)

train_loader = DataLoader(train_dataset,
              batch_size=opt.batch_size,
              num_workers=12,
              shuffle=True)
valid_loader = DataLoader(valid_dataset,
              batch_size=opt.batch_size,
              num_workers=8,
              shuffle=True)
test_loader = DataLoader(test_dataset,
              batch_size=opt.batch_size,
              num_workers=8,
              shuffle=False)

model = PSKT(opt.q_num, opt.kc_num, opt.embed_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=0.000001)
early_stopping = EarlyStopping(patience = opt.early_stopp_patience, verbose=True)

result_path =  './result/'+ opt.dataset + '/%s' % ('{0:%Y-%m-%d-%H-%M-%S}'.format(datetime.datetime.now()))

os.makedirs(result_path)
info_file = open('%s/info.txt' % result_path, 'w+')

save_info_opt(opt.dataset,opt.length, opt.batch_size, opt.learning_rate, opt.seed, 
              opt.gpu, opt.embed_dim, opt.cv_num, opt.early_stopp_patience, info_file)


max_auc = 0.0
for epoch in range(1, opt.epochs + 1):

    time_start = time.time()
    train_auc, train_loss, train_mse, train_rmse, train_mae, train_acc, train_r2 = train(model, train_loader, optimizer, device)
    time_end = time.time()

    valid_auc, valid_loss, valid_mse, valid_rmse, valid_mae, valid_acc, valid_r2 = evaluate(model, valid_loader,device)


    if max_auc < valid_auc:
        max_auc = valid_auc
        torch.save(model.state_dict(), '%s/model' % ('%s' % result_path))
        current_max_model = copy.deepcopy(model)
    
    save_info_train_valid(opt.cv_num, epoch, max_auc,
                valid_auc, valid_loss, valid_mse, valid_rmse,valid_mae, valid_acc, valid_r2,
                train_auc, train_loss, train_mse, train_rmse,train_mae, train_acc, train_r2,
                time_end, time_start, info_file)

    early_stopping(valid_loss)    
    if early_stopping.early_stop:
        print("Early stopping")
      

        print('The training has been completed and the final result is: ')
        info_file.write('The training has been completed and the final result is: ')
        test_auc, test_loss, test_mse, test_rmse, test_mae, test_acc, test_r2 = evaluate(current_max_model, test_loader,device)
        save_info_test(opt.cv_num, test_auc, test_loss, test_mse, test_rmse, test_mae, test_acc,  test_r2, info_file)

        break

    

