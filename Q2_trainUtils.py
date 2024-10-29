import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc, mean_squared_error, mean_absolute_error, accuracy_score,r2_score
import numpy as np

detach = lambda o: o.cpu().detach().numpy().tolist()

def train(model, dataloader, optimizer, device):
    model.train(mode=True)
    y_pred, y_true = [], []
    loss_all = []

    y_criterion = nn.BCELoss()

    for item in dataloader:
      U = item[0].to(device).long()
      Q = item[1].to(device).long()
      KC = item[2].to(device).long()
      R = item[3].to(device).long()
      TS = item[4].to(device).long()
      
      optimizer.zero_grad()

      P = model(Q, KC, R, TS)
      R_label = R[:,1:]

      index = R_label != 2

      P, R_label = P[index], R_label[index]
                                       
      loss = y_criterion(P, R_label.float()) 
      loss_all.append(loss.item())


      loss.backward()
      optimizer.step()

      y_pred += detach(P)
      y_true += detach(R_label.float())

    fpr, tpr, thres = roc_curve(y_true, y_pred, pos_label=1)
    mse_value = mean_squared_error(y_true, y_pred)
    rmse_value = np.sqrt(mse_value)
    mae_value = mean_absolute_error(y_true, y_pred)
    bi_y_pred = [1 if i >= 0.5 else 0 for i in y_pred]
    acc_value = accuracy_score(y_true, bi_y_pred)
    r2_value = r2_score(y_true, y_pred)

    return auc(fpr, tpr), np.mean(loss_all), mse_value, rmse_value, mae_value, acc_value, r2_value


def evaluate(model, dataloader, device):
    model.eval()
    y_pred, y_true = [], []
    loss_all = []

    y_criterion = nn.BCELoss()

    for item in dataloader:
      U = item[0].to(device).long()
      Q = item[1].to(device).long()
      KC = item[2].to(device).long()
      R = item[3].to(device).long()
      TS = item[4].to(device).long()
      
      with torch.no_grad():
         P = model(Q, KC, R, TS)
      R_label = R[:,1:]

      index = R_label != 2
      P, R_label = P[index], R_label[index]

      loss = y_criterion(P, R_label.float())
      loss_all.append(loss.item())

      y_pred += detach(P)
      y_true += detach(R_label.float())

    fpr, tpr, thres = roc_curve(y_true, y_pred, pos_label=1)
    mse_value = mean_squared_error(y_true, y_pred)
    rmse_value = np.sqrt(mse_value)
    mae_value = mean_absolute_error(y_true, y_pred)
    bi_y_pred = [1 if i >= 0.5 else 0 for i in y_pred]
    acc_value = accuracy_score(y_true, bi_y_pred)
    r2_value = r2_score(y_true, y_pred)

    return auc(fpr, tpr), np.mean(loss_all), mse_value, rmse_value, mae_value, acc_value, r2_value
