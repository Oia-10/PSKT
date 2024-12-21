import numpy as np
import pandas as pd
import json
import re
import csv
import copy
import random
import time
from tqdm import tqdm
from collections import Counter
import warnings
from sklearn.model_selection import train_test_split, KFold
import pickle

# Loading Raw Data
data=pd.read_csv("./assist17/anonymized_full_release_competition_dataset.csv",encoding = "ISO-8859-15")

# Extracting Required Features
order = ['studentId','problemId','correct','skill','problemType', 'startTime','endTime']
data = data[order]

# Delete Records Without Knowledge Concepts
data['skill'].fillna('nan',inplace=True)
data = data[data['skill'] != 'nan'].reset_index(drop=True)

# Rename Features
data = data.rename(columns={'endTime': 'time_stamp'})
data = data.rename(columns={'studentId': 'user_id'})
data = data.rename(columns={'problemId': 'problem_id'})

# Reorder by Timestamp
data.sort_values(by=['user_id', 'time_stamp'], inplace=True)

# Set Sequence Length to 100
length = 100
data['new_user_id'] = data.groupby('user_id').cumcount() // length
data['new_user_id'] = data['user_id'].astype(str) + '_' + data['new_user_id'].astype(str)
data = data.rename(columns={'user_id': 'user_id_old'})
data = data.rename(columns={'new_user_id': 'user_id'})


# Delete Questions Appearing Less Than 10 Times
df_di = data["correct"].groupby(data["problem_id"])
data["popularity"] = df_di.transform('size')
data = data[data['popularity']>10]

# Delete Learners with Fewer Than 3 Responses
counts = data['user_id'].value_counts()
keep = counts[counts > 3].index
data = data[data['user_id'].isin(keep)]

# Reassign IDs
student_id_dict = {label:idx for idx,label in enumerate(set(data['user_id']))}
data['user_id'] = data['user_id'].map(student_id_dict)
skill_id_dict = {label:idx for idx,label in enumerate(set(data['skill']))}
data['skill_id'] = data['skill'].map(skill_id_dict)
quest_id_dict = {label:idx for idx,label in enumerate(set(data['problem_id']))}
data['problem_id'] = data['problem_id'].map(quest_id_dict)


data['correct']=data['correct'].astype(int)
data['skill_id']=data['skill_id'].astype(int)

# Five-Fold Cross-Validation
user_ids = np.unique(data['user_id'])  # 提取唯一的 user_id
user_len = len(user_ids)

# Check if user_id starts from 0 and is consecutive
if not np.array_equal(user_ids, np.arange(user_len)):
    raise ValueError("user_id is not consecutive or does not start from 0. Please reindex user_id.")

# Split into test set (20%) and train-validation set (80%)
train_valid_ids, test_ids = train_test_split(user_ids, test_size=0.2, random_state=5)

with open('./assist17/test_assist17_100_split.pkl', 'wb') as f:
    pickle.dump(test_ids, f)

# Using KFold to split the training set and validation set
kfold = KFold(n_splits=5, shuffle=True, random_state=5)
for i, (train_indices, valid_indices) in enumerate(kfold.split(train_valid_ids)):
    train_ids = train_valid_ids[train_indices]
    valid_ids = train_valid_ids[valid_indices]

    with open(f"./assist17/{i}_train_valid_assist17_100_split.pkl", "wb") as f:
        pickle.dump((train_ids,valid_ids), f)

# Check for internal duplicates
train_ids1 = set(train_ids)  
valid_ids1 = set(valid_ids)
test_ids1 = set(test_ids)

print("Check for internal duplicates:")
train_valid_overlap = train_ids1.intersection(valid_ids1)
if train_valid_overlap:
    print(f"Training set and validation set have duplicate IDs: {train_valid_overlap}")
else:
    print("Training set and validation set have NO duplicate IDs")


print("\nChecking for duplicates with the test set:")
train_test_overlap = train_ids1.intersection(test_ids1)
if train_test_overlap:
    print(f"Training set and test set have duplicate IDs: {train_test_overlap}")
else:
    print("Training set and test set have NO duplicate IDs")

valid_test_overlap = valid_ids1.intersection(test_ids1)
if valid_test_overlap:
    print(f"Validation set and test set have duplicate IDs: {valid_test_overlap}")
else:
    print("Validation set and test set have NO duplicate IDs")

print("\nCheck completed.")

# Save the Processed Data
data.to_csv('./assist17/assist17_pro.csv',index=False)

print('Data processing is complete!')
