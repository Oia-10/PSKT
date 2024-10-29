from torch.utils.data import Dataset
import numpy as np

class KTDataset(Dataset):
    def __init__(self, group, max_seq, min_seq=2):
        self.max_seq = max_seq
        self.group = group
        
        super(KTDataset, self).__init__()

        self.samples = {}
        self.user_ids = []
        for user_id in group.index:
            u, q, kc, r, ts = group[user_id]
            if len(q) < min_seq:
                continue

            if len(q) > self.max_seq:
                total_questions = len(q)
                initial = total_questions % self.max_seq
                if initial >= 2:
                    self.user_ids.append(f"{user_id}_0")
                    self.samples[f"{user_id}_0"] = (u[:initial],q[:initial], kc[:initial], r[:initial],
                                     ts[:initial])
                for seq in range(total_questions // self.max_seq):
                    self.user_ids.append(f"{user_id}_{seq+1}")
                    start = initial + seq * self.max_seq
                    end = initial + (seq + 1) * self.max_seq
                    self.samples[f"{user_id}_{seq+1}"] = (u[start:end],q[start:end], kc[start:end], r[start:end],
                                        ts[start:end])
            else:
                user_id = str(user_id)
                self.user_ids.append(user_id)
                self.samples[user_id] = (u, q, kc, r, ts)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        u_, q_, kc_, r_, ts_ = self.samples[user_id]
        seq_len = len(q_)

        ## for padding
        u = np.zeros(self.max_seq, dtype=int)
        q = np.zeros(self.max_seq, dtype=int)
        kc = np.zeros(self.max_seq, dtype=int)
        r = np.zeros(self.max_seq, dtype=int) + 2
        ts = np.zeros(self.max_seq, dtype=int)
        
        u[0:seq_len] = u_
        q[0:seq_len] = q_
        kc[0:seq_len] = kc_
        r[0:seq_len] = r_
        ts[0:seq_len] = ts_

        return u, q, kc, r, ts