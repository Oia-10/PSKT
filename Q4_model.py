import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from Q1_utils import *

class PSKT(torch.nn.Module):
    def __init__(self, q_num, kc_num,  embed_dim):
        super(PSKT, self).__init__()
        self.q_num = q_num
        self.kc_num = kc_num
        self.embed_dim = embed_dim

        self.q_embedding = nn.Embedding(q_num + 1, embed_dim, padding_idx=0) 
        self.c_embedding = nn.Embedding(kc_num + 1, embed_dim, padding_idx=0)
        self.r_embedding = nn.Embedding(2 + 1, embed_dim, padding_idx=2) 
        self.TD_embedding = nn.Embedding(43200 + 1, embed_dim) 

        self.trans_QDiff = nn.Linear(embed_dim, 1)
        self.trans_QAlpha = nn.Linear(embed_dim, 1)

        self.knowledge_init = nn.Parameter(init.xavier_uniform_(torch.rand(1, kc_num + 1, dtype=torch.float32)))

        self.trans_skq = nn.Linear(embed_dim + (kc_num + 1),  (kc_num + 1)) 
        self.trans_ekq = nn.Linear(embed_dim + (kc_num + 1),  (kc_num + 1)) 

        self.ks_gate1 = nn.Linear(2 * (kc_num + 1), (kc_num + 1))
        self.ks_gate2 = nn.Linear(3 * (kc_num + 1), (kc_num + 1))

        self.ka_gate1 = nn.Linear((kc_num + 1) , (kc_num + 1))
        self.ka_gate2 = nn.Linear(2 * (kc_num + 1) , (kc_num + 1))

        self.ki_gate1 = nn.Linear(3 * (kc_num + 1) , (kc_num + 1))
        self.ki_gate2 = nn.Linear(2 * (kc_num + 1) + embed_dim, (kc_num + 1))

        self.fo_gate = nn.Linear(embed_dim * 2 + (kc_num + 1), (kc_num + 1))

        self.sigmoid = nn.Sigmoid()
    
        self.trans_knowledge_all = nn.Sequential(
            nn.Linear(embed_dim + (kc_num+1), embed_dim//2), 
            nn.ReLU(),
            nn.Linear(embed_dim//2, kc_num+1)
        )

        self.a = nn.Parameter(torch.FloatTensor([1]))
        self.b = nn.Parameter(torch.FloatTensor([1]))

        self.trans_G = nn.Linear(embed_dim + kc_num +1, 1)
        self.trans_S = nn.Linear(embed_dim + kc_num +1, 1)

        self.reset_parameters()

       
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)
    

    def forward(self, Q, KC, R, TS):
        Q_embed = self.q_embedding(Q) 
        C_embed = self.c_embedding(KC) 
        R_embed = self.r_embedding(R) 
        
        bs, length = Q_embed.size(0), Q_embed.size(1)

        Q_Diff = self.sigmoid(self.trans_QDiff(Q_embed))
        Q_alpha = self.sigmoid(self.trans_QAlpha(Q_embed))

        QC_encoder = C_embed + Q_embed * (self.a * Q_Diff + self.b * Q_alpha)
    
        slice_qc = torch.chunk(QC_encoder, length, dim=1) 
        slice_r = torch.chunk(R_embed, length, dim=1)
        slice_diff = torch.chunk(Q_Diff, length, dim=1) 
        slice_alpha = torch.chunk(Q_alpha, length, dim=1)

        Conehot = torch.eye(self.kc_num + 1).cuda()
        C_oneHot = Conehot[KC]

        TD = (TS[:,1:] - TS[:,:-1]) // 60  # Time interval (in minutes)
        TD = torch.clamp(TD, min = 0, max = 43200) # Maximum 1 month
        TD_embedding = self.TD_embedding(TD.long()) 
        slice_ts = torch.chunk(TD_embedding, length-1, dim=1)
        
        h_kc_pre =  torch.tile(self.knowledge_init, (bs, 1))
        pred_all = []
        
        for i in range(0, length):
            qc = torch.squeeze(slice_qc[i], 1) 
            qa = torch.squeeze(slice_r[i], 1)  
            diff = torch.squeeze(slice_diff[i], 1)
            alpha = torch.squeeze(slice_alpha[i], 1) 

            skq = self.trans_skq(torch.cat([qc, h_kc_pre],dim=-1)) 
            ekq = self.trans_ekq(torch.cat([qc, h_kc_pre],dim=-1)) 
            resk = ekq - skq

            sin1 = torch.cat([skq, h_kc_pre], dim=-1) 
            ks_title = torch.tanh(self.ks_gate1(sin1))
            sin2 = torch.cat([ekq, skq, resk], dim=-1) 
            ks = self.sigmoid(self.ks_gate2(sin2)) 
            ks =  ks * ks_title 

            ka_title = torch.tanh(resk)  
            ain2 = torch.cat([ekq, ks], dim=-1) 
            ka = self.sigmoid(self.ka_gate2(ain2)) 
            ka = ka * ka_title

            next_info = torch.cat([ka, qc],dim=-1)
            knowledge_for_next_concept =  self.sigmoid(self.trans_knowledge_all(next_info))

            exp1 = torch.exp(-1.702 * 4 * alpha * (knowledge_for_next_concept -  diff)) 
            correct_pred =  1 / (1 + exp1) 
      
            G = self.sigmoid(self.trans_G(next_info)) * 0.5
            S = self.sigmoid(self.trans_S(next_info)) * 0.5
            correct_pred = G * (1 - correct_pred) + (1 - S) * correct_pred 
            
            pred_all.append(correct_pred.unsqueeze(1))

            if i !=length-1:
                iin1 =  torch.cat([resk, skq, ekq], dim=-1) 
                ki_title = (torch.tanh(self.ki_gate1(iin1)) + 1)/2
                iin2 =  torch.cat([ks, ka, qa], dim=-1)
                ki = self.sigmoid(self.ki_gate2(iin2))
                ki = ki * ki_title

                h_kc_pre = h_kc_pre + ki
                qt = torch.squeeze(slice_qc[i+1], 1)  
                ts = torch.squeeze(slice_ts[i], 1) 
                ifo = torch.cat([qt, ts, h_kc_pre], dim=-1)
                foin = self.sigmoid(self.fo_gate(ifo)) 

                h_kc_pre = h_kc_pre * foin

        correct_pred_all = torch.cat(pred_all, dim=1)
        correct_pred = torch.sum(correct_pred_all * C_oneHot, dim=-1) 
        correct_pred = correct_pred[:,1:]

        return correct_pred
       



