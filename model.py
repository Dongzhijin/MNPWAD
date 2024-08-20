import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import torch.nn.functional as F


class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.LeakyReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

class AEPretrain(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class AE(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LeakyReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.LeakyReLU(),
            nn.Linear(hidden_dims[0], input_dim),
            nn.LeakyReLU(),
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class AnomalyScore(nn.Module):
    def __init__(self,hidden_dim=8,recon_dim=1,sim_dim=1):
        super().__init__()
        D=hidden_dim+recon_dim+sim_dim
        self.regression=nn.Sequential(
            nn.Linear(D,16),
            nn.LeakyReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x_hidden, x_recon,loss_anchor):
        x=torch.cat((x_hidden,x_recon,loss_anchor),1)
        score=self.regression(x)
        return score



class MNP(nn.Module):
    def __init__(self,input_dim, hidden_dims,loss_type='smooth',device='cuda',PretrainAE=None,multinormalprototypes=None,n_prototypes=2,AS_type='AnomalyScore',alpha=1,):
        super().__init__()
        if multinormalprototypes is None:
            self.multinormalprototypes=nn.parameter.Parameter(torch.rand(n_prototypes,hidden_dims[-1]))
            self.AE=AE(input_dim=input_dim,hidden_dims=hidden_dims)
        else:
            self.multinormalprototypes=nn.parameter.Parameter(multinormalprototypes)
            self.AE=PretrainAE
        self.AS=eval(AS_type)(hidden_dims[-1])
        self.device=device
        self.alpha=alpha
        if loss_type == 'mse':
            self.loss_reg = torch.nn.MSELoss(reduction='none')
        elif loss_type == 'mae':
            self.loss_reg = torch.nn.MAELoss(reduction='none')
        elif loss_type == 'smooth':
            self.loss_reg = torch.nn.SmoothL1Loss(
                reduction='none',
                # beta=beta
            )
        else:
            raise ValueError('unsupported loss')
        
    def forward(self, x,output_hidden=False):
        x_encoded,x_decoded=self.AE(x)
        residual_recon=self.loss_reg(x,x_decoded)
        loss_recon=residual_recon.mean(dim=1)
        
        anchors=F.normalize(self.multinormalprototypes,dim=1) 
        x_encoded=F.normalize(x_encoded,dim=1)
        anchors_tmp=anchors.unsqueeze(0)
        x_encoded_tmp=x_encoded.unsqueeze(1) 
        dis_anchor=torch.sum((x_encoded_tmp-anchors_tmp)**2,dim=2)
        sim_anchor=(1+dis_anchor/self.alpha)**(-(self.alpha+1)/2) 
        sim_anchor_max,_=torch.max(sim_anchor,dim=1)  
        
        score=self.AS(x_encoded,loss_recon.unsqueeze(1),sim_anchor_max.unsqueeze(1))
        if output_hidden:
            return loss_recon,score,sim_anchor,x_encoded,anchors
        return loss_recon,score,sim_anchor

class Loss_MNP(torch.nn.Module):
    def __init__(self,score_loss='mse', device='cuda',T=2,m1=0.02,lambda_kl=1,beta=1):
        super(Loss_MNP, self).__init__()
        self.device=device
        self.T = T
        self.m1=m1
        self.lambda_kl=lambda_kl
        self.beta=beta
        if score_loss == 'mse':
            self.loss_reg = torch.nn.MSELoss(reduction='none')
        elif score_loss == 'mae':
            self.loss_reg = torch.nn.MAELoss(reduction='none')
        elif score_loss == 'smooth':
            self.loss_reg = torch.nn.SmoothL1Loss(
                reduction='none',
                # beta=beta
            )
        else:
            raise ValueError('unsupported loss')
        self.sigmoid = nn.Sigmoid()
        self.relu=nn.ReLU()
        
    def forward(self, basenet,x_pos,x_neg,pre_recon_loss, pre_score_loss,pre_anchor_loss):
        recon_pos,score_pos,sim_anchor_pos=basenet(x_pos)
        recon_neg,score_neg,sim_anchor_neg=basenet(x_neg)
        neg_weight=self.sigmoid(self.beta*torch.max(sim_anchor_neg,dim=1)[0])
        neg_weight=(neg_weight/neg_weight.sum())*neg_weight.shape[0]
        loss_recon_neg=(recon_neg*neg_weight).mean()
        loss_recon_pos=self.relu(-recon_pos+loss_recon_neg+self.m1).mean()
        loss_recon=loss_recon_neg+loss_recon_pos
        
        loss_score_pos=self.loss_reg(score_pos,torch.ones(score_pos.shape).to(self.device)) # 1
        loss_score_neg=self.loss_reg(score_neg,torch.zeros(score_neg.shape).to(self.device)) # 0
        loss_score=loss_score_pos.mean()+(loss_score_neg*neg_weight).mean()
        
        rsim_anchor_neg=sim_anchor_neg/torch.sum(sim_anchor_neg,dim=1,keepdim=True)  
        f_anchors=torch.sum(rsim_anchor_neg,dim=0,keepdim=True)
        target_d=torch.square(rsim_anchor_neg)/f_anchors
        rtarget_d=target_d/torch.sum(target_d,dim=1,keepdim=True)  
        loss_kl=torch.sum(rtarget_d * (torch.log(rtarget_d) - torch.log(rsim_anchor_neg)))
        
        max_sim_anchor_pos_mean=(torch.max(sim_anchor_pos,dim=1)[0]).mean()
        max_sim_anchor_neg_mean=(neg_weight*(torch.max(sim_anchor_neg,dim=1)[0])).mean()
        loss_anchor=-torch.log(self.sigmoid(-max_sim_anchor_pos_mean+max_sim_anchor_neg_mean))
    
        loss_anchor+=self.lambda_kl*loss_kl
        #dynamic weight averaging
        k1 = torch.exp((loss_recon / pre_recon_loss) / self.T) if pre_recon_loss != 0 else 0
        k2 = torch.exp((loss_score / pre_score_loss) / self.T) if pre_score_loss != 0 else 0
        k3 = torch.exp((loss_anchor / pre_anchor_loss) / self.T) if pre_anchor_loss != 0 else 0
    
        loss = (k1 / (k1 + k2+k3)) * loss_recon + (k2 / (k1 + k2+k3)) * loss_score+(k3 / (k1 + k2+k3)) * loss_anchor  
        return loss,(loss_recon,loss_recon_pos,loss_recon_neg),\
            (loss_score,loss_score_pos.mean(),(loss_score_neg*neg_weight).mean()),\
                (loss_anchor,max_sim_anchor_pos_mean,max_sim_anchor_neg_mean) 

