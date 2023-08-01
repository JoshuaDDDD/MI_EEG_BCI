import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import DataLoader, Dataset

class load_shu_mi():
    def __init__(self, sub_i, ses_j, num_cls=2):
        self.sub_i, self.ses_j = sub_i, ses_j
        self.data = self.load_mat()
        self.data_split = self.split_by_label(num_cls)

    def data_select(self):
        sub, ses, str_i = "sub-0", "ses-0", "0"
        if self.sub_i < 10:
            str_i += str(self.sub_i)
        else:
            str_i = str(self.sub_i)
        sub += str_i
        ses += str(self.ses_j)
        return "dataset\\shu_motor_2022\\" + sub + "_" + ses + "_task_motorimagery_eeg.mat"
    
    def load_mat(self):
        data = loadmat(self.data_select())
        s = data['data']
        label = data['labels']
        return s, label.squeeze()
    
    def split_by_label(self, num_cls):
        split_list = []
        for i in range(num_cls):
            index = (self.data[1] == i+1)
            split_list.append(self.data[0][index, :, :])
        return split_list
    
def my_data_set_corss_sub(te_ses=1, tr_per=0.8, num_cls=2):
    tr_set, cv_set, te_set = [], [], []
    tr_lable, cv_lable, te_lable = [], [], []
    for i in range(25):
         sub_i = i+1
         for j in range(5):
            ses_j = j+1
            data = load_shu_mi(sub_i=sub_i, ses_j=ses_j).data_split

            np.random.seed(2023)
            n = [data[i].shape[0] for i in range(num_cls)]
            index_list = np.random.randn(np.max(n))
            index_list = [index_list[:n[0]], index_list[:n[1]]]
            index_bool_list = [index > tr_per for index in index_list]
             
            if ses_j == te_ses:
                te = torch.cat([torch.from_numpy(d) for d in data], dim=0)
                t_lab = torch.cat([torch.ones(n[i])*i for i in range(num_cls)])
                te_set.append(te)
                te_lable.append(t_lab)
            else:
                tr = torch.cat([torch.from_numpy(data[i][np.array([index_bool_list[i]==False]).squeeze(), :, :]) for i in range(num_cls)], dim=0)
                cv = torch.cat([torch.from_numpy(data[i][index_bool_list[i], :, :]) for i in range(num_cls)], dim=0)
                tr_lab = torch.cat([torch.ones(np.array([index_bool_list[i]==False]).sum())*i for i in range(num_cls)])
                cv_lab = torch.cat([torch.ones(np.array(index_bool_list[i]).sum())*i for i in range(num_cls)])
                tr_set.append(tr)
                tr_lable.append(tr_lab)
                cv_set.append(cv)
                cv_lable.append(cv_lab)
    tr_set_cross_sub, cv_set_cross_sub = torch.cat(tr_set, dim=0), torch.cat(cv_set, dim=0)
    tr_lable_cross_sub, cv_lable_cross_sub = torch.cat(tr_lable), torch.cat(cv_lable)
    """
    tr_set_cross_sub.shape = [N_tr_cs, num_channel, num_T_point]
    tr_lable_cross_sub.shape = [N_tr_cs, ]
    cv_set_cross_sub.shape = [N_cv_cs, num_channel, num_T_point]
    cv_lable_cross_sub.shape = [N_tr_cs, ]
    te_set = [sub1_teses, ..., sub25_teses]
    """
    return tr_set_cross_sub, cv_set_cross_sub, \
            tr_lable_cross_sub, cv_lable_cross_sub,\
            te_set, te_lable

class MyDataset(Dataset):
    def __init__(self, data, label=None):
        if label is None:
            self.y = label
        else:
            self.y = torch.FloatTensor(label)
        x = torch.FloatTensor(data)
        self.x = x.unsqueeze(1)

    def __getitem__(self, index):
        if self.y is None:
            return self.x[index]
        else:
            return self.x[index], torch.unsqueeze(self.y[index], 0)

    def __len__(self):
        return len(self.x)

class MySemiDataset(Dataset):
    def __init__(self, data, label=None):
        self.y = torch.FloatTensor(label)
        x = torch.FloatTensor(data)
        self.x = x.unsqueeze(1)

    def __getitem__(self, index):
        A_cls = self.y[index]
        Anchor = self.x[index, :, :, :]
        Positive, Negative = self.__get_pn__(A_cls)
        return Anchor, Positive, Negative, A_cls.to(torch.long)

    def __len__(self):
        return len(self.y)
    
    def __get_pn__(self, A_cls):
        P, N = None, None
        while P is None or N is None:
            idx = torch.randint(0, self.__len__(), (1,)).squeeze()
            if P is None and self.y[idx] == A_cls:
                P = self.x[idx, :, :, :]
            elif N is None and self.y[idx] != A_cls:
                N = self.x[idx, :, :, :]
        return P, N

if __name__=='__main__':
    Data = my_data_set_corss_sub(te_ses=1, tr_per=0.8, num_cls=2)
    tr_set, cv_set = MyDataset(Data[0], Data[2]), MyDataset(Data[1], Data[3])
    tr_len, cv_len = tr_set.__len__(), cv_set.__len__()
    print(tr_len, cv_len)