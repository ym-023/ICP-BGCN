import sys,os
sys.path.append(os.getcwd())
from Process.process import *
import torch as th
from torch_scatter import scatter_mean,scatter_add
import torch.nn.functional as F
import numpy as np
from tools.earlystopping import EarlyStopping
#from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from Process.rand5fold import *
from tools.evaluate import *
from torch_geometric.nn import GCNConv
import copy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import matplotlib.pyplot as plt
from datetime import datetime
from torch_geometric.data import DataLoader
from torch.nn import Linear
from torch_geometric.nn import global_max_pool as gmp
import torch.nn.functional as F
from sklearn.manifold import TSNE

from torch_geometric.nn import GCNConv, SAGEConv, GATConv

def visualize(embedding, value):
    z = TSNE(n_components=2).fit_transform(embedding.detach().cpu().numpy())
    plt.figure(figsize=(16, 12))
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0], z[:, 1], s=30, c=value, cmap="Set2", zorder=2)
    plt.show()

class TDrumorGCN(th.nn.Module):
    def __init__(self,TD_dropout_rate,in_feats,hid_feats,out_feats):
        super(TDrumorGCN, self).__init__()
        self.dropout_rate = TD_dropout_rate
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats+in_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.TD_edge_index
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        batch_size = max(data.batch) + 1
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[index].mean(dim=0)
        x = th.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[index].mean(dim=0)
        x = th.cat((x, root_extend), 1)
        x= scatter_mean(x, data.batch, dim=0)
        return x

class BUrumorGCN(th.nn.Module):
    def __init__(self,BU_dropout_rate,in_feats,hid_feats,out_feats):
        super(BUrumorGCN, self).__init__()
        self.dropout_rate = BU_dropout_rate
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats+in_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.BU_edge_index
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        batch_size = max(data.batch) + 1
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[index].mean(dim=0)
        x = th.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[index].mean(dim=0)
        x = th.cat((x, root_extend), 1)
        x= scatter_mean(x, data.batch, dim=0)
        return x


class GCN(th.nn.Module):
    def __init__(self,dropout_rate,TD_dropout_rate,BU_dropout_rate,in_feats,hid_feats,out_feats):
        super(GCN, self).__init__()
		
        self.dropout_rate = dropout_rate
        self.TD_dropout_rate = TD_dropout_rate
        self.BU_dropout_rate = BU_dropout_rate
        self.TDrumorGCN = TDrumorGCN(TD_dropout_rate,in_feats, hid_feats, out_feats)
        self.BUrumorGCN = BUrumorGCN(BU_dropout_rate,in_feats, hid_feats, out_feats)
        self.fc1 = th.nn.Linear(in_feats,out_feats+hid_feats)
        self.fc2 = th.nn.Linear((out_feats+hid_feats)*2,out_feats+hid_feats)
        self.attention = nn.MultiheadAttention(embed_dim=in_feats, num_heads=1)

    def forward(self, data):
        x = data.x
        text, _ = self.attention(x, x, x)
        text = F.relu(text)
        text = self.fc1(text)
        text = F.dropout(text, p=self.dropout_rate, training=self.training)
        #x = scatter_mean(x, data.batch, dim=0)
        batch_size = th.bincount(data.batch)
        text = scatter_add(x, data.batch, dim=0)
        text = text / batch_size.unsqueeze(1)
        text = F.log_softmax(text, dim=1)
		
        TD_x = self.TDrumorGCN(data)
        BU_x = self.BUrumorGCN(data)
        x = th.cat((BU_x,TD_x), 1)
        x=self.fc2(x)
        x = F.log_softmax(x, dim=1)

        fu = th.cat((x,text),1)
        #print(fu.shape)
        #fu = scatter_mean(fu, batch, dim=0)
        fu = F.log_softmax(fu, dim=1)
        return fu

def train_GCN(droprate,x_test, x_train,TDdroprate,BUdroprate,lr, weight_decay,patience,n_epochs,batchsize,dataname,iter):
    model = GCN(droprate,TDdroprate,BUdroprate,in_feats,hid_feats,out_feats).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train() 
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(patience=patience, verbose=True) 

    for epoch in range(n_epochs): 
        traindata_list, testdata_list = loadBiData(dataname, x_train, x_test, TDdroprate,BUdroprate) 
        
        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=5)
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=5)   
        avg_loss = [] 
        avg_acc = []
        batch_idx = 0
        tqdm_train_loader = tqdm(train_loader)
        for Batch_data in tqdm_train_loader:
            Batch_data.to(device)
            out_labels = model(Batch_data)
            finalloss = F.nll_loss(out_labels,Batch_data.y)
            loss = finalloss
            optimizer.zero_grad()
            loss.backward()
            avg_loss.append(loss.item())
            optimizer.step()
            _, pred = out_labels.max(dim=-1)
            correct = pred.eq(Batch_data.y).sum().item()
            train_acc = correct / len(Batch_data.y)
            avg_acc.append(train_acc)
            print("Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(iter,epoch, batch_idx,
                                                                                                 loss.item(),
                                                                                                 train_acc))
            
            batch_idx = batch_idx + 1
            #print('train_loss: ', loss.item())
        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))
        
        temp_val_losses = []
        temp_val_accs = []
        temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
        temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
        temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3, \
        temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        model.eval() 
        tqdm_test_loader = tqdm(test_loader)
        for Batch_data in tqdm_test_loader:
            Batch_data.to(device)
            val_out= model(Batch_data)
            val_loss = F.nll_loss(val_out, Batch_data.y)
            temp_val_losses.append(val_loss.item())
            _, val_pred = val_out.max(dim=1)
            correct = val_pred.eq(Batch_data.y).sum().item()
            val_acc = correct / len(Batch_data.y) 
            Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(
                val_pred, Batch_data.y) 
            temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
                Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
            temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
                Recll2), temp_val_F2.append(F2), \
            temp_val_Acc3.append(Acc3), temp_val_Prec3.append(Prec3), temp_val_Recll3.append(
                Recll3), temp_val_F3.append(F3), \
            temp_val_Acc4.append(Acc4), temp_val_Prec4.append(Prec4), temp_val_Recll4.append(
                Recll4), temp_val_F4.append(F4)
            temp_val_accs.append(val_acc)
        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))
        print("Epoch {:05d} | Val_Loss {:.4f} | Val_Accuracy {:.4f}".format(epoch, np.mean(avg_loss), np.mean(temp_val_losses),
                                                                           np.mean(temp_val_accs)))
        res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
               'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                       np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
               'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                       np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
               'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3),
                                                       np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
               'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc4), np.mean(temp_val_Prec4),
                                                       np.mean(temp_val_Recll4), np.mean(temp_val_F4))]
        print('results:', res)
      
        if epoch > 25:
            early_stopping(np.mean(temp_val_losses), np.mean(temp_val_accs), np.mean(temp_val_F1), np.mean(temp_val_F2),
                        np.mean(temp_val_F3), np.mean(temp_val_F4), model, 'GACL', dataname)
        accs =np.mean(temp_val_accs)
        F1 = np.mean(temp_val_F1)
        F2 = np.mean(temp_val_F2)
        F3 = np.mean(temp_val_F3)
        F4 = np.mean(temp_val_F4)
        if early_stopping.early_stop:
            print("Early stopping")
            accs=early_stopping.accs
            F1=early_stopping.F1
            F2 = early_stopping.F2
            F3 = early_stopping.F3
            F4 = early_stopping.F4
            break
    
    '''
    show_val = list(val_accs)

    dt = datetime.now()
    save_time = dt.strftime('%Y_%m_%d_%H_%M_%S')
    
    fig = plt.figure()
    plt.plot(range(1, len(train_accs) + 1), train_accs, color='b', label='train')
    plt.plot(range(1, len(show_val) + 1), show_val, color='r', label='dev')
    plt.grid()
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend()
    plt.xticks(np.arange(1, len(train_accs), step=4))
    fig.savefig('result/' + '{}_accuracy_{}.png'.format(dataname, save_time))

    fig = plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, color='b', label='train')
    plt.plot(range(1, len(val_losses) + 1), val_losses, color='r', label='dev')
    plt.grid()
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.xticks(np.arange(1, len(train_losses) + 1, step=4))
    fig.savefig('result/' + '{}_loss_{}.png'.format(dataname, save_time))'''

    return train_losses , val_losses ,train_accs, val_accs,accs,F1,F2,F3,F4


scale = 1
lr=0.0005 * scale
weight_decay=1e-4
patience=10
n_epochs=200
batchsize=120
out_dim = 128
datasetname=sys.argv[1] #"Twitter15"、"Twitter16"
iterations=int(sys.argv[2]) 
droprate = 0.2
TDdroprate=0.2
BUdroprate=0.2
in_feats = 768
hid_feats = 64
out_feats = 64

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
test_accs = [] 
NR_F1 = [] # NR
FR_F1 = [] # FR
TR_F1 = [] # TR
UR_F1 = [] # UR

for iter in range(iterations):
    fold0_x_test, fold0_x_train, \
    fold1_x_test,  fold1_x_train,  \
    fold2_x_test, fold2_x_train, \
    fold3_x_test, fold3_x_train, \
    fold4_x_test,fold4_x_train = load5foldData(datasetname)
    train_losses, val_losses, train_accs, val_accs0, accs0, F1_0, F2_0, F3_0, F4_0 = train_GCN(droprate,
                                                                                               fold0_x_test,
                                                                                               fold0_x_train,
                                                                                               TDdroprate,BUdroprate,
                                                                                               lr, weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter)
    train_losses, val_losses, train_accs, val_accs1, accs1, F1_1, F2_1, F3_1, F4_1 = train_GCN(droprate,
                                                                                               fold1_x_test,
                                                                                               fold1_x_train,
                                                                                               TDdroprate,BUdroprate, lr,
                                                                                               weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter)
    train_losses, val_losses, train_accs, val_accs2, accs2, F1_2, F2_2, F3_2, F4_2 = train_GCN(droprate,
                                                                                               fold2_x_test,
                                                                                               fold2_x_train,
                                                                                               TDdroprate,BUdroprate, lr,
                                                                                               weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter)
    train_losses, val_losses, train_accs, val_accs3, accs3, F1_3, F2_3, F3_3, F4_3 = train_GCN(droprate,
                                                                                               fold3_x_test,
                                                                                               fold3_x_train,
                                                                                               TDdroprate,BUdroprate, lr,
                                                                                               weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter)
    train_losses, val_losses, train_accs, val_accs4, accs4, F1_4, F2_4, F3_4, F4_4 = train_GCN(droprate,
                                                                                               fold4_x_test,
                                                                                               fold4_x_train,
                                                                                               TDdroprate,BUdroprate, lr,
                                                                                               weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter)
    test_accs.append((accs0+accs1+accs2+accs3+accs4)/5)
    NR_F1.append((F1_0+F1_1+F1_2+F1_3+F1_4)/5)
    FR_F1.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
    TR_F1.append((F3_0 + F3_1 + F3_2 + F3_3 + F3_4) / 5)
    UR_F1.append((F4_0 + F4_1 + F4_2 + F4_3 + F4_4) / 5)
print("Total_Test_Accuracy: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}".format(
    sum(test_accs) / iterations, sum(NR_F1) /iterations, sum(FR_F1) /iterations, sum(TR_F1) / iterations, sum(UR_F1) / iterations))
