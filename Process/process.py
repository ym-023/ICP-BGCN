import os
from Process.dataset import BiGraphDataset

cwd=os.getcwd()

def loadTree(dataname):
    treePath = os.path.join(cwd,'data/'+dataname+'/data.TD_RvNN.vol_5000.txt')
    print("reading twitter tree")
    treeDic = {}
    for line in open(treePath):
        line = line.rstrip()
        eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
        max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]
        if not treeDic.__contains__(eid):
            treeDic[eid] = {}
        treeDic[eid][indexC] = {'parent': indexP, 'max_degree': max_degree, 'maxL': maxL, 'vec': Vec}
    print('tree no:', len(treeDic))
    return treeDic



def loadBiData(dataname, fold_x_train, fold_x_test, TDdroprate,BUdroprate):
    print("loading train set", )
    traindata_list = BiGraphDataset(fold_x_train, dataname = dataname,tddroprate=TDdroprate, budroprate=BUdroprate)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = BiGraphDataset(fold_x_test,dataname = dataname,tddroprate=0, budroprate=0)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list
