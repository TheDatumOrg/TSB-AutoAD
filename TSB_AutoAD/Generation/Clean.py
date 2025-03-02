#  This function is adapted from [AutoOD_Demo] by [dhofmann34]
#  Original source: [https://github.com/dhofmann34/AutoOD_Demo]

import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.utils import data
import torch.optim as optim
from torch.autograd import Variable
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import metrics

from TSB_AD.evaluation.metrics import get_metrics
import time
from sklearn.calibration import CalibratedClassifierCV
torch.manual_seed(0)
learning_rate = 0.01
log_interval = 10

class Net(nn.Module):
    def __init__(self, dim = 10):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(dim, 50)
        self.fc2 = nn.Linear(50, 100)
        self.fc3 = nn.Linear(100,50)
        self.fc4 = nn.Linear(50,2)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_device():
    if torch.cuda.is_available():  
        # print("GPU detected, use gpu")
        dev = "cuda:0" 
    else:  
        dev = "cpu" 
    return dev 

def inference_NN(net, testing_X, testing_y = None):
    dev = get_device()
    device = torch.device(dev)
    test_dataloader = data.DataLoader(data.TensorDataset(torch.tensor(testing_X,device=device), torch.tensor(testing_y,device=device)), 
                                      batch_size=100, shuffle=False) 
    net.eval()
    predict_proba = []
    for batch_idx, (input_data, target) in enumerate(test_dataloader):
        input_data = Variable(input_data)
        net_out = net(input_data.float())
        predict_proba.append(F.softmax(net_out, dim=1).data.cpu().numpy())
    return np.concatenate(predict_proba)

def run_NN(X,y, epochs = 3,  dim = 10, train_batch_size=100,eval_batch_size=1, return_loss=False):
    dev = get_device()
    device = torch.device(dev)
    net = Net(dim)
    net = net.to(device)
    # create a stochastic gradient descent optimizer
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    # create a loss function
    criterion = nn.NLLLoss()
    # create dataset
    tensor_x = torch.tensor(X, device=device) # transform to torch tensor
    tensor_y = torch.tensor(y,dtype=torch.long, device=device)
    my_dataset = data.TensorDataset(tensor_x,tensor_y) # create your datset
#     class_sample_count = np.array([len(np.where(y == t)[0]) for t in np.unique(y)])
#     weight = 1. / class_sample_count
#     samples_weight = np.array([weight[t] for t in y])
#     samples_weight = torch.from_numpy(samples_weight)
#     samples_weight = samples_weight.double()
#     sampler = data.WeightedRandomSampler(samples_weight, len(samples_weight))
    train_dataloader = data.DataLoader(my_dataset, batch_size=train_batch_size, shuffle = True) # create your dataloader
    
    # run the main training loop
    for epoch in range(epochs):
        net.train()
        for batch_idx, (input_data, target) in enumerate(train_dataloader):
            input_data, target = Variable(input_data), Variable(target)
            #input_data = input_data.to(device)
            #target = input_data.to(device)
            net_out = net(input_data.float())
            loss = criterion(net_out, target.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        net.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        top2 = AverageMeter()
    
    if return_loss:
        net.eval()
        criterion = nn.NLLLoss(reduce=False)
         
        top1 = AverageMeter()
        top2 = AverageMeter()
        loss_list = []
        for batch_idx, (input_data, target) in enumerate(data.DataLoader(my_dataset, batch_size=eval_batch_size, shuffle=False)):
            input_data, target = Variable(input_data), Variable(target)
            net_out = net(input_data.float())
    #             print(F.softmax(net_out, dim=1))
            loss = criterion(net_out, target)
            prec = accuracy(net_out.data, target)
            loss_list.append(loss.data.cpu().numpy())
            top1.update(prec[0], input_data.size(0))
        # print('Final Training Result: '
        #           'Prec @ 1 {top1.avg:.3f}%'.format(top1=top1))   
        return np.concatenate(loss_list,axis=0), net
    else:
        return None, net


# Working solution: Prune points using NN, add good prediction results back
def Label_Clean(X, y, L, det_scores, max_iteration=20, initial_set='Individual', ratio_to_remove=0.05, separate_inline_outlier=False, inlier=0.0001, outlier=0.9999, early_stop=True, slidingWindow=0, n_jobs=1):

    dim = np.shape(X)[1]
    remain_points = np.array(range(len(y)))

    # store clf
    # clf_predict_proba_X_list = []

    label_of_point = np.full((len(y)), 0)

    if initial_set=='Majority':     # Case 1: anomalies which 25% of detectors agree on
        label_of_point[np.sum(L, axis = 1) > np.shape(L)[1]/4] = 1
    elif initial_set=='Ratio':        # Case 2: Top 15% detected
        threshold = np.sort(np.sum(det_scores, axis = 1))[::-1][int(len(y)*0.15)]
        label_of_point[np.sum(det_scores, axis = 1) >= threshold] = 1
    elif initial_set=='Average':        # Case 3: avg ens
        ens_scores = np.mean(det_scores, axis=1)
        threshold = np.mean(ens_scores) + np.std(ens_scores)
        label_of_point[ens_scores > threshold] = 1
    elif initial_set=='Individual':        # Case 4: Top 5% anomalies from each model
        for i in range(det_scores.shape[1]):
            sorted_nums = np.sort(det_scores[:, i])[::-1]
            threshold = sorted_nums[int(det_scores.shape[0] * 0.05)]
            label_of_point[det_scores[:, i] > threshold] = 1
    else:
        raise Exception('None defined initial_set:{}'.format(initial_set)) 

    print('Num of initial abnormal points: ', list(label_of_point).count(1))

    transformer = RobustScaler().fit(X)
    X_transformed = transformer.transform(X)

    clf_X = None
    for i_range in range(0, max_iteration):
        # print('Iteration = {}'.format(i_range))
        # print('F1 for training data:', metrics.f1_score(y[remain_points], label_of_point[remain_points]))
        # print('accuracy for training data:', metrics.accuracy_score(y[remain_points], label_of_point[remain_points]))
        if (len(np.unique(label_of_point[remain_points])) < 2):
            print('Only one class left...')
            break
        
        t1 = time.time()
        clf_X = SVC(gamma='auto', probability=True, random_state=0)
        clf_X.fit(X_transformed[remain_points], label_of_point[remain_points])
        clf_predict_proba_X = clf_X.predict_proba(X_transformed)[:,1]
        # print("AUC_ROC score from SVM:", metrics.roc_auc_score(y, clf_predict_proba_X))
        print('Time spent on LinearSVC: ', time.time()-t1)

        temp_remain_points = remain_points.copy()
        # start pruning points
        loss_list, model = run_NN(X_transformed[temp_remain_points],label_of_point[temp_remain_points], 3, dim = dim, train_batch_size=512, eval_batch_size=512, return_loss=True)
        
        if not separate_inline_outlier:
            # loss_threshold = np.sort(loss_list)[::-1][int(ratio_to_remove * len(loss_list))]
            # print(min(loss_list), max(loss_list), loss_threshold, np.mean(loss_list)+ np.std(loss_list))
            loss_threshold = np.mean(loss_list) + np.std(loss_list)
            points_to_remove = temp_remain_points[(loss_list > loss_threshold)]
        else:
            inlier_labels = np.where(label_of_point[temp_remain_points] == 0)[0]
            # loss_threshold = np.sort(loss_list[inlier_labels])[::-1][int(ratio_to_remove * len(loss_list[inlier_labels]))]
            loss_threshold = np.mean(loss_list[inlier_labels])+ np.std(loss_list[inlier_labels])
            points_to_remove = temp_remain_points[inlier_labels][(loss_list[inlier_labels] > loss_threshold)]

            outlier_labels = np.where(label_of_point[temp_remain_points] == 1)[0]
            loss_threshold = np.sort(loss_list[outlier_labels])[::-1][int(ratio_to_remove * len(loss_list[outlier_labels]))]
            # loss_threshold = np.mean(loss_list[outlier_labels])+ np.std(loss_list[outlier_labels])
            points_to_remove = np.append(points_to_remove, temp_remain_points[outlier_labels][(loss_list[outlier_labels] > loss_threshold)])

        _, model = run_NN(X_transformed[temp_remain_points],label_of_point[temp_remain_points],10, dim = dim, train_batch_size=512, eval_batch_size=512, return_loss=False)
        predict_proba = inference_NN(model, X_transformed, y)[:,1]
        # print("F-1 score from NN:", metrics.f1_score(y, np.array([int(i) for i in predict_proba > 0.5])))
        # print("AUC_ROC score from NN:", metrics.roc_auc_score(y, predict_proba))

        temp_remain_points = np.setdiff1d(np.array(temp_remain_points), points_to_remove)

        predict_outlier_indexes = np.where(predict_proba > outlier)[0]
        new_outlier_indexes = np.setdiff1d(predict_outlier_indexes, temp_remain_points)
        # new_outlier_indexes = new_outlier_indexes[label_of_point[new_outlier_indexes]==0]
        # print('new_outlier_indexes: ', new_outlier_indexes)
        # print(f'Number of new points with confidence > {outlier}: {len(new_outlier_indexes)}')
        if(len(new_outlier_indexes) > 0):
            # print('F-1 before: ', metrics.f1_score(y[new_outlier_indexes], label_of_point[new_outlier_indexes]))
            label_of_point[new_outlier_indexes] = 1
            # print('F-1 after: ', metrics.f1_score(y[new_outlier_indexes], label_of_point[new_outlier_indexes]))
            temp_remain_points = np.union1d(temp_remain_points, predict_outlier_indexes)

        predict_inlier_indexes = np.where(predict_proba < inlier)[0]
        new_inlier_indexes = np.setdiff1d(predict_inlier_indexes, temp_remain_points)
        # new_inlier_indexes = new_inlier_indexes[label_of_point[new_inlier_indexes]==1]
        # print('new_inlier_indexes: ', new_inlier_indexes)
        # print(f'Number of points with confidence < {inlier}', len(new_inlier_indexes))
        
        if(len(new_inlier_indexes) > 0):
            label_of_point[new_inlier_indexes] = 0
            temp_remain_points = np.union1d(temp_remain_points, predict_inlier_indexes)
            # if(len(new_outlier_indexes) <= len(points_to_remove)):
        # print("numAdd: ", (len(new_outlier_indexes) + len(new_inlier_indexes)), "  - numRmv: ", len(points_to_remove))
        
        if(early_stop and len(new_outlier_indexes) + len(new_inlier_indexes) >= len(points_to_remove)):
        # if len(remain_points) < np.shape(L)[0]/3:
            clf_X = SVC(gamma='auto', probability=True, random_state=0)
            clf_X.fit(X_transformed[remain_points], label_of_point[remain_points])
            clf_predict_proba_X = clf_X.predict_proba(X_transformed)[:,1]
            # print("AUC_ROC score from SVM early:", metrics.roc_auc_score(y, clf_predict_proba_X))
            # print("Early Stop ...")
            break
        else:
            remain_points = temp_remain_points
    
    return clf_predict_proba_X

def training_clean_dev(X, y, L, det_scores, max_iteration=20, initial_set='avg', ratio_to_remove=0.05, separate_inline_outlier=False, inlier=0.0001, outlier=0.9999, early_stop=True, slidingWindow=0):

    dim = np.shape(X)[1]
    remain_points = np.array(range(len(y)))

    # store clf
    # clf_predict_proba_X_list = []

    label_of_point = np.full((len(y)), 0)

    if initial_set=='majority':     # Case 1: anomalies which 25% of detectors agree on
        label_of_point[np.sum(L, axis = 1) > np.shape(L)[1]/4] = 1
    elif initial_set=='ratio':        # Case 2: Top 15% detected
        threshold = np.sort(np.sum(det_scores, axis = 1))[::-1][int(len(y)*0.15)]
        label_of_point[np.sum(det_scores, axis = 1) >= threshold] = 1
    elif initial_set=='avg':        # Case 3: avg ens
        ens_scores = np.mean(det_scores, axis=1)
        threshold = np.mean(ens_scores) + np.std(ens_scores)
        label_of_point[ens_scores > threshold] = 1
    elif initial_set=='individual':        # Case 4: Top 5% anomalies from each model
        for i in range(det_scores.shape[1]):
            sorted_nums = np.sort(det_scores[:, i])[::-1]
            threshold = sorted_nums[int(len(y)*0.05)]
            label_of_point[det_scores[:, i] >= threshold] = 1
    else:
        raise Exception('None defined initial_set:{}'.format(initial_set)) 

    print('Num of initial abnormal points: ', list(label_of_point).count(1))

    transformer = RobustScaler().fit(X)
    X_transformed = transformer.transform(X)

    clf_X = None
    for i_range in range(0, max_iteration):
        # print('Iteration = {}'.format(i_range))
        # print('F1 for training data:', metrics.f1_score(y[remain_points], label_of_point[remain_points]))
        # print('accuracy for training data:', metrics.accuracy_score(y[remain_points], label_of_point[remain_points]))
        if (len(np.unique(label_of_point[remain_points])) < 2):
            # print('Only one class left...')
            break

        t1 = time.time()
        # clf_X = SVC(gamma='auto', probability=True, random_state=0)
        linear_svc = LinearSVC(random_state=0)
        clf_X = CalibratedClassifierCV(linear_svc)
        clf_X.fit(X_transformed[remain_points], label_of_point[remain_points])
        clf_predict_proba_X = clf_X.predict_proba(X_transformed)[:,1]
        # print("AUC_ROC score from SVM:", metrics.roc_auc_score(y, clf_predict_proba_X))
        # print('Time spent on LinearSVC: ', time.time()-t1)

        temp_remain_points = remain_points.copy()
        # start pruning points
        loss_list, model = run_NN(X_transformed[temp_remain_points],label_of_point[temp_remain_points], 3, dim = dim, train_batch_size=512, eval_batch_size=512, return_loss=True)
        
        if not separate_inline_outlier:
            # loss_threshold = np.sort(loss_list)[::-1][int(ratio_to_remove * len(loss_list))]
            # print(min(loss_list), max(loss_list), loss_threshold, np.mean(loss_list)+ np.std(loss_list))
            loss_threshold = np.mean(loss_list) + np.std(loss_list)
            points_to_remove = temp_remain_points[(loss_list > loss_threshold)]
        else:
            inlier_labels = np.where(label_of_point[temp_remain_points] == 0)[0]
            # loss_threshold = np.sort(loss_list[inlier_labels])[::-1][int(ratio_to_remove * len(loss_list[inlier_labels]))]
            loss_threshold = np.mean(loss_list[inlier_labels])+ np.std(loss_list[inlier_labels])
            points_to_remove = temp_remain_points[inlier_labels][(loss_list[inlier_labels] > loss_threshold)]

            outlier_labels = np.where(label_of_point[temp_remain_points] == 1)[0]
            loss_threshold = np.sort(loss_list[outlier_labels])[::-1][int(ratio_to_remove * len(loss_list[outlier_labels]))]
            # loss_threshold = np.mean(loss_list[outlier_labels])+ np.std(loss_list[outlier_labels])
            points_to_remove = np.append(points_to_remove, temp_remain_points[outlier_labels][(loss_list[outlier_labels] > loss_threshold)])

        _, model = run_NN(X_transformed[temp_remain_points],label_of_point[temp_remain_points],10, dim = dim, train_batch_size=512, eval_batch_size=512, return_loss=False)
        predict_proba = inference_NN(model, X_transformed, y)[:,1]
        # print("F-1 score from NN:", metrics.f1_score(y, np.array([int(i) for i in predict_proba > 0.5])))
        # print("AUC_ROC score from NN:", metrics.roc_auc_score(y, predict_proba))

        temp_remain_points = np.setdiff1d(np.array(temp_remain_points), points_to_remove)

        predict_outlier_indexes = np.where(predict_proba > outlier)[0]
        new_outlier_indexes = np.setdiff1d(predict_outlier_indexes, temp_remain_points)
        # new_outlier_indexes = new_outlier_indexes[label_of_point[new_outlier_indexes]==0]
        # print('new_outlier_indexes: ', new_outlier_indexes)
        # print(f'Number of new points with confidence > {outlier}: {len(new_outlier_indexes)}')
        if(len(new_outlier_indexes) > 0):
            # print('F-1 before: ', metrics.f1_score(y[new_outlier_indexes], label_of_point[new_outlier_indexes]))
            label_of_point[new_outlier_indexes] = 1
            # print('F-1 after: ', metrics.f1_score(y[new_outlier_indexes], label_of_point[new_outlier_indexes]))
            temp_remain_points = np.union1d(temp_remain_points, predict_outlier_indexes)

        predict_inlier_indexes = np.where(predict_proba < inlier)[0]
        new_inlier_indexes = np.setdiff1d(predict_inlier_indexes, temp_remain_points)
        # new_inlier_indexes = new_inlier_indexes[label_of_point[new_inlier_indexes]==1]
        # print('new_inlier_indexes: ', new_inlier_indexes)
        # print(f'Number of points with confidence < {inlier}', len(new_inlier_indexes))
        
        if(len(new_inlier_indexes) > 0):
            label_of_point[new_inlier_indexes] = 0
            temp_remain_points = np.union1d(temp_remain_points, predict_inlier_indexes)
            # if(len(new_outlier_indexes) <= len(points_to_remove)):
        # print("numAdd: ", (len(new_outlier_indexes) + len(new_inlier_indexes)), "  - numRmv: ", len(points_to_remove))
        
        if(early_stop and len(new_outlier_indexes) + len(new_inlier_indexes) >= len(points_to_remove)):
        # if len(remain_points) < np.shape(L)[0]/3:
            # clf_X = SVC(gamma='auto', probability=True, random_state=0)
            linear_svc = LinearSVC(random_state=0)
            clf_X = CalibratedClassifierCV(linear_svc)
            clf_X.fit(X_transformed[remain_points], label_of_point[remain_points])
            clf_predict_proba_X = clf_X.predict_proba(X_transformed)[:,1]
            # print("AUC_ROC score from SVM early:", metrics.roc_auc_score(y, clf_predict_proba_X))
            # print("Early Stop ...")
            break
        else:
            remain_points = temp_remain_points
    
    return clf_predict_proba_X