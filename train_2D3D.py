import argparse
import json
import os

from torch import optim
from torch.utils.data import DataLoader

from src.metrics import *
from src.model import *
from src.utils import *


def train(model, optimizer, trainingData, metrics, class_weights, labels):
    model.train()
    loss_batch = 0
    batch_count = 0
    is_fst_loss = True
    loader_len = len(trainingData)
    turn_point = int(loader_len / args.batch_size) * args.batch_size + loader_len % args.batch_size - 1

    for cnt, batch in enumerate(trainingData):
        batch_count += 1

        # Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
        loss_mask, V_obs, A_obs, V_tr, A_tr, obs_classes = batch
        optimizer.zero_grad()
        # Forward
        V_obs_tmp = V_obs.permute(0, 3, 1, 2).contiguous() 
        V_pred, _ = model(V_obs_tmp, A_obs.squeeze(), obs_classes)

        V_pred = V_pred.permute(0, 2, 3, 1).contiguous() 

        V_tr = V_tr.squeeze() # pred traj gt
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze() # pred traj

        if batch_count % args.batch_size != 0 and cnt != turn_point:
            l = graph_loss(V_pred, V_tr, obs_classes[0], class_weights, labels)
            if is_fst_loss:
                loss = l
                is_fst_loss = False
            else:
                loss = l + loss

        else:
            loss = loss / args.batch_size
            is_fst_loss = True
            loss.backward()

            optimizer.step()
            # Metrics
            loss_batch = loss.item() + loss_batch
            #print('TRAIN:', '\t Epoch:', epoch, '\t Loss:', loss_batch / batch_count)

    metrics['train_loss'].append(loss_batch / batch_count)


def valid(model, validationData, metrics, class_weights, labels):
    model.eval()
    loss_batch = 0
    batch_count = 0
    is_fst_loss = True
    loader_len = len(validationData)
    turn_point = int(loader_len / args.batch_size) * args.batch_size + loader_len % args.batch_size - 1

    for cnt, batch in enumerate(validationData):
        batch_count += 1

        # Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
        loss_mask, V_obs, A_obs, V_tr, A_tr, obs_classes = batch

        V_obs_tmp = V_obs.permute(0, 3, 1, 2).contiguous()

        V_pred, _ = model(V_obs_tmp, A_obs.squeeze(), obs_classes)

        V_pred = V_pred.permute(0, 2, 3, 1).contiguous()

        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()

        if batch_count % args.batch_size != 0 and cnt != turn_point:
            l = graph_loss(V_pred, V_tr, obs_classes[0], class_weights, labels)
            if is_fst_loss:
                loss = l
                is_fst_loss = False
            else:
                loss = l + loss

        else:
            loss = loss / args.batch_size
            is_fst_loss = True
            # Metrics
            loss_batch = loss.item() + loss_batch
            #print('VALD:', '\t Epoch:', epoch, '\t Loss:', loss_batch / batch_count)

    metrics['val_loss'].append(loss_batch / batch_count)



def graph_loss(V_pred, V_target, obs_classes, class_weights, labels):
    if args.dataset == '2D':
        return bivariate_loss(V_pred, V_target, obs_classes, class_weights, labels)
    if args.dataset == '3D':
        return skeleton_loss(V_pred, V_target, obs_classes, class_weights, labels)


def start_training(data_set, num_epochs=250):
    print('*' * 30)
    print("Training initiating....")
    print(args)

    # Data prep
    obs_seq_len = args.obs_seq_len
    pred_seq_len = args.pred_seq_len

    if args.dataset == '2D':
        feature_dim = 2
        out_dim = 5
        scaling_factor = 10
        labels = ["Biker","Pedestrian","Car","Bus","Skater","Cart"]
    elif args.dataset == '3D':
        feature_dim = 3
        out_dim = 3
        scaling_factor = 1000
        labels = ['LeftHip','LeftKnee','LeftFeet','LeftToe','RightHip','RightKnee','RightFeet','RightToe','Spine1','Spine2','Neck1','Neck2', 'Head','LeftClavicle','LeftHumerus','LeftRadius','LeftWrist','LeftHand','LeftFinger','RightClavicle','RightHumerus','RightRadius','RightWrist','RightHand','RightFinger']
    with open(os.path.join(data_set, 'classInfo.json')) as f:
        class_info = json.load(f)
        class_weights = class_info["class_weights"]
    dset_train = TrajectoryDataset(
        os.path.join(data_set, 'train'),
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1, norm_lap_matr=True, label=labels, dim=feature_dim, sf=scaling_factor)
    print(dset_train)
    loader_train = DataLoader(
        dset_train,
        batch_size=1,
        shuffle=True,
        num_workers=0)

    dset_val = TrajectoryDataset(
        os.path.join(data_set, 'val'),
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1, norm_lap_matr=True, label=labels, dim=feature_dim, sf=scaling_factor)

    loader_val = DataLoader(
        dset_val,
        batch_size=1,
        shuffle=True,
        num_workers=0)


    # Defining the model
    model = label_gcnn(n_layer=args.n_layer, input_feat=feature_dim, output_feat=out_dim, seq_len=args.obs_seq_len, pred_seq_len=args.pred_seq_len,   
                          kernel_size=args.kernel_size, hot_enc_length=len(labels)).cuda()

    # Training settings
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training
    metrics = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        train(model, optimizer, loader_train, metrics, class_weights, labels)
        valid(model, loader_val, metrics, class_weights, labels)

        print('*' * 30)
        print('Epoch:', epoch)
        for k, v in metrics.items():
            if len(v) > 0:
                print(k, v[-1])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Model specific parameters
    parser.add_argument('--n_layer', type=int, default=1, help='number of Label-GCN layers')
    parser.add_argument('--kernel_size', type=int, default=3, help='graph convolving kernel size')

    # Data specific paremeters
    parser.add_argument('--dataset', type=str, default='3D', help='2D traffic prediction or 3D skeleton prediciton')
    parser.add_argument('--obs_seq_len', type=int, default=8, help='length of the observed trajectory')
    parser.add_argument('--pred_seq_len', type=int, default=12, help='length of the trajectory to be predicted')

    # Training specific parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='minibatch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')

    args = parser.parse_args()
    
    if args.dataset == '2D':
        path = os.path.join('data', 'stanfordProcessed')
    elif args.dataset == '3D':
        path = os.path.join('data', 'cmuProcessed')

    start_training(path, num_epochs=10)
