import os
import logging
import sys
import argparse
import shutil
import time
import _pickle as pkl
import numpy as np
import argparse
import scipy
import sklearn
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data_utils
from torchvision import datasets, models, transforms
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from mymodels import WideResNet, DenseNet161, SqueezeNet
from utils import *
from fastknn import KNNClassifier, KNNRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR


def train(model, train_loader, test_loader=None, epochs=100, init_epoch=0,
          model_name='model', early_stopping=None):
    "Trains a PyTorch model."
    best_acc = 0.
    stop_counter = 0
    for epoch in range(init_epoch, epochs): 
        # Train
        model.train()
        epoch_loss, epoch_acc = [], []
        update_learning_rate(optimizer, args.lr, epoch, args.max_epochs, args.lr_schedule)
        for i, (x, y) in enumerate(train_loader):
            if args.use_cuda:
                x, y = x.to(device), y.to(device)
            x, y = Variable(x), Variable(y)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = loss_func(logits, y)
            loss.backward()
            optimizer.step()
            predictions = torch.argmax(logits, -1)
            epoch_loss.append(loss.detach().cpu().numpy())
            epoch_acc.append(np.mean(predictions.detach().cpu().numpy() == y.data.cpu().numpy()))
        logging.info('Epoch {}, Train-Loss={}, Train-Accuracy={}'.format(
            epoch, np.mean(epoch_loss), np.mean(epoch_acc)))
        if test_loader is None:
            if epoch % args.save_every == 0 or epoch == epochs - 1:
                save_model(model, optimizer, args.experiment_dir, epoch, model_name=model_name)
            continue
        # Evaluate
        model.eval()
        epoch_test_loss, epoch_test_acc = [], []
        num_test_points = 0
        for i, (x, y) in enumerate(test_loader):
            if args.use_cuda:
                x, y = x.to(device), y.to(device)
            x, y = Variable(x), Variable(y)
            logits, _ = model(x)
            num_test_points += len(logits)
            test_loss = loss_func(logits, y)
            test_predictions = torch.argmax(logits, -1)
            epoch_test_loss.append(test_loss.detach().cpu().numpy() * len(logits))
            epoch_test_acc.append(np.mean(test_predictions.detach().cpu().numpy() ==\
                                          y.data.cpu().numpy()) * len(logits))
        if np.sum(epoch_test_acc) / num_test_points >= best_acc:
            stop_counter = 0
            best_acc = np.sum(epoch_test_acc) / num_test_points
            save_model(model, optimizer, args.experiment_dir, epoch, model_name=model_name)
        else:
            stop_counter += 1
        print(stop_counter, num_test_points)
        logging.info('Epoch {}, Test-Loss={}, Test-Accuracy={}, Best-Accuracy={}'.format(
            epoch, np.sum(epoch_test_loss) / num_test_points, np.sum(epoch_test_acc) / num_test_points,
                    best_acc))
        if early_stopping is not None and stop_counter > early_stopping:
            break
    # Only keep last three checkpoints        
    checkpoints = list_checkpoints(args.experiment_dir, model_name)
    saved_epochs = [checkpoint_epoch(checkpoint) for checkpoint in checkpoints]
    [os.remove(checkpoint) for checkpoint in checkpoints
     if checkpoint_epoch(checkpoint) in np.sort(saved_epochs)[:-3]]
    [os.remove(checkpoint.replace('checkpoint', 'opt-checkpoint')) for checkpoint in checkpoints
     if checkpoint_epoch(checkpoint) in np.sort(saved_epochs)[:-3]]
    open(args.experiment_dir + '/complete.txt', 'w').write('True!')
        
def test(model, test_loader):
    "Evaluates a PyTorch model."
    model.eval()
    acc = []
    for i, (x, y) in enumerate(test_loader):
        if args.use_cuda:
            x, y = x.to(device), y.to(device)
        x, y = Variable(x), Variable(y)
        logits, _ = model(x)
        predictions = torch.argmax(logits, -1)
        acc.append(np.mean(predictions.detach().cpu().numpy() == y.data.cpu().numpy()))
    return(np.mean(acc))

def extract_embeddings(model, batch_loader):
    "Extracts Pre-logit embeddings."
    features, labels = [], []
    for x, y in tqdm(batch_loader):
        if args.use_cuda:
            x, y = x.to(device), y.to(device)
        x, y = Variable(x), Variable(y)
        _, f = model(x)
        features.append(f.detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())
    return np.concatenate(features), np.concatenate(labels)

def extract_predictions(model, batch_loader):
    "Extracts a PyTorch model's predictions."
    probs, labels = [], []
    for x, y in tqdm(batch_loader):
        if args.use_cuda:
            x, y = x.to(device), y.to(device)
        x, y = Variable(x), Variable(y)
        p = nn.functional.softmax(model(x)[0])
        probs.append(p.detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())
    return np.concatenate(probs), np.concatenate(labels)


parser = argparse.ArgumentParser('My params!')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--overwrite', default=False, action='store_true', help='overwrite')
parser.add_argument('--experiment_dir', default='experimentsGone', type=str, help='experiment directory')
parser.add_argument('--seed', default=666, type=int, help='random seed')
parser.add_argument('--dataset', default='celeba_fairness', type=str, help='dataset')
parser.add_argument('--batch_size', default=64, type=int, help='training batch size')
parser.add_argument('--test_batch_size', default=64, type=int, help='test batch size')
parser.add_argument('--budget', default=5000, type=int, help='labeling budget')
parser.add_argument('--initial_size', default=5000, type=int, help='initial labeled size')
parser.add_argument('--max_epochs', default=100, type=int, help='training batch size')
parser.add_argument('--lr_schedule', default='cosine', type=str, help='learning rate schedule')
parser.add_argument('--save_every', default=5, type=int, help='frequency of saving model checkpoints')
parser.add_argument('--method', default='rnd', type=str, help='active learning method')
parser.add_argument('--step', default=0, type=int, help='labelling step')
args = parser.parse_args()

step = args.step #Active learning step
method = args.method #Active learning methods.
args.experiment_dir = os.path.join(
    args.experiment_dir,
    args.dataset,
    'initsize{}_budget{}'.format(args.initial_size, args.budget),
    '{}_{}'.format(method, step))
parameters = {'method': method, 'step': step, 'budget': args.budget, 'initial_size': args.initial_size}

if os.path.exists(args.experiment_dir):
    if args.overwrite:
        shutil.rmtree(args.experiment_dir)
        os.makedirs(args.experiment_dir)
else:
    os.makedirs(args.experiment_dir)
# Create logger    
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.experiment_dir, 'log.txt')),
        logging.StreamHandler()
    ])
logger = logging.getLogger()

args.use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if args.use_cuda else 'cpu')
if os.path.exists(os.path.join(args.experiment_dir, 'new_labeled_idx.npy')):
    raise ValueError('Already done!')
if step:
    prev_step_dir = '_'.join(args.experiment_dir.split('_')[:-1] + ['{}'.format(step - 1)])
    print(prev_step_dir)
    while not os.path.exists(os.path.join(prev_step_dir, 'new_labeled_idx.npy')):
        print('waiting for previous step to finish...')
        time.sleep(5)
        continue
    labeled_idx = np.load(os.path.join(prev_step_dir, 'new_labeled_idx.npy'))
else:
    labeled_idx = np.arange(args.initial_size)
np.save(os.path.join(args.experiment_dir, 'labeled_idx.npy'), labeled_idx)
if args.step == 0 and args.method != 'rnd':
    checkpoint_dir = args.experiment_dir.replace(args.method, 'rnd')
    while not os.path.exists(checkpoint_dir + '/complete.txt'):
        print('Waiting for rnd method to finish!')
        time.sleep(60)
# Set random seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.use_cuda:
    torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
args.dataset = args.dataset.lower()
# Preprocessing steps specific to each dataset
if 'cifar10' in args.dataset or 'cinic10' in args.dataset:
    args.num_classes = 10
    train_t_list = [
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]
    test_t_list = [transforms.ToTensor()]
    model = WideResNet(depth=28, num_classes=args.num_classes, widen_factor=10)
elif 'lmnist' in args.dataset:
    args.num_classes = 26
    train_t_list = [transforms.ToTensor()]
    test_t_list = [transforms.ToTensor()]
    model = WideResNet(depth=16, num_classes=args.num_classes, widen_factor=8)
elif 'emnist' in args.dataset:
    args.num_classes = 47
    train_t_list = [transforms.ToTensor()]
    test_t_list = [transforms.ToTensor()]
    model = WideResNet(depth=16, num_classes=args.num_classes, widen_factor=8)
elif 'pcam' in args.dataset:
    args.num_classes = 2
    train_t_list = [
        transforms.ToPILImage(),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(.4,.4,.4),
        transforms.ToTensor()
    ]
    test_t_list = [transforms.ToTensor()]
    model = WideResNet(depth=28, num_classes=args.num_classes, widen_factor=10)
elif 'celeba' in args.dataset or 'ppb' in args.dataset:
    args.num_classes = 2
    train_t_list = [
        transforms.ToPILImage(),
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]
    test_t_list = [transforms.ToTensor()]
    model = WideResNet(depth=28, num_classes=args.num_classes, widen_factor=10)
    #model = WideResNet(depth=28, num_classes=args.num_classes, widen_factor=10)
elif 'tinyimagenet' in args.dataset:
    args.num_classes = 200
    train_t_list = [
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=64, scale=(0.5, 1.0)),
        transforms.ColorJitter(.4,.4,.4),
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]
    test_t_list = [transforms.ToTensor()]
    model = WideResNet(depth=28, num_classes=args.num_classes, widen_factor=10)
elif 'pannuke' in args.dataset:
    args.num_classes = 5
    train_t_list = [
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]
    test_t_list = [transforms.ToTensor()]
    model = WideResNet(depth=28, num_classes=args.num_classes, widen_factor=10)
elif 'svhn' in args.dataset or 'mnist' in args.dataset:
    args.num_classes = 10
    train_t_list = [transforms.ToTensor()]
    test_t_list = [transforms.ToTensor()]
    model = WideResNet(depth=16, num_classes=args.num_classes, widen_factor=8)
else:
    raise ValueError('Invalid dataset argument!')


# Training data
X_all, y_all, X_test_all, y_test_all = load_dataset(args.dataset)
train_transforms = transforms.Compose(train_t_list)
test_transforms = transforms.Compose(test_t_list)
train_dataset = ArrayDataset(X_all[labeled_idx], y_all[labeled_idx],
                          transform=train_transforms)
batch_dataset = ArrayDataset(X_all[labeled_idx], y_all[labeled_idx],
                          transform=test_transforms)
unlabeled_dataset = ArrayDataset(np.delete(X_all, labeled_idx, 0),
                          np.delete(y_all, labeled_idx, 0),
                          transform=test_transforms)
X_test, X_val, y_test, y_val = train_test_split(X_test_all, y_test_all, test_size=600, stratify=y_test_all)
test_dataset = ArrayDataset(X_test, y_test,
                         transform=test_transforms)
val_dataset = ArrayDataset(X_val, y_val,
                         transform=test_transforms)
train_loader = data_utils.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True)
batch_loader = data_utils.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
test_loader = data_utils.DataLoader(
    test_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False)
val_loader = data_utils.DataLoader(
    val_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False)
unlabeled_loader = data_utils.DataLoader(
    unlabeled_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False)

## Model training
model_name = 'step{}'.format(step)
if args.use_cuda:
    model = nn.DataParallel(model).cuda()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
loss_func = nn.CrossEntropyLoss()
# Load model
if args.step == 0 and args.method != 'rnd':
    checkpoint_dir = args.experiment_dir.replace(args.method, 'rnd')
else:
    checkpoint_dir = args.experiment_dir
checkpoints = list_checkpoints(checkpoint_dir, model_name)
if not os.path.exists(checkpoint_dir + '/complete.txt'):
    if args.overwrite or len(checkpoints) == 0:
        init_epoch = 0
    else:
        init_epoch = np.max([checkpoint_epoch(checkpoint) for checkpoint in checkpoints]) + 1
        checkpoint = torch.load(os.path.join(
            checkpoint_dir, '{}-checkpoint-epoch{}.pt'.format(model_name, init_epoch - 1)))
        op_checkpoint = torch.load(os.path.join(
            checkpoint_dir, '{}-opt-checkpoint-epoch{}.pt'.format(model_name, init_epoch - 1)))
        model.load_state_dict(checkpoint)
        optimizer.load_state_dict(op_checkpoint)
    train(model, train_loader, val_loader, args.max_epochs,
          init_epoch=init_epoch, model_name='step{}'.format(step))
    
test_acc_file = os.path.join(args.experiment_dir, 'test-acc.txt')
checkpoints = list_checkpoints(checkpoint_dir, model_name)
final_epoch = np.max([checkpoint_epoch(checkpoint) for checkpoint in checkpoints]) + 1
checkpoint = torch.load(os.path.join(
    checkpoint_dir, '{}-checkpoint-epoch{}.pt'.format(model_name, final_epoch - 1)))
model.load_state_dict(checkpoint)
op_checkpoint = torch.load(os.path.join(
    checkpoint_dir, '{}-opt-checkpoint-epoch{}.pt'.format(model_name, final_epoch - 1)))
optimizer.load_state_dict(op_checkpoint)
if os.path.exists(test_acc_file):
    test_acc = float(open(test_acc_file).read())
else:
    test_acc = test(model, test_loader)
    open(test_acc_file, 'w').write(str(test_acc))
print('Test accuracy = '.format(test_acc))


if method != 'rnd':
    print('Computing embeddings!')
    t_init = time.time()
    X_f, y_f = extract_embeddings(model, batch_loader)
    X_f_v, y_f_v = extract_embeddings(model, val_loader)
    X_f_u, y_f_u = extract_embeddings(model, unlabeled_loader)
    print('Embeddings computed! in {} seconds'.format(time.time() - t_init))
un_weights, _ = extract_predictions(model, unlabeled_loader)
t_init = time.time()   
if method == 'rnd':
    new_labeled_idx = np.random.choice(
        np.delete(np.arange(len(X_all)), labeled_idx), args.budget, replace=False)
elif method == 'minconf':
    predicted_proba = np.max(un_weights, -1)
    chosen = np.argsort(predicted_proba)[:args.budget]
    new_labeled_idx = np.delete(np.arange(len(X_all)), labeled_idx)[chosen]
elif method == 'entropy':
    un_weights_clipped = np.clip(un_weights, 1e-12, None)
    entropies = -np.sum(np.log(un_weights_clipped) * un_weights_clipped, -1)
    chosen = np.argsort(-entropies)[:args.budget]
    new_labeled_idx = np.delete(np.arange(len(X_all)), labeled_idx)[chosen]
elif method == 'margin':
    un_weights_first = np.sort(un_weights, -1)[:, -1]
    un_weights_second = np.sort(un_weights, -1)[:, -2]
    margins = un_weights_first - un_weights_second
    chosen = np.argsort(margins)[:args.budget]
    new_labeled_idx = np.delete(np.arange(len(X_all)), labeled_idx)[chosen]
elif 'ADS' in method:
    ratio = int(args.method.split('_')[-1])
    best_score = 0.0
    for k in range(3, 19, 2):
        knn = KNNClassifier(k)
        knn.fit(X_f, y_f)
        k_score = knn.score(X_f_v, y_f_v)
        if k_score > best_score:
            best_score, best_k = k_score, k
    individual_vals = knn_shapley(X_f, y_f, X_f_v, y_f_v, K=best_k)
    shap_vals = np.mean(individual_vals, 0)
    regressors, scores = fit_value_regressors(X_f, y_f, shap_vals, ['knn'])
    if args.num_classes <= 10:
        possible_vals = predict_value(X_f_u, regressors)
    else:        
        possible_vals = predict_value(X_f_u, regressors, top_classes=10, weights=un_weights)
    aggregated_vals = np.max(possible_vals, -1)
    shap_chosen = np.argsort(-aggregated_vals)[:ratio * args.budget]
    X_f_u_shap_chosen = X_f_u[shap_chosen]
    ub, greedy_centers = greedy_k_center(np.concatenate([X_f, X_f_u_shap_chosen]), len(X_f), args.budget)
    unlabeled_shap_centers = greedy_centers[len(X_f):] - len(X_f)
    chosen = shap_chosen[unlabeled_shap_centers]
    new_labeled_idx = np.delete(np.arange(len(X_all)), labeled_idx)[chosen]
elif method == 'coresetgreedy':
    ub, greedy_centers = greedy_k_center(np.concatenate([X_f, X_f_u]), len(X_f), args.budget)
    chosen = greedy_centers[len(X_f):] - len(X_f)
    new_labeled_idx = np.delete(np.arange(len(X_all)), labeled_idx)[chosen]
else:
    raise ValueError('Invalid method!')
print('AL time spent {}s'.format(time.time() - t_init))    
np.save(os.path.join(args.experiment_dir, 'new_labeled_idx.npy'),
        np.concatenate([labeled_idx, new_labeled_idx]))