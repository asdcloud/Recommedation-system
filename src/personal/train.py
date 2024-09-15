import os
import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from model import NeuMF
from data import PreprocessData
from main import config  # delete later


class Train:
    def __init__(self, config, model):
        self.config = config
        self.criterion = torch.nn.BCELoss

        if config['optimizer'] == 'adam':
            self.optimizer = torch.optim.adam(model.parameters(), lr=config['adam_lr'], weight_decay=0)
        else:
            pass  # SGD -> implement later
      
    def train_single_batch(self, users, items, ratings):
        if config['use_cuda'] == True:
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
        self.optimizer.zero_grad()
        ratings_pred = self.model(config)
        loss = self.criterion(ratings_pred.view(-1), ratings)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_an_epoch(self, train_loader, epoch_id, writer):
        self.model.train()
        total_loss = 0
        for batch_id, batch in enumerate(train_loader):
            user, item, rating = batch[0], batch[1], batch[2]
            rating = rating.float()  # necessary ?
            loss = self.train_single_batch(user, item, rating)
            print('[Training Epoch {}] Batch {}, Loss {}'.format(epoch_id, batch_id, loss))
            total_loss += loss
        writer.add_scalar('model/loss', total_loss, epoch_id)

    def evaluate(self, evaluate_data, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        with torch.no_grad():
            test_users, test_items = evaluate_data[0], evaluate_data[1]
            negative_users, negative_items = evaluate_data[2], evaluate_data[3]
            if self.config['use_cuda'] is True:
                test_users = test_users.cuda()
                test_items = test_items.cuda()
                negative_users = negative_users.cuda()
                negative_items = negative_items.cuda()

        if self.config['use_bachify_eval'] == False:    
            test_scores = self.model(test_users, test_items)
            negative_scores = self.model(negative_users, negative_items)
        else:
            test_scores = []
            negative_scores = []
            bs = self.config['batch_size']
            for start_idx in range(0, len(test_users), bs):
                end_idx = min(start_idx + bs, len(test_users))
                batch_test_users = test_users[start_idx:end_idx]
                batch_test_items = test_items[start_idx:end_idx]
                test_scores.append(self.model(batch_test_users, batch_test_items))
            for start_idx in tqdm(range(0, len(negative_users), bs)):
                end_idx = min(start_idx + bs, len(negative_users))
                batch_negative_users = negative_users[start_idx:end_idx]
                batch_negative_items = negative_items[start_idx:end_idx]
                negative_scores.append(self.model(batch_negative_users, batch_negative_items))
            test_scores = torch.concatenate(test_scores, dim=0)
            negative_scores = torch.concatenate(negative_scores, dim=0)


            if self.config['use_cuda'] is True:
                test_users = test_users.cpu()
                test_items = test_items.cpu()
                test_scores = test_scores.cpu()
                negative_users = negative_users.cpu()
                negative_items = negative_items.cpu()
                negative_scores = negative_scores.cpu()
            self._metron.subjects = [test_users.data.view(-1).tolist(),
                                 test_items.data.view(-1).tolist(),
                                 test_scores.data.view(-1).tolist(),
                                 negative_users.data.view(-1).tolist(),
                                 negative_items.data.view(-1).tolist(),
                                 negative_scores.data.view(-1).tolist()]
        hit_ratio, ndcg = self._metron.cal_hit_ratio(), self._metron.cal_ndcg()
        self._writer.add_scalar('performance/HR', hit_ratio, epoch_id)
        self._writer.add_scalar('performance/NDCG', ndcg, epoch_id)
        print('[Evluating Epoch {}] HR = {:.4f}, NDCG = {:.4f}'.format(epoch_id, hit_ratio, ndcg))
        return hit_ratio, ndcg

    def save(self, alias, epoch_id, hit_ratio, ndcg):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = self.config['model_dir'].format(alias, epoch_id, hit_ratio, ndcg)
        save_checkpoint(self.model, model_dir)



