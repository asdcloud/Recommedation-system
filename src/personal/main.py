import pandas as pd
import torch
import torch.nn as nn
from data import PreprocessData
from model import NeuMFEngine

data_path = "./res/ml-1m/ratings.csv"
config = {
    'alias': 'neumf_factor8neg4',
    'num_epoch': 200,
    'batch_size': 1024,
    'optimizer': 'adam',
    'adam_lr': 1e-3,
    'num_users': 6040,
    'num_items': 3706,
    'latent_dim_mf': 8,
    'latent_dim_mlp': 8,
    'num_negative': 4,
    'layers': [64, 32, 16, 8],  # layers[0] is the concat of latent user vector & latent item vector
    'l2_regularization': 0.0000001,
    'weight_init_gaussian': True,
    'use_cuda': True,
    'use_bachify_eval': True,
    'device_id': 0,
    'pretrain': True,
    'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
    'pretrain_mlp': 'checkpoints/{}'.format('mlp_factor8neg4_Epoch100_HR0.5606_NDCG0.2463.model'),
    'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
}

if __name__ == '__main__':
    data_path = "./res/ml-1m/ratings.csv"
    ml_1m = pd.read_csv(data_path)
    # print(data.head())

    sample_generator = PreprocessData(ratings= ml_1m, datatype="implicit")
    print('success ****')
    evaluate_data = sample_generator.evaluate_data

    engine = NeuMFEngine(config)
    for epoch in range(config['num_epoch']):
        print('Epoch {} starts !'.format(epoch))
        print('-' * 80)
        train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
        engine.train_an_epoch(train_loader, epoch_id=epoch)
        hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
        engine.save(config['alias'], epoch, hit_ratio, ndcg)


