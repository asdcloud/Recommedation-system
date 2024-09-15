import torch
import torch.nn as nn
from data import PreprocessData
from model import NeuMF

data_path = "./res/ml-1m/ratings.csv"
config = {
    'num_epoch': 200,
    'batch_size': 1024,
    'num_users': 6040,  # Replace with actual user count
    'num_items': 3706,  # Replace with actual item count
    'latent_dim_mf': 8,
    'latent_dim_mlp': 16,  # 這個跟上面那個屬於可以調整的參數
    'layers': [16, 64, 32, 16, 8],  # Fully connected layers for MLP
    'num_negative': 4,
    'adam_lr': 0.001,
    'data_type': 'implicit',
    'optimizer': 'adam',
    'weight_init_gaussian': True,
    'use_cuda': True,  # Set to True if using GPU
    'device_id': 0,
    'pretrain': False,
    'model_dir_template': '',
}

# neumf_config = {'alias': 'neumf_factor8neg4',
#                 'num_epoch': 200,
#                 'batch_size': 1024,
#                 'optimizer': 'adam',
#                 'adam_lr': 1e-3,
#                 'num_users': 6040,
#                 'num_items': 3706,
#                 'latent_dim_mf': 8,
#                 'latent_dim_mlp': 8,
#                 'num_negative': 4,
#                 'layers': [16, 64, 32, 16, 8],  # layers[0] is the concat of latent user vector & latent item vector
#                 'l2_regularization': 0.0000001,
#                 'weight_init_gaussian': True,
#                 'use_cuda': True,
#                 'use_bachify_eval': True,
#                 'device_id': 0,
#                 'pretrain': False,
#                 'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
#                 'pretrain_mlp': 'checkpoints/{}'.format('mlp_factor8neg4_Epoch100_HR0.5606_NDCG0.2463.model'),
#                 'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
#                 }




# Create the model
model = NeuMF(config)
train_loader = PreprocessData.instance_a_train_loader(num_negatives=config['num_negative'], batch_size=config['batch_size'])

# engine -> train an epoch

# Initialize optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.BCELoss()

# Assume that preprocess_data is an instance of the PreprocessData class
preprocess_data = PreprocessData(ratings=data_path, datatype=config['data_type'])



# Train the model
train_model(model, train_loader, optimizer, loss_function, num_epochs=10)

