import torch
import torch.nn as nn
from engine import Engine

class NeuMF(torch.nn.Module):
    def __init__(self, config):
        super(NeuMF, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim_mf = config['latent_dim_mf']
        self.latent_dim_mlp = config['latent_dim_mlp']

        # Embeddings for MLP and MF parts
        self.embedding_user_mlp = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mlp)
        self.embedding_item_mlp = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mlp)
        self.embedding_user_mf = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mf)
        self.embedding_item_mf = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mf)

        # Fully connected layers for MLP
        self.fc_layers = nn.ModuleList()
        for in_size, out_size in zip(config['layers'][:-1], config['layers'][1:]):
            self.fc_layers.append(nn.Linear(in_size, out_size))

        # Output layer
        self.affine_output = nn.Linear(in_features=config['layers'][-1] + config['latent_dim_mf'], out_features=1)
        self.logistic = nn.Sigmoid()

        # Initialize model parameters with Gaussian distribution
        if config.get('weight_init_gaussian', False):
            for module in self.modules():
                if isinstance(module, (nn.Embedding, nn.Linear)):
                    torch.nn.init.normal_(module.weight.data, 0.0, 0.01)

    def forward(self, user_indices, item_indices):
        # MLP part
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        
        # MF part
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)
        
        # Concatenate user and item embeddings for MLP part
        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
        
        # Element-wise product of user and item embeddings for MF part
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)
        
        # Pass through MLP layers
        for layer in self.fc_layers:
            mlp_vector = layer(mlp_vector)
            mlp_vector = nn.ReLU()(mlp_vector)
        
        # Concatenate MF and MLP vectors
        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        
        # Final output layer
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

class NeuMFEngine(Engine):
    def __init__(self, config):
        self.model = NeuMF(config)
        if torch.cuda.is_available():
            print('using GPU...')
            self.model.cuda()
        super(NeuMFEngine, self).__init__(config)
        print(self.model)
        