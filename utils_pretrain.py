import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from model import DenoisingAutoencoder,AEPretrain
from Dataset import DataGenerator_pretrain,DataGenerator
import utils
import random
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def pretrain_autoencoder(
    train_x,
    train_semi_y,
    val_x,
    val_y,
    output_path,
    dataset_name,
    layers_dims,
    logger,
    device,
    seed,
    noise_factor=0.05,
    epochs=200,
    batch_size=128,
    nbatch_pemr_epoch=64,
    learning_rate=0.005,
    n_prototypes=0,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    input_dim = train_x.shape[1]
    current_train_x = torch.tensor(train_x, dtype=torch.float32)
    current_val_x = torch.tensor(val_x[val_y == 0], dtype=torch.float32)
    encoder_layers = []
    decoder_layers = []
     
    for hidden_dim in layers_dims:
        logger.info(f"Training layer with {input_dim} -> {hidden_dim}")
         
        dae = DenoisingAutoencoder(input_dim, hidden_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(dae.parameters(), lr=learning_rate, weight_decay=5e-4)

        train_x_noise = current_train_x + noise_factor * torch.randn_like(
            current_train_x
        )
        data = DataGenerator_pretrain(
            current_train_x,
            train_x_noise,
            train_semi_y,
            batch_size=batch_size,
            device=device,
        )
        early_stp = utils.EarlyStopping(patience=15, model_name=f"{output_path}/checkpoints/{dataset_name}_DAE", verbose=False)
        
        for epoch in range(epochs):
            dae.train()
            running_loss = 0.0
            batch_triplets_inputs,batch_triplets_targets = data.load_batches(n_batches=nbatch_pemr_epoch)
            batch_triplets_inputs = torch.from_numpy(batch_triplets_inputs).float().to(device)
            batch_triplets_targets = torch.from_numpy(batch_triplets_targets).float().to(device)
            for inputs, targets in zip(batch_triplets_inputs,batch_triplets_targets):
                pos, neg = inputs[:batch_size], inputs[batch_size:]
                pos_target, neg_target = targets[:batch_size], targets[batch_size:]
                outputs = dae(neg)
                loss = criterion(outputs, neg_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
             
            dae.eval()
            with torch.no_grad():
                val_outputs = dae(current_val_x.to(device))
                val_loss = criterion(
                    val_outputs, current_val_x.to(device)
                ).item()
                early_stp(val_loss, model=dae)
                if early_stp.early_stop:
                    dae.load_state_dict(torch.load(early_stp.path))
                    logger.info(f"early stop at {epoch} epoch!")
                    break
        
        dae.eval()
        current_train_x = dae.encode(current_train_x.to(device)).cpu().detach()
        current_val_x = dae.encode(current_val_x.to(device)).cpu().detach()
        input_dim = hidden_dim
        encoder_layers.append(dae.encoder)
        decoder_layers.insert(0, dae.decoder)  
    
     
    encoder,decoder= nn.Sequential(*encoder_layers), nn.Sequential(*decoder_layers)
    autoencoder =AEPretrain(encoder, decoder).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate, weight_decay=5e-4)
    data = DataGenerator(
            train_x,
            train_semi_y,
            batch_size=batch_size,
            device=device,
        )
    val_x=torch.tensor(val_x[val_y==0],dtype=torch.float32)
    early_stp = utils.EarlyStopping(patience=15, model_name=f"{output_path}/checkpoints/{dataset_name}_AE", verbose=False)
    for epoch in range(epochs):
        autoencoder.train()
        running_loss = 0.0
        batch_triplets = data.load_batches(n_batches=nbatch_pemr_epoch)
        batch_triplets = torch.from_numpy(batch_triplets).float().to(device)
        for inputs in batch_triplets:
            pos, neg = inputs[:batch_size], inputs[batch_size:]
            _,outputs = autoencoder(neg)
            loss = criterion(outputs, neg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
         
        autoencoder.eval()
        with torch.no_grad():
            _,val_outputs = autoencoder(val_x.to(device))
            val_loss = criterion(
                val_outputs, val_x.to(device)
            ).item()
            early_stp(val_loss, model=autoencoder)
            if early_stp.early_stop:
                autoencoder.load_state_dict(torch.load(early_stp.path))
                logger.info("early stop!")
                break
    
    cluster_centers=initialize_cluster_centers(autoencoder,val_x,device,seed,logger,n_prototypes)
    return autoencoder,cluster_centers

  
def initialize_cluster_centers(autoencoder,data,device,seed,logger,n_prototypes):
    with torch.no_grad():
        embedded_data= autoencoder(data.to(device))[0].cpu().numpy()
    silhouette_avg_list=[]
    centers_list=[]
    if n_prototypes==0:
        for n_clusters in range(2, 11):
            kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
            kmeans.fit(embedded_data)
            labels = kmeans.labels_
            silhouette_avg = silhouette_score(embedded_data, labels)
            silhouette_avg_list.append(silhouette_avg)
            centers_list.append(kmeans.cluster_centers_)
        n_clusters = np.argmax(silhouette_avg_list)+2
        centers=centers_list[n_clusters-2]
        logger.info(f"optimal number of clusters:{n_clusters}")
    else:
        kmeans = KMeans(n_clusters=n_prototypes, random_state=seed)
        kmeans.fit(embedded_data)
        centers=kmeans.cluster_centers_
    return torch.tensor(centers,dtype=torch.float32).to(device)

