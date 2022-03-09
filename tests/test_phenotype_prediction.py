import pytest

from model import SpiceMixPlus
import torch
from torch import nn
from util import clustering_louvain_nclust, evaluate_embedding, evaluate_prediction_wrapper
from sklearn.cluster import AgglomerativeClustering
from pathlib import Path
import numpy as np
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from tqdm.auto import tqdm, trange
from umap import UMAP

@pytest.fixture(scope="module")
def phenotype_prediction():
    # evaluate = lambda model, f=evaluate_embedding: f(model, embedding='X', do_plot=True, do_sil=True)
    
    path2dataset = Path('../tests/test_data/obsolete_synthetic_500_100_20_15_0_0_i4')
    repli_list = ['0', '1']
    
    context = dict(device='cuda:0', dtype=torch.float32)

    K = 10
    
    model = SpiceMixPlus(
        K=K, lambda_Sigma_x_inv=1e-2,
        repli_list=repli_list,
        context=context,
    #     context_Y=dict(dtype=torch.float32, device='cpu'),
        context_Y=context,
    )
    model.load_dataset(path2dataset)
    model.meta['cell type'] = model.meta['cell type'].astype('category')
    
    phenotype2predictor = {}
    df = model.meta

    # cell type
    label_raw = df['cell type']
    label_a2i = dict(zip(np.unique(label_raw), itertools.count()))
    label = list(map(label_a2i.get, label_raw))
    df['cell type encoded'] = np.array(label, dtype=int)
    
    predictor = nn.Linear(K, len(set(label)))
    predictor = predictor.to(context['device'])

    # Initializing weights
    predictor.weight.data.normal_()
    predictor.bias.data.normal_()

    # Changed from Adam to AdamW...? Empirically, seems to make performance worse.
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-2, weight_decay=1e-2)
    
    # compute_loss = nn.CrossEntropyLoss()
    def compute_loss(yhat, y, mode="train"):
        if mode == 'train':
             return torch.nn.functional.cross_entropy(yhat, y, reduction='mean')
        elif mode == 'eval':
             return torch.nn.functional.cross_entropy(yhat, y, reduction='sum') * 10
        else:
            raise NotImplementError

    phenotype2predictor['cell type encoded'] = (predictor, optimizer, compute_loss)
    # del label_raw, label, predictor, optimizer, compute_loss
    model.register_phenotype_predictors(phenotype2predictor)
    print(model.Ys)
    model.phenotypes[1]['cell type encoded'] = None
    
    model.initialize(
        method='svd',
    )
    # evaluate(model)
    for iiter in trange(20):
        model.estimate_parameters(iiter=iiter, use_spatial=[False]*model.num_repli)
        model.estimate_weights(iiter=iiter, use_spatial=[False]*model.num_repli)
    # evaluate(model)
    model.initialize_Sigma_x_inv()
    
    for iiter in range(1, 11):
        model.estimate_parameters(iiter=iiter, use_spatial=[True]*model.num_repli)
        model.estimate_weights(iiter=iiter, use_spatial=[True]*model.num_repli)
 
    model.save_results(path2dataset, iiter)
 
    return model

def test_silhouette(phenotype_prediction):
    Xs = [X.cpu().numpy() for X in phenotype_prediction.Xs]
    x = np.concatenate(Xs, axis=0)
    x = StandardScaler().fit_transform(x)
    x = UMAP(
        random_state=phenotype_prediction.random_state,
        n_neighbors=10,
    ).fit_transform(x)
 
    silhouette = silhouette_score(x, phenotype_prediction.meta['cell type'], random_state=phenotype_prediction.random_state)
    print(silhouette)
    assert 0.7584578394889832 == pytest.approx(silhouette)
