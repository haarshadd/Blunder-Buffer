"""Inference adapter for the existing ECO-aware Wide & Deep meta learner.

This model is only valid after an ECO-conditioned expert prediction exists.
It is not used for T0.
"""
from __future__ import annotations
import os, pickle
import numpy as np
import torch
import torch.nn as nn

ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
MODEL_PATH=os.path.join(ROOT,'saved_models','meta_learner_wide_deep.pkl')
LABELS=['black','draw','white']

class WideAndDeepContext(nn.Module):
    def __init__(self,deep_in,wide_in):
        super().__init__(); self.deep_path=nn.Sequential(nn.Linear(deep_in,24),nn.ReLU(),nn.Dropout(.3)); self.combine=nn.Sequential(nn.Linear(24+wide_in,16),nn.ReLU(),nn.Dropout(.2),nn.Linear(16,3))
    def forward(self,w,d): return self.combine(torch.cat((self.deep_path(d),w),1))

def predict_opening_meta(wide_values, deep_probs, path=MODEL_PATH):
    art=pickle.load(open(path,'rb'))
    deep_cols=art['deep_cols']; wide_cols=art['wide_cols']
    if len(deep_probs)!=len(deep_cols):
        raise ValueError(f'Opening meta expects {len(deep_cols)} expert probabilities, got {len(deep_probs)}')
    if len(wide_values)!=len(wide_cols):
        raise ValueError(f'Opening meta expects {len(wide_cols)} wide features, got {len(wide_values)}')
    m=WideAndDeepContext(len(deep_cols),len(wide_cols)); m.load_state_dict(art['model_state_dict']); m.eval()
    w=art['wide_scaler'].transform([wide_values])
    with torch.no_grad():
        o=m(torch.tensor(w,dtype=torch.float32),torch.tensor([deep_probs],dtype=torch.float32)); p=torch.softmax(o,1).numpy()[0]
    return dict(zip(LABELS,map(float,p)))
