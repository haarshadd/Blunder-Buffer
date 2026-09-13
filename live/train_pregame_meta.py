"""Train the T0 pregame meta model from existing Baseline + Momentum OOF data.

ECO is intentionally excluded from this model. The historical ECO-aware model
remains the T1 opening-conditioned model.
"""
from __future__ import annotations
import os, sqlite3, pickle
import numpy as np, pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, accuracy_score

ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
DB_PATH=os.path.join(ROOT,'data','omni_pundit.db')
OUT=os.path.join(ROOT,'saved_models','meta_learner_pregame_wide_deep.pkl')

class WideAndDeepContext(nn.Module):
    def __init__(self,deep_in,wide_in):
        super().__init__(); self.deep_path=nn.Sequential(nn.Linear(deep_in,24),nn.ReLU(),nn.Dropout(.3)); self.combine=nn.Sequential(nn.Linear(24+wide_in,16),nn.ReLU(),nn.Dropout(.2),nn.Linear(16,3))
    def forward(self,w,d): return self.combine(torch.cat((self.deep_path(d),w),1))

def context(df):
    tl=[]
    for r in df.itertuples():
        tl += [(r.game_id,r.date,r.white,'white',1 if r.target_result==1 else 0 if r.target_result==0 else -1),(r.game_id,r.date,r.black,'black',1 if r.target_result==-1 else 0 if r.target_result==0 else -1)]
    p=pd.DataFrame(tl,columns=['game_id','date','player','color','res']).sort_values(['player','date'])
    p['prev']=p.groupby('player').date.shift(1); p['rest']=(p.date-p.prev).dt.days.fillna(30).clip(0,90)
    x=p.set_index('date'); p['fatigue']=x.groupby('player').game_id.rolling('7D').count().values-1; p=p.reset_index(drop=True)
    p['loss']=(p.res==-1).astype(int); p['tilt']=p.groupby('player').loss.transform(lambda x:x*(x.groupby((x!=x.shift()).cumsum()).cumcount()+1)); p['tilt']=p.groupby('player').tilt.shift(1).fillna(0)
    p['draw']=(p.res==0).astype(int); p['black_draw']=np.where(p.color=='black',p.draw,np.nan); p['bdr']=p.groupby('player').black_draw.transform(lambda x:x.shift(1).rolling(10,min_periods=1).mean()).fillna(0)
    w=p[p.color=='white'][['game_id','rest','fatigue','tilt']].rename(columns={'rest':'wrest','fatigue':'wfatigue','tilt':'wtilt'}); b=p[p.color=='black'][['game_id','rest','fatigue','tilt','bdr']].rename(columns={'rest':'brest','fatigue':'bfatigue','tilt':'btilt','bdr':'black_draw_rate_10'})
    z=df.merge(w,on='game_id').merge(b,on='game_id'); z['rest_diff']=z.wrest-z.brest; z['fatigue_diff']=z.wfatigue-z.bfatigue; z['tilt_diff']=z.wtilt-z.btilt; return z

def main():
    conn=sqlite3.connect(DB_PATH); q='''SELECT b.game_id,b.date,b.white,b.black,b.target_result,b.elo_diff,base.base_oof_prob_black,base.base_oof_prob_draw,base.base_oof_prob_white,mom.momentum_oof_prob_black,mom.momentum_oof_prob_draw,mom.momentum_oof_prob_white FROM baseline_features b JOIN oof_predictions_baseline base ON b.game_id=base.game_id JOIN oof_predictions_momentum mom ON b.game_id=mom.game_id ORDER BY b.date'''; df=pd.read_sql(q,conn); conn.close(); df['date']=pd.to_datetime(df.date); df=context(df)
    wide=['elo_diff','rest_diff','fatigue_diff','tilt_diff','black_draw_rate_10']; deep=['base_oof_prob_black','base_oof_prob_draw','base_oof_prob_white','momentum_oof_prob_black','momentum_oof_prob_draw','momentum_oof_prob_white']; df=df.dropna(subset=wide+deep+['target_result']).reset_index(drop=True)
    Xw=df[wide].values; Xd=df[deep].values.astype(np.float32); y=df.target_result.map({-1:0,0:1,1:2}).values
    t=TimeSeriesSplit(n_splits=4); acc=[]; losses=[]
    for tr,va in t.split(Xw):
        sc=StandardScaler().fit(Xw[tr]); m=WideAndDeepContext(6,5); opt=optim.Adam(m.parameters(),lr=.001,weight_decay=1e-4); lossfn=nn.CrossEntropyLoss(); tw=torch.tensor(sc.transform(Xw[tr]),dtype=torch.float32); td=torch.tensor(Xd[tr]); ty=torch.tensor(y[tr]);
        ds=TensorDataset(tw,td,ty); loader=DataLoader(ds,batch_size=512,shuffle=True)
        m.train()
        for _ in range(80):
            for wb,db,yb in loader: opt.zero_grad(); o=m(wb,db); l=lossfn(o,yb); l.backward(); opt.step()
        m.eval();
        with torch.no_grad(): pr=torch.softmax(m(torch.tensor(sc.transform(Xw[va]),dtype=torch.float32),torch.tensor(Xd[va])),1).numpy()
        acc.append(accuracy_score(y[va],pr.argmax(1))); losses.append(log_loss(y[va],pr,labels=[0,1,2]))
    print(f'Walk-forward pregame meta accuracy: {np.mean(acc)*100:.2f}%'); print(f'Walk-forward pregame meta log loss: {np.mean(losses):.4f}')
    sc=StandardScaler().fit(Xw); m=WideAndDeepContext(6,5); opt=optim.Adam(m.parameters(),lr=.001,weight_decay=1e-4); lossfn=nn.CrossEntropyLoss(); ds=TensorDataset(torch.tensor(sc.transform(Xw),dtype=torch.float32),torch.tensor(Xd),torch.tensor(y)); loader=DataLoader(ds,batch_size=512,shuffle=True); m.train()
    for _ in range(100):
        for wb,db,yb in loader: opt.zero_grad(); l=lossfn(m(wb,db),yb); l.backward(); opt.step()
    os.makedirs(os.path.dirname(OUT),exist_ok=True); pickle.dump({'model_state_dict':m.state_dict(),'wide_scaler':sc,'wide_cols':wide,'deep_cols':deep},open(OUT,'wb')); print('Saved',OUT)
if __name__=='__main__': main()
