"""Live Baseline + Momentum feature adapter.

This is intentionally conservative: if player history is unavailable, it uses
neutral values instead of fabricating a rating/form signal.
"""
from __future__ import annotations
import os, sqlite3
import numpy as np
import pandas as pd
import pickle

ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DB_PATH=os.path.join(ROOT,'data','omni_pundit.db')
BASE_MODEL=os.path.join(ROOT,'saved_models','baseline_expert.pkl')
MOM_MODEL=os.path.join(ROOT,'saved_models','momentum_expert_xgb.pkl')

def _key(s):
    return str(s or "").replace(" ", "").strip().lower()


def _surname(player):
    value = str(player or "").strip()
    return value.split(",", 1)[0].strip().lower()


def _latest(conn, player, date):
    surname = _surname(player)

    q = """
        SELECT white, black, white_elo, black_elo, date
        FROM baseline_features
        WHERE date < ?
          AND (
              lower(trim(white)) LIKE ?
              OR lower(trim(black)) LIKE ?
          )
        ORDER BY date DESC
    """

    pattern = surname + ",%"
    rows = conn.execute(q, [date, pattern, pattern]).fetchall()

    for r in rows:
        if _surname(r[0]) == surname:
            return float(r[2]) if r[2] is not None else np.nan

        if _surname(r[1]) == surname:
            return float(r[3]) if r[3] is not None else np.nan

    return np.nan

def baseline_probs(white,black,date,db_path=DB_PATH):
    art=pickle.load(open(BASE_MODEL,'rb'))
    model,scaler=art['model'],art['scaler']
    conn=sqlite3.connect(db_path)
    we,be=_latest(conn,white,date),_latest(conn,black,date)
    wh=bh=d=0
    wk = _surname(white)
    bk = _surname(black)
    rows = conn.execute(
    """
    SELECT white, black, target_result
    FROM baseline_features
    WHERE date < ?
      AND (
          (
              lower(trim(white)) LIKE ?
              AND lower(trim(black)) LIKE ?
          )
          OR
          (
              lower(trim(white)) LIKE ?
              AND lower(trim(black)) LIKE ?
          )
      )
    """,
    [date, wk + ",%", bk + ",%", bk + ",%", wk + ",%"],
).fetchall()
    conn.close()
    for rw, rb, y in rows:
        wwin = (y == 1) if _surname(rw) == wk else (y == -1)
        bwin = (y == -1) if _surname(rw) == wk else (y == 1)
        wh+=int(wwin); bh+=int(bwin); d+=int(y==0)
    has=int(np.isfinite(we) and np.isfinite(be))
    we=0 if not np.isfinite(we) else we; be=0 if not np.isfinite(be) else be
    X=np.array([[we,be,we-be,has,wh,bh,d]],dtype=float)
    p=model.predict_proba(scaler.transform(X))[0]
    return dict(zip(['black','draw','white'],map(float,p)))

def baseline_and_momentum(white,black,date,db_path=DB_PATH):
    base=baseline_probs(white,black,date,db_path)
    # Momentum artifact needs four residual features. Compute historical player
    # timelines up to the cutoff exactly from stored baseline_features.
    conn=sqlite3.connect(db_path)
    df=pd.read_sql("SELECT date,white,black,white_elo,black_elo,target_result FROM baseline_features WHERE date < ? ORDER BY date ASC",conn,params=[date])
    conn.close()
    df['date']=pd.to_datetime(df['date'])
    rows=[]
    for r in df.itertuples(index=False):
        rows += [(_key(r.white),'white',r.date,r.white_elo,r.black_elo,1.0 if r.target_result==1 else .5 if r.target_result==0 else 0.0),(_key(r.black),'black',r.date,r.black_elo,r.white_elo,1.0 if r.target_result==-1 else .5 if r.target_result==0 else 0.0)]
    pdf=pd.DataFrame(rows,columns=['player','color','date','own','opp','score']).sort_values(['player','date'])
    pdf['expected']=1/(1+10**(-(pdf['own']-pdf['opp'])/400))
    pdf['resid']=pdf['score']-pdf['expected']
    pdf['past']=pdf.groupby('player')['resid'].shift(1)
    pdf['roll']=pdf.groupby('player')['past'].rolling(10,min_periods=1).mean().reset_index(0,drop=True)
    pdf['vol']=pdf.groupby('player')['past'].rolling(10,min_periods=1).std().reset_index(0,drop=True).fillna(0)
    pdf['past_elo']=pdf.groupby('player')['own'].shift(1)
    pdf['elo10']=pdf.groupby('player')['past_elo'].shift(9)
    pdf['trend']=pdf['own']-pdf['elo10']
    pdf['cpast']=pdf.groupby(['player','color'])['resid'].shift(1)
    pdf['cform']=pdf.groupby(['player','color'])['cpast'].rolling(5,min_periods=1).mean().reset_index([0,1],drop=True).fillna(0)
    def get(player,color):
        z=pdf[(pdf.player==_key(player))&(pdf.color==color)]
        if z.empty: return [0,0,0,0]
        q=z.iloc[-1]
        return [q.roll if pd.notna(q.roll) else 0,q.trend if pd.notna(q.trend) else 0,q.vol if pd.notna(q.vol) else 0,q.cform if pd.notna(q.cform) else 0]
    w=get(white,'white'); b=get(black,'black')
    X=np.array([[w[0]-b[0],w[1]-b[1],w[2]-b[2],w[3]-b[3]]])
    art=pickle.load(open(MOM_MODEL,'rb'))
    p=art['model'].predict_proba(X)[0]
    return base,dict(zip(['black','draw','white'],map(float,p))),{}
