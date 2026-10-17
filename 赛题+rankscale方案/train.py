# -*- coding: utf-8 -*-
"""
================================================================================
 复现训练脚本 —— 从比赛公开数据完整复现 model/ 下的 40 个 booster
================================================================================

 用法
     python train.py --data_dir <公开数据目录> --out_dir ./model

 --data_dir 需包含以下子目录（比赛公开数据）：
     MarketData/  MarketDataRawOpen/  Factor33_winsor/  Barra_factor/
     HS300_weight/          （仅用于确定股票池；与 Barra_factor 代码集合一致）

 与推理的一致性
   · 特征全部由同目录下的 features.py 计算 —— 训练与推理【同一份代码】
   · 标签、特征处理、目标定义与 inference.py 完全一致

 严格 Point-in-Time：因子只用 shift(正数) 与向后 rolling；最长回看 100 个交易日。
================================================================================
"""
from __future__ import annotations

import os
import glob
import json
import argparse

import numpy as np
import pandas as pd
import lightgbm as lgb

import features as FT

SEEDS = [42, 202, 777, 1234, 2024, 7, 99, 555, 888, 31337]
GROUPS = [
    dict(name='g1', num_leaves=15, max_depth=4, learning_rate=0.04, n_estimators=400,
         min_child_samples=200, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0),
    dict(name='g2', num_leaves=31, max_depth=5, learning_rate=0.03, n_estimators=500,
         min_child_samples=150, subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0),
    dict(name='g3', num_leaves=7, max_depth=3, learning_rate=0.05, n_estimators=350,
         min_child_samples=300, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0),
    dict(name='g4', num_leaves=63, max_depth=6, learning_rate=0.02, n_estimators=600,
         min_child_samples=100, subsample=0.7, colsample_bytree=0.6, reg_lambda=10.0),
]
BASE = dict(objective='regression', subsample_freq=1, n_jobs=-1, verbose=-1, random_state=42)
F33_NEED = ['float_market_cap', 'total_market_cap', 'pe_ttm', 'pb', 'beta_000300SH_22']
TRAIN_START, TRAIN_END = 20160101, 20221230


def _read_parquet_dir(root, name, **kw):
    files = sorted(glob.glob(os.path.join(root, name, '*', '*', '*.parquet')))
    if not files:
        files = sorted(glob.glob(os.path.join(root, name, '*.parquet')))
    return pd.concat([pd.read_parquet(f, **kw) for f in files], ignore_index=True)


def load_data(data_dir):
    mk = _read_parquet_dir(data_dir, 'MarketData')
    f33 = _read_parquet_dir(data_dir, 'Factor33_winsor')
    bar = _read_parquet_dir(data_dir, 'Barra_factor')
    ro = _read_parquet_dir(data_dir, 'MarketDataRawOpen')
    keep33 = [c for c in dict.fromkeys(FT.F33_NEEDED + FT.F33_RAW) if c in f33.columns]
    df = mk.merge(f33[['日期', '代码'] + keep33], on=['日期', '代码'], how='left')
    df = df.merge(bar[['日期', '代码'] + FT.STYLE10], on=['日期', '代码'], how='left')
    ro = ro.rename(columns={'open': 'raw_open'})
    df = df.merge(ro[['日期', '代码', 'raw_open']], on=['日期', '代码'], how='left')
    return df.loc[:, ~df.columns.duplicated()]


def build_training_set(df):
    F = FT.compute_features(df)
    F = F.merge(df[['日期', '代码', 'raw_open', 'is_suspend']], on=['日期', '代码'], how='left')
    F = F.sort_values(['代码', '日期']).reset_index(drop=True)
    g = F.groupby('代码', sort=False)
    o1, o2 = g['raw_open'].shift(-1), g['raw_open'].shift(-2)
    lab = np.where((o1 > 0) & (o2 > 0), o2 / o1 - 1.0, np.nan)
    F['label'] = np.where(np.abs(lab) > 0.5, np.nan, lab)
    s1, s2 = g['is_suspend'].shift(-1), g['is_suspend'].shift(-2)
    F['ok'] = F['label'].notna() & (s1.fillna(1) == 0) & (s2.fillna(1) == 0)

    tr = F[F['ok']].copy()
    tr = tr[(tr['日期'] >= TRAIN_START) & (tr['日期'] <= TRAIN_END)]
    tr = tr[tr[FT.ALL_FEATURES].notna().sum(axis=1) >= 30]
    tr['y'] = (tr.groupby('日期')['label'].rank(pct=True) * 10).clip(upper=9.999)
    Z = FT.cs_zscore(tr, FT.ALL_FEATURES, clip=50.0).fillna(0.0)
    return tr, Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--out_dir', default='./model')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print('载入数据 ...')
    df = load_data(args.data_dir)
    print('  %d 行，%d 交易日' % (len(df), df['日期'].nunique()))

    tr, Z = build_training_set(df)
    print('训练样本 %d 行，%d 交易日' % (len(tr), tr['日期'].nunique()))
    Zv = np.ascontiguousarray(Z.to_numpy(dtype=np.float32))
    yv = tr['y'].to_numpy(dtype=np.float64)
    assert Zv.shape[1] == len(FT.ALL_FEATURES)

    models = []
    print('训练 %d 组 × %d 种子 = %d 个模型 ...'
          % (len(GROUPS), len(SEEDS), len(GROUPS) * len(SEEDS)))
    for grp in GROUPS:
        for sd in SEEDS:
            p = {**BASE, **{k: v for k, v in grp.items() if k != 'name'}, 'random_state': sd}
            m = lgb.LGBMRegressor(**p)
            m.fit(Zv, yv)
            fn = 'model_%s_s%d.txt' % (grp['name'], sd)
            m.booster_.save_model(os.path.join(args.out_dir, fn),
                                  num_iteration=m.best_iteration_)
            models.append(dict(file=fn, group=grp['name'], seed=sd,
                               weight=round(1.0 / (len(GROUPS) * len(SEEDS)), 6)))
        print('  %s 完成' % grp['name'])

    meta = dict(
        plan='CSI300 Index Enhancement — 38 因子 LightGBM 多种子集成',
        n_features=len(FT.ALL_FEATURES), feature_cols=FT.ALL_FEATURES,
        barra_cols=FT.STYLE10, lookback=FT.LOOKBACK, max_lookback=FT.MAX_LOOKBACK,
        groups=GROUPS, seeds=SEEDS, base_params=BASE,
        blend='weighted per-day percentile-rank average over all models',
        target='per-day decile rank (0-9) of Open(t+2)/Open(t+1)-1',
        label='Open(t+2)/Open(t+1)-1, |y|>0.5 dropped',
        n_models=len(models), models=models,
        nan_fill=0.0, clip_z=50.0, min_feature_nonnull=30, min_cs_size=50,
        prediction_direction='higher score = better expected relative performance',
        train_rows=int(len(tr)), train_dates=int(tr['日期'].nunique()),
        train_period='%d..%d' % (TRAIN_START, TRAIN_END), score_power=1.0,
    )
    with open(os.path.join(args.out_dir, 'meta.json'), 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    print('完成：%d 个模型 + meta.json → %s' % (len(models), args.out_dir))


if __name__ == '__main__':
    main()
