# -*- coding: utf-8 -*-
"""
提交包自检脚本（不需要原始数据，使用包内固化的校验样本）

    python selfcheck.py

校验内容
  1. 加载 model/ 下的 40 个 booster 与 meta.json
  2. 读取 model/_check_features.parquet（2022-06-30 的 343 只股票的 38 个已 z-score 特征）
     直接喂给模型，得到分数
  3. 与 model/_check_expected.parquet（打包时用完全相同流程算出的期望分数）逐行比对
  4. 检查输出契约

预期输出：max|Δ| = 0.000e+00  → 模型文件完整且可复现
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, 'model')


def main():
    meta = json.load(open(os.path.join(MODEL_DIR, 'meta.json'), encoding='utf-8'))
    print('模型数 %d ｜ 特征数 %d ｜ 最长回看 %d 日 ｜ 目标：%s'
          % (meta['n_models'], meta['n_features'], meta['max_lookback'], meta['target']))

    X = pd.read_parquet(os.path.join(MODEL_DIR, '_check_features.parquet'))
    exp = pd.read_parquet(os.path.join(MODEL_DIR, '_check_expected.parquet'))
    print('校验样本：%d 行 × %d 个特征列' % (X.shape[0], X.shape[1] - 1))

    Xv = np.ascontiguousarray(X[meta['feature_cols']].to_numpy(dtype=np.float32))
    acc = np.zeros(len(Xv))
    wsum = 0.0
    for spec in meta['models']:
        b = lgb.Booster(model_file=os.path.join(MODEL_DIR, spec['file']))
        acc += spec['weight'] * pd.Series(b.predict(Xv)).rank(pct=True).to_numpy()
        wsum += spec['weight']
    got = acc / wsum

    m = pd.DataFrame({'代码': X['代码'].to_numpy(), 'got': got}).merge(exp, on='代码', how='inner')
    maxdiff = float((m['got'] - m['expected_score']).abs().max())
    rho = m['got'].corr(m['expected_score'], method='spearman')

    print('\n与期望分数比对：n=%d  max|Δ| = %.3e  Spearman = %.8f' % (len(m), maxdiff, rho))
    ok = maxdiff < 1e-9
    print('判定：%s' % ('PASS ✅ 模型文件完整、可复现' if ok else 'FAIL ⚠ 模型文件与打包时不一致'))

    n_booster = len([f for f in os.listdir(MODEL_DIR) if f.startswith('model_') and f.endswith('.txt')])
    print('\n契约检查：')
    print('  booster 文件数：%d（应为 %d）%s'
          % (n_booster, meta['n_models'], '✅' if n_booster == meta['n_models'] else '⚠'))
    print('  meta.json 存在：%s' % os.path.exists(os.path.join(MODEL_DIR, 'meta.json')))
    print('  权重之和 = %.6f' % wsum)
    print('  特征列顺序 = meta["feature_cols"]：%s' % (list(X.columns[1:]) == meta['feature_cols']))
    print('  分数全为有限值：%s' % bool(np.isfinite(got).all()))
    print('  score_power = %s（应为 1.0）' % meta['score_power'])
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
