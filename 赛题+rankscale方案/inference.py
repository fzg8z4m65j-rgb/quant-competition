# -*- coding: utf-8 -*-
"""
================================================================================
 CSI 300 Index Enhancement Challenge — Inference entry (compliant)
================================================================================

 Implements the required interface:

     from inference import InferenceModel
     model = InferenceModel(model_dir="./model/")            # called once
     df = model.predict(as_of_date="2022-01-04",
                        data={"MarketData": ...,
                              "Factor33_winsor": ...,
                              "Barra_factor": ...})          # per signal day

 Returns exactly three columns: date, code, score
   date  : python str (object dtype), equal to as_of_date
   code  : object dtype, one row per eligible stock, no dups
   score : float64, higher = more bullish

 Design constraints honoured
   * Self-contained: imports only numpy / pandas / lightgbm + the sibling
     features.py (which ships inside this same directory).
   * NOTHING is fitted at inference time: every learned parameter lives in
     model_dir (the boosters listed in meta.json + meta.json itself).
   * Only model_dir is read; no network, no other files.
   * No sys.exit / subprocess / os.system / eval / exec / __import__.
   * Cross-call state-free; features are pure per-day cross-sectional transforms
     computed from the supplied history only (lookback <= 120 trading days).
================================================================================
"""
from __future__ import annotations

import os
import sys
import json

import numpy as np
import pandas as pd

# 允许以单文件目录形式被导入（平台把整个提交包放在同一目录）
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import features as FT          # noqa: E402  （同目录下的 features.py）

try:                            # lightgbm 的 Booster 是唯一外部依赖
    import lightgbm as lgb
except Exception as _e:         # pragma: no cover
    lgb = None
    _LGB_ERR = _e

_DATE, _CODE = '日期', '代码'


class InferenceModel:
    """加载冻结模型，对单个交易日截面打分。"""

    def __init__(self, model_dir: str):
        self.model_dir = str(model_dir)
        with open(os.path.join(self.model_dir, 'meta.json'), 'r', encoding='utf-8') as fh:
            self.meta = json.load(fh)

        self.feature_cols = list(self.meta['feature_cols'])
        self.n_features = int(self.meta.get('n_features', len(self.feature_cols)))
        self.clip_z = float(self.meta.get('clip_z', 50.0))
        self.nan_fill = float(self.meta.get('nan_fill', 0.0))
        self.min_cs_size = int(self.meta.get('min_cs_size', 50))
        self.score_power = float(self.meta.get('score_power', 1.0))
        self.max_lookback = int(self.meta.get('max_lookback', 120))
        # 特征横截面标准化方式：'zscore'（默认，向后兼容）或 'rank'（百分位排名）
        self.feature_scale = str(self.meta.get('feature_scale', 'zscore')).lower()

        if lgb is None:                                   # pragma: no cover
            raise RuntimeError('lightgbm 不可用: %s' % _LGB_ERR)

        self.models = []
        for spec in self.meta['models']:
            b = lgb.Booster(model_file=os.path.join(self.model_dir, spec['file']))
            self.models.append((b, float(spec.get('weight', 1.0))))
        if not self.models:                                # pragma: no cover
            raise RuntimeError('meta.json 中没有任何模型')

        # 注：曾有版本使用行业映射表计算「行业内相对因子」(IN_*)，
        #     但该因子在平台 120 日窗口下数值不稳定（与全历史相关系数低至 0.26），
        #     导致线上与训练错配，已永久停用。本版所有横截面因子都不依赖行业数据。

        # 长窗口因子的 PIT 表（可选）：250 日回看无法在推理时计算，故随包发布。
        self.pit = None
        pit_name = self.meta.get('pit_file')
        if pit_name:
            try:
                p = os.path.join(self.model_dir, pit_name)
                if os.path.exists(p):
                    self.pit = pd.read_parquet(p)
            except Exception:
                self.pit = None

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_frame(obj, name):
        if obj is None:
            return None
        if not isinstance(obj, pd.DataFrame):
            try:
                obj = pd.DataFrame(obj)
            except Exception:
                return None
        if obj.empty or _DATE not in obj.columns or _CODE not in obj.columns:
            return None
        return obj

    @staticmethod
    def _slice(table, date_int, codes):
        if table is None:
            return pd.DataFrame(index=codes)
        day = table.loc[table[_DATE] == date_int]
        if day.empty:
            return pd.DataFrame(index=codes)
        day = day.drop_duplicates(subset=[_CODE], keep='last')
        return day.set_index(_CODE).reindex(codes)

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def predict(self, as_of_date: str, data: dict) -> pd.DataFrame:
        as_of = str(as_of_date)
        t = int(as_of.replace('-', '').replace('/', ''))

        market = self._as_frame(data.get('MarketData'), 'MarketData')
        factors = self._as_frame(data.get('Factor33_winsor'), 'Factor33_winsor')
        barra = self._as_frame(data.get('Barra_factor'), 'Barra_factor')

        # ── 当日合法股票池：Barra_factor 的代码集合（与 HS300 成分零差异）；
        #    若 Barra 缺失则退回 MarketData ──
        universe = []
        for tbl in (barra, market):
            if tbl is not None:
                universe = (tbl.loc[tbl[_DATE] == t, _CODE]
                            .astype(str).dropna().unique().tolist())
            if universe:
                break
        if not universe:
            return pd.DataFrame({'date': pd.Series([], dtype=object),
                                 'code': pd.Series([], dtype=object),
                                 'score': pd.Series([], dtype='float64')})
        universe = list(dict.fromkeys(universe))

        # ── 用 ≤t 的历史计算 38 个因子（只保留需要的列，控制内存）──
        hist = market.loc[market[_DATE] <= t] if market is not None else None
        if hist is None or hist.empty:
            scores = np.zeros(len(universe), dtype='float64')
            return self._out(as_of, universe, scores)

        keep_mk = [_DATE, _CODE] + [c for c in ('open', 'high', 'low', 'close',
                                                'volume', 'amount', 'is_suspend')
                                    if c in hist.columns]
        merged = hist[keep_mk].copy()
        # Factor33_winsor：自算因子需要的 5 列 + 直接作为特征的 32 列
        f33_cols = [c for c in dict.fromkeys(FT.F33_NEEDED + FT.F33_RAW)]
        for tbl, cols in ((factors, f33_cols), (barra, list(FT.STYLE10))):
            if tbl is None:
                continue
            have = [_DATE, _CODE] + [c for c in cols if c in tbl.columns]
            if len(have) > 2:
                sub = tbl.loc[tbl[_DATE] <= t, have]
                merged = merged.merge(sub, on=[_DATE, _CODE], how='left')
        merged = merged.loc[:, ~merged.columns.duplicated()]

        feats = FT.compute_features(merged)

        # ── 取当日截面 ──
        cur = feats.loc[feats[_DATE] == t].drop_duplicates(subset=[_CODE], keep='last')
        cur = cur.set_index(_CODE).reindex(universe)
        X = cur.reindex(columns=self.feature_cols)
        X = X.apply(pd.to_numeric, errors='coerce')

        # ── 当日横截面标准化（与训练完全同源：FT.cs_scale）──
        Xz = pd.DataFrame(X.to_numpy(dtype='float64'),
                          index=range(len(universe)), columns=self.feature_cols)
        Xz.insert(0, _DATE, t)
        Z = FT.cs_scale(Xz, self.feature_cols, method=self.feature_scale, clip=self.clip_z)
        Z = Z.replace([np.inf, -np.inf], np.nan).fillna(self.nan_fill)
        Xv = np.ascontiguousarray(Z.to_numpy(dtype=np.float32))

        # ── 多模型：当日截面百分位排名加权平均（对量纲不敏感）──
        blended = np.zeros(len(universe), dtype='float64')
        wsum = 0.0
        if len(universe) >= self.min_cs_size:
            for booster, w in self.models:
                p = np.asarray(booster.predict(Xv), dtype='float64')
                r = pd.Series(p).rank(pct=True).to_numpy()
                blended += w * r
                wsum += w
        scores = blended / wsum if wsum > 0 else blended

        # ── 分数形态：score_power 固定 1.0（单调变换不改排序，但会改变平台侧权重分布）──
        if self.score_power != 1.0:
            scores = np.power(np.clip(scores, 0.0, None), self.score_power)

        return self._out(as_of, universe, np.nan_to_num(scores, nan=0.0,
                                                        posinf=0.0, neginf=0.0))

    @staticmethod
    def _out(as_of, universe, scores):
        """显式 dtype：date 为 str(object)，code 为 object，score 为 float64。"""
        return pd.DataFrame({
            'date': pd.Series([as_of] * len(universe), dtype=object),
            'code': pd.Series(list(universe), dtype=object),
            'score': pd.Series(np.asarray(scores, dtype='float64'), dtype='float64'),
        })
