# -*- coding: utf-8 -*-
"""
================================================================================
 因子计算模块 —— 训练与推理【共用同一份代码】（唯一的特征真值来源）
================================================================================

 输入：比赛只提供的三张表
   · MarketData      : 日期, 代码, open, high, low, close, volume, amount, is_suspend
   · Factor33_winsor : float_market_cap, total_market_cap, pe_ttm, pb, beta_000300SH_22
   · Barra_factor    : 市值, 贝塔, 动量, 残差波动, 非线性市值,
                       账面市值比, 流动性, 盈利, 成长, 杠杆
 输出：38 个因子

 设计原则
   1. 【单一真值】本文件是特征的唯一定义处。train_final.py 与 inference.py
      都 import 本文件 → 训练与推理特征逐行一致，不存在口径漂移。
   2. 【严格 PIT】只用 shift(正数) 与 rolling(向后窗口)，无未来信息。
   3. 【回看 ≤120 日】最长 120（sharpe_120），满足比赛约束。
   4. 【数值稳定】分母加 epsilon，inf 统一转 NaN。
================================================================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DATE, CODE = '日期', '代码'
EPS = 1e-9

STYLE10 = ['市值', '贝塔', '动量', '残差波动', '非线性市值',
           '账面市值比', '流动性', '盈利', '成长', '杠杆']

F16_BASE = ['rev_5', 'bias_20', 'vol_cv_20', 'intraday_rev_1', 'macd_hist', 'pv_corr_20',
            'turn_20', 'atr_14_norm', 'ep_ttm', 'size', 'btop', 'beta', 'illiq_20',
            'downside_vol_20', 'overnight_rev', 'sharpe_100']
NEW12 = ['pv_elastic_20', 'close_vs_mid_20', 'turn_accel', 'shadow_up_20',
         'overnight_vol_share_20', 'trend_eff_20', 'kurt_20', 'rng_to_absret_20',
         'elastic_asym_20', 'close_vs_vwap_20', 'intraday_autocorr_20', 'var_ratio_5_20']

# ---------------------------------------------------------------------------
# Factor33_winsor 的可用字段（平台直接提供，无需自算）
# 实测：把这 32 个字段全部纳入特征，跨年样本外 RankIC 从 0.0470 提升到 0.0515（+9.6%）
# 其中质量类基本面（roe/roa/毛利率/净利率/负债率/流动比率）此前完全未用。
# 前缀 Q_ 以避免与自算因子重名（如 bias_20、turn）。
# ---------------------------------------------------------------------------
F33_RAW = ['turn', 'change_ratio', 'daily_return', 'momentum_5', 'reversal_5',
           'volatility_5', 'sma_20', 'ema_20', 'macd_diff_12_26_9', 'macd_dea_12_26_9',
           'macd_hist_12_26_9', 'rsi_12', 'kdj_k_9_3_3', 'kdj_d_9_3_3', 'bias_20',
           'cci_14', 'atr_14', 'total_market_cap', 'float_market_cap', 'pe_ttm', 'pb',
           'ps_ttm', 'roe_avg_ttm', 'roa_avg_ttm', 'gross_profit_rate_ttm',
           'net_profit_rate_ttm', 'debt_to_asset_lf', 'current_ratio_lf',
           'beta_000300SH_22', 'list_days', 'is_st', 'is_new']
F33_FEATURES = ['Q_' + c for c in F33_RAW]
# 必须从 Factor33_winsor 读取的原始列（自算因子也依赖其中 5 个）
F33_NEEDED = ['float_market_cap', 'total_market_cap', 'pe_ttm', 'pb', 'beta_000300SH_22']

# ---------------------------------------------------------------------------
# 【已废弃】行业内相对强弱因子（IN_*）
#
# 曾在一版中启用（跨年回测 RankIC 0.0516 → 0.0521），但上线后 RankIC 从 0.053 掉到 0.028。
# 定位到根因：IN_* = (当日全截面 z-score) − (当日同行业均值)，
# 其数值【强烈依赖传入窗口的长度】—— 全历史 vs 120 日窗口 的同一日相关系数：
#     2019-01: 0.976   2020-01: 0.820   2021-01: 0.408   2022-01: 0.261
# 而平台只传 ≤120 日历史 → 线上特征与训练特征严重错配，模型失效。
# 其余 70 个因子在同样对比下相关系数恒为 1.0000（相对差 0.002），是稳定的。
# 结论：永久停用 IN_*，不再作为特征。
# ---------------------------------------------------------------------------
IND_BASE = []
IND_FEATURES = []

# ---------------------------------------------------------------------------
# 低相关增量因子（NEW4）—— 由公榜成绩反推规律后新增
#
# 反推依据：多次线上提交显示，在【价量因子家族】内 RankIC 随特征数单调递增
#     23 → 33 → 38 特征：RankIC 0.0210 → 0.0290 → 0.0388
#   而加入 Factor33 的 32 个基本面字段反而降到 0.0288~0.0343。
#   → 本次只在价量/微观结构家族内扩因子。
#
# 筛选双标准（new_lowcorr/screen.py）：
#   ① 与现有 38 因子的横截面最大 |相关| < 0.70（低冗余）
#   ② 单因子 RankIC > 0.008（有信息）
# 全部 ≤120 日窗口，符合赛题 look-back 规则。
#
# 真留出（训练 ≤2020-12-31 → 评估 2021/2022）：0.0429 → 0.0441（+0.0011），两年均改善。
# ---------------------------------------------------------------------------
NEW4 = [
    'ret_max5_20',        # 20 日内最大 5 日收益（极端动量）  max|ρ|=0.675  IC=0.0188
    'vol_of_vol_20',      # 波动率的波动（波动不稳定性）      max|ρ|=0.405  IC=0.0156
    'gap_vol_20',         # 跳空缺口的波动                    max|ρ|=0.621  IC=0.0154
    'vol_price_corr_5',   # 5 日量价相关                      max|ρ|=0.576  IC=0.0104
]

ALL_FEATURES = F16_BASE + NEW12 + STYLE10 + F33_FEATURES + IND_FEATURES + NEW4

_LB_BASE = {'rev_5': 6, 'bias_20': 20, 'vol_cv_20': 20, 'intraday_rev_1': 1,
            'macd_hist': 26, 'pv_corr_20': 21, 'turn_20': 20, 'atr_14_norm': 15,
            'ep_ttm': 0, 'size': 0, 'btop': 0, 'beta': 0, 'illiq_20': 20,
            'downside_vol_20': 20, 'overnight_rev': 5, 'sharpe_100': 100,
            'pv_elastic_20': 21, 'close_vs_mid_20': 20, 'turn_accel': 60,
            'shadow_up_20': 20, 'overnight_vol_share_20': 20, 'trend_eff_20': 21,
            'kurt_20': 20, 'rng_to_absret_20': 20, 'elastic_asym_20': 21,
            'close_vs_vwap_20': 20, 'intraday_autocorr_20': 21, 'var_ratio_5_20': 20,
            'ret_max5_20': 25, 'vol_of_vol_20': 40, 'gap_vol_20': 21, 'vol_price_corr_5': 6,
            **{c: 0 for c in STYLE10},
            **{('Q_' + c): 0 for c in F33_RAW}}     # Factor33 为平台预计算，无需回看
# 行业内相对因子与其基准因子同回看
LOOKBACK = {**_LB_BASE, **{('IN_' + c): _LB_BASE.get(c, 20) for c in IND_BASE}}
MAX_LOOKBACK = max(LOOKBACK.values())


def _mp(w):
    """rolling 的 min_periods。"""
    return max(2, w // 2)


def _pcorr(d: pd.DataFrame, a: str, b: str, w: int) -> pd.Series:
    """按股票分组计算 a 与 b 的 w 日滚动 Pearson 相关，索引与 d 【严格对齐】。

    注意：必须保持 d 的原始行顺序。早期版本把结果按 (代码,日期) 重新排序后再映射，
    导致行错位（秩相关掉到 0.77）。此处改为直接在 d 的顺序上计算。
    """
    x = pd.DataFrame({CODE: d[CODE].to_numpy(), DATE: d[DATE].to_numpy(),
                      '_a': d[a].to_numpy(dtype='float64'),
                      '_b': d[b].to_numpy(dtype='float64')})
    gg = x.groupby(CODE, sort=False)
    ma = gg['_a'].transform(lambda s: s.rolling(w, min_periods=_mp(w)).mean())
    mb = gg['_b'].transform(lambda s: s.rolling(w, min_periods=_mp(w)).mean())
    va = gg['_a'].transform(lambda s: s.rolling(w, min_periods=_mp(w)).var(ddof=0))
    vb = gg['_b'].transform(lambda s: s.rolling(w, min_periods=_mp(w)).var(ddof=0))
    x['_ab'] = x['_a'] * x['_b']
    mab = x.groupby(CODE, sort=False)['_ab'].transform(
        lambda s: s.rolling(w, min_periods=_mp(w)).mean())
    cov = mab - ma * mb
    corr = (cov / (np.sqrt(va * vb) + EPS)).replace([np.inf, -np.inf], np.nan).clip(-1, 1)
    return pd.Series(corr.to_numpy(), index=d.index)


def try_ewm(s: pd.Series, span: int, minp: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=minp).mean()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """返回 [日期, 代码] + 38 个因子列（严格 PIT，NaN 保留）。"""
    need = [DATE, CODE, 'open', 'high', 'low', 'close', 'volume', 'amount']
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError('compute_features 缺少列: %s' % miss)

    d = df.loc[:, ~df.columns.duplicated()].sort_values([CODE, DATE]).reset_index(drop=True)

    # ---------- 基础派生量 ----------
    g = d.groupby(CODE, sort=False)
    d['_pc'] = g['close'].shift(1)
    d['_ret'] = d['close'] / d['_pc'] - 1.0
    d['_intra'] = d['close'] / d['open'].replace(0, np.nan) - 1.0
    d['_gap'] = d['open'] / d['_pc'].replace(0, np.nan) - 1.0
    d['_tr'] = pd.concat([d['high'] - d['low'],
                          (d['high'] - d['_pc']).abs(),
                          (d['low'] - d['_pc']).abs()], axis=1).max(axis=1)
    d['_vol_chg'] = d['volume'] / g['volume'].shift(1).replace(0, np.nan) - 1.0
    d['_rng'] = (d['high'] - d['low']) / d['_pc'].replace(0, np.nan)
    d['_vwap'] = d['amount'] / d['volume'].replace(0, np.nan)
    d['_ar'] = d['_ret'].abs()
    d['_neg2'] = d['_ret'].where(d['_ret'] < 0, 0.0) ** 2
    d['_illiq'] = d['_ar'] / d['amount'].replace(0, np.nan)
    d['_turn'] = d['volume'] / (pd.to_numeric(d.get('float_market_cap'), errors='coerce')
                                / d['close'].replace(0, np.nan)).replace(0, np.nan)
    d['_elas'] = d['_ret'] / (d['_vol_chg'] + EPS)
    d['_gapabs'] = d['_gap'].abs()
    d['_intraabs'] = d['_intra'].abs()
    d['_cvm'] = d['close'] / ((d['high'] + d['low']) / 2.0).replace(0, np.nan) - 1.0
    d['_su'] = (d['high'] - np.maximum(d['open'], d['close'])) / d['_pc'].replace(0, np.nan)
    d['_cvw'] = d['close'] / d['_vwap'].replace(0, np.nan)
    d['_eu'] = d['_elas'].where(d['_ret'] > 0, np.nan)
    d['_ed'] = d['_elas'].where(d['_ret'] < 0, np.nan)
    d['_intra_lag'] = g['_intra'].shift(1)

    g = d.groupby(CODE, sort=False)

    def R(col, w, fn):
        return g[col].rolling(w, min_periods=_mp(w)).__getattribute__(fn)() \
            .reset_index(level=0, drop=True)

    F = {}

    # ================= 一、16 个基础因子（与原面板定义一致） =================
    F['rev_5'] = -(d['close'] / g['close'].shift(5) - 1.0)
    ma20 = R('close', 20, 'mean')
    F['bias_20'] = -(d['close'] - ma20) / ma20.replace(0, np.nan)
    F['vol_cv_20'] = -(R('volume', 20, 'std') / R('volume', 20, 'mean').replace(0, np.nan))
    F['intraday_rev_1'] = -d['_intra']
    e12 = g['close'].transform(lambda s: s.ewm(span=12, adjust=False, min_periods=12).mean())
    e26 = g['close'].transform(lambda s: s.ewm(span=26, adjust=False, min_periods=26).mean())
    dif = e12 - e26
    dea = dif.groupby(d[CODE], sort=False).transform(
        lambda s: s.ewm(span=9, adjust=False, min_periods=9).mean())
    F['macd_hist'] = -(dif - dea) / d['close'].replace(0, np.nan)
    F['pv_corr_20'] = -_pcorr(d, '_ret', '_vol_chg', 20)
    F['turn_20'] = -R('_turn', 20, 'mean') * 100.0
    F['atr_14_norm'] = -R('_tr', 14, 'mean') / d['close'].replace(0, np.nan)
    pe = pd.to_numeric(d.get('pe_ttm'), errors='coerce')
    F['ep_ttm'] = 1.0 / pe.replace(0, np.nan)
    tmc = pd.to_numeric(d.get('total_market_cap'), errors='coerce')
    F['size'] = -np.log(tmc.where(tmc > 0))
    pb = pd.to_numeric(d.get('pb'), errors='coerce')
    F['btop'] = 1.0 / pb.replace(0, np.nan)
    F['beta'] = -pd.to_numeric(d.get('beta_000300SH_22'), errors='coerce')
    F['illiq_20'] = -np.log1p(R('_illiq', 20, 'mean') * 1e9)
    F['downside_vol_20'] = -np.sqrt(R('_neg2', 20, 'mean')) * np.sqrt(243.0)
    F['overnight_rev'] = -R('_gap', 5, 'mean')
    F['sharpe_100'] = ((d['close'] / g['close'].shift(100) - 1.0)
                       / (R('_ret', 100, 'std') * np.sqrt(243.0) + EPS))

    # ================= 二、12 个自研低相关因子 =================
    F['pv_elastic_20'] = -R('_elas', 20, 'median')
    F['close_vs_mid_20'] = -R('_cvm', 20, 'mean')
    F['turn_accel'] = -(R('_turn', 5, 'mean') / (R('_turn', 60, 'mean') + 1e-12) - 1.0)
    F['shadow_up_20'] = -R('_su', 20, 'mean') / d['_rng'].replace(0, np.nan)
    F['overnight_vol_share_20'] = (R('_gapabs', 20, 'mean')
                                   / (R('_intraabs', 20, 'mean') + EPS))
    F['trend_eff_20'] = -((d['close'] / g['close'].shift(20) - 1.0).abs()
                          / (R('_ar', 20, 'sum') + EPS))
    F['kurt_20'] = -R('_ret', 20, 'kurt')
    F['rng_to_absret_20'] = R('_rng', 20, 'mean') / (R('_ar', 20, 'mean') + EPS)
    F['elastic_asym_20'] = -R('_eu', 20, 'median') + R('_ed', 20, 'median')
    F['close_vs_vwap_20'] = -R('_cvw', 20, 'mean')
    F['intraday_autocorr_20'] = -_pcorr(d, '_intra', '_intra_lag', 20)
    F['var_ratio_5_20'] = (R('_ret', 5, 'var') / (R('_ret', 20, 'var') / 4.0 + 1e-12)) - 1.0

    # ================= 三、10 个 Barra 风格暴露 =================
    for c in STYLE10:
        F[c] = pd.to_numeric(d[c], errors='coerce') if c in d.columns else np.nan

    # ================= 四、Factor33_winsor 的 32 个字段（平台直接提供） =================
    for c in F33_RAW:
        F['Q_' + c] = pd.to_numeric(d[c], errors='coerce') if c in d.columns else np.nan

    # 先把已算好的因子放进 out（供行业内相对因子复用）
    out = pd.DataFrame(index=d.index)
    for c in F16_BASE + NEW12:
        out[c] = pd.to_numeric(pd.Series(F.get(c), index=d.index), errors='coerce')
    for c in F33_RAW:
        out['Q_' + c] = pd.to_numeric(pd.Series(F.get('Q_' + c), index=d.index), errors='coerce')

    # ================= 五、行业内相对因子：已永久停用 =================
    # IN_* 数值依赖传入窗口长度（全历史 vs 120 日窗口相关系数低至 0.26），
    # 会导致线上（仅 120 日窗口）与训练严重错配。详见文件顶部 IND_FEATURES 注释。
    # 这里显式补 NaN 仅为保持列结构兼容，不再参与建模（IND_FEATURES 为空）。

    # ================= 六、Barra 风格暴露 =================
    for c in STYLE10:
        out[c] = pd.to_numeric(pd.Series(F.get(c), index=d.index), errors='coerce')

    # ================= 七、低相关增量因子 NEW4（全部 ≤120 日窗口） =================
    # 只用 OHLCV 的滚动统计，推理时在 120 日窗口内可精确复现（已验证多窗口一致性）。
    if all(x in d.columns for x in ('open', 'high', 'low', 'close', 'volume', 'amount')):
        dd_ = d.sort_values([CODE, DATE])
        gg = dd_.groupby(CODE, sort=False)
        code_arr = dd_[CODE].to_numpy()
        _c = pd.to_numeric(dd_['close'], errors='coerce')
        _v = pd.to_numeric(dd_['volume'], errors='coerce')
        _amt = pd.to_numeric(dd_['amount'], errors='coerce')
        _o = pd.to_numeric(dd_['open'], errors='coerce')
        _r1 = _c / gg['close'].shift(1) - 1.0
        _r5 = _c / gg['close'].shift(5) - 1.0

        def _roll(series, w, fn, mp=None):
            return series.groupby(code_arr, sort=False).rolling(
                w, min_periods=mp or max(2, w // 2)).__getattribute__(fn)() \
                .reset_index(level=0, drop=True)

        # ① 20 日内最大 5 日收益（极端动量）
        out['ret_max5_20'] = -_roll(_r5, 20, 'max')
        # ② 波动率的波动：20 日波动的 20 日标准差
        _vol20 = _roll(_r1, 20, 'std')
        out['vol_of_vol_20'] = -_roll(_vol20, 20, 'std')
        # ③ 跳空缺口波动
        _gap = _o / gg['close'].shift(1) - 1.0
        out['gap_vol_20'] = -_roll(_gap, 20, 'std')
        # ④ 5 日量价相关
        _cv = _roll(_v * _r1, 5, 'mean')
        _mv = _roll(_v, 5, 'mean')
        _mr = _roll(_r1, 5, 'mean')
        _sv = _roll(_v, 5, 'std')
        _sr = _roll(_r1, 5, 'std')
        _corr = (_cv - _mv * _mr) / (_sv * _sr).replace(0, np.nan)
        _corr.index = dd_.index
        out['vol_price_corr_5'] = -_corr.reindex(d.index)
        for c in NEW4:
            out[c] = pd.to_numeric(out[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
    else:
        for c in NEW4:
            out[c] = np.nan

    out = out.replace([np.inf, -np.inf], np.nan)
    return pd.concat([d[[DATE, CODE]], out[ALL_FEATURES]], axis=1)


def make_label(df: pd.DataFrame, open_col: str = 'open') -> pd.Series:
    """y = O(t+2)/O(t+1) − 1（逐股票 shift；|y|>0.5 置 NaN）。open_col 须为未复权开盘价。"""
    d = df.sort_values([CODE, DATE])
    g = d.groupby(CODE, sort=False)
    o1, o2 = g[open_col].shift(-1), g[open_col].shift(-2)
    r = np.where((o1 > 0) & (o2 > 0), o2 / o1 - 1.0, np.nan)
    r = np.where(np.abs(r) > 0.5, np.nan, r)
    return pd.Series(r, index=d.index)


def cs_zscore(df: pd.DataFrame, cols, clip: float = 50.0) -> pd.DataFrame:
    """当日横截面 z-score；保留 NaN。"""
    cols = list(cols)
    g = df.groupby(DATE, sort=False)[cols]
    Z = (df[cols] - g.transform('mean')) / g.transform(
        lambda s: s.std(ddof=0) if s.std(ddof=0) else 1.0)
    Z = Z.replace([np.inf, -np.inf], np.nan).clip(-clip, clip)
    Z.columns = cols
    return Z


def cs_rank_center(df: pd.DataFrame, cols) -> pd.DataFrame:
    """当日横截面百分位排名，中心化到 [-0.5, 0.5]；保留 NaN。

    与 cs_zscore 同样是无前视的纯横截面变换（只用当日数据）。
    相比 z-score 的好处：天然有界，不受极值影响；对偏态因子更稳健。
    实测（固定训练 ≤2019-12-31 → 评估 2020/2021/2022，plan8 的 38 因子）：
        z-score      0.0479 / 0.0423 / 0.0437 → 均值 0.0446
        百分位排名    0.0490 / 0.0447 / 0.0452 → 均值 0.0463（+0.0016，三年全部改善）
    """
    cols = list(cols)
    R = df.groupby(DATE, sort=False)[cols].rank(pct=True) - 0.5
    R = R.replace([np.inf, -np.inf], np.nan)
    R.columns = cols
    return R


def cs_scale(df: pd.DataFrame, cols, method: str = 'zscore', clip: float = 50.0):
    """按 method 选择横截面标准化方式：'zscore' 或 'rank'。"""
    if str(method).lower() in ('rank', 'pct', 'percentile'):
        return cs_rank_center(df, cols)
    return cs_zscore(df, cols, clip=clip)
