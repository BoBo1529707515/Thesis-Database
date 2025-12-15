
import os               # 操作系统相关：路径、目录等
import re               # 正则表达式，用于从文件名中解析 Mouse 标签
import numpy as np      # 数值计算库
import scipy.io as sio  # 读取 .mat 文件
import matplotlib.pyplot as plt                 # 作图
from matplotlib import gridspec, transforms     # 网格排版与坐标变换
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # 在轴内嵌子轴（色条位置）
from scipy.stats import rankdata, mannwhitneyu  # 秩转换与曼-惠特尼 U 检验
from scipy.optimize import linear_sum_assignment # 匈牙利算法（线性分配）
from scipy.ndimage import gaussian_filter1d     # 一维高斯平滑

# ===== Paths & IO =====
INPUT_FILES = [
    r"F:\工作文件\RA\数据集\Fig2C\FVB_Panneuronal_Reunion_Isolation_Day3_Mouse#1_preprocessed.mat",  # 第1只鼠数据
    r"F:\工作文件\RA\数据集\Fig2C\FVB_Panneuronal_Reunion_Isolation_Day3_Mouse#2_preprocessed.mat",  # 第2只鼠
    r"F:\工作文件\RA\数据集\Fig2C\FVB_Panneuronal_Reunion_Isolation_Day3_Mouse#3_preprocessed.mat",  # 第3只鼠
    r"F:\工作文件\RA\数据集\Fig2C\FVB_Panneuronal_Reunion_Isolation_Day3_Mouse#4_preprocessed.mat",  # 第4只鼠
    r"F:\工作文件\RA\数据集\Fig2C\FVB_Panneuronal_Reunion_Isolation_Day3_Mouse#5_(FVB1_2)_preprocessed.mat",  # 第5只鼠
    r"F:\工作文件\RA\数据集\Fig2C\FVB_Panneuronal_Reunion_Isolation_Day3_Mouse#6_(FVB3_2)_preprocessed.mat",  # 第6只鼠
]
OUT_DIR = r"F:\工作文件\RA\数据集\Fig2C\fig2c_out"  # 输出目录（图与汇总）

# ===== Windows (sec) =====
PRE_SEC  = 300.0  # 事件前窗口长度（秒），用于截窗与对比
POST_SEC = 600.0  # 事件后窗口长度（秒）

# ===== ROC/AUC configuration =====
# Signal used for ROC: 'raw' | 'raw_smooth' | 'denoised'
AUC_SIGNAL = "raw_smooth"  # AUC 使用的信号来源：原始、平滑原始或去噪
SMOOTH_SEC = 1.0            # raw_smooth 时，高斯平滑的 σ（秒）
AUC_POST_SEC = 120.0        # AUC 只评估事件后前120秒（提高分离度）

# Threshold mode: 'fixed' | 'quantile'
AUC_THR_MODE = "fixed"      # AUC 阈值模式：固定阈或分位数
AUC_REUNION_THR  = 0.65     # 固定阈模式下：重聚高响应阈值
AUC_ISOLATION_THR = 0.35    # 固定阈模式下：隔离高响应阈值
AUC_Q = 0.18                # 分位数模式下：低/高尾分位（例如 18% 与 82%）

# Optional significance test (Mann–Whitney U). If True, keep only p<ALPHA.
USE_MWU_SIGNIF = False      # 是否启用显著性检验（默认关闭）
ALPHA = 0.05                # 显著性水平

# ===== Selection rule for "714" =====
EXCLUDE_TZ_ROWS = True      # 是否排除能被 traces_zscored 匹配上的 C 行
UNION_CELLS_IDX = True      # 是否把 mat['cells'] 指定的细胞集合再并回（支持 TZ 或 C 空间）

# ===== Plot style =====
CMAP = "viridis"            # 颜色映射
VMIN, VMAX = 0.0, 3.0       # 热图值域裁剪（Z-score）
GREEN = "#2ca02c"           # 绿色（Isolation）
PURPLE = "#7b61ff"          # 紫色（Reunion）
FIGSIZE = (12.3, 8.6)       # 画布尺寸（英寸）
DPI = 300                   # 输出分辨率

# ===== Helpers =====
def parse_mouse_tag(path):
    """从文件名中提取 'Mouse#\d+' 作为标签；若失败则返回文件名。
    参数: path(str)
    返回: 标签字符串
    """
    m = re.search(r"Mouse#\d+", os.path.basename(path))  # 正则在文件名中搜 Mouse#数字
    return m.group(0) if m else os.path.basename(path)     # 找到则返回匹配；否则返回文件名


def get_struct_field(mat, struct_name, field_name):
    """鲁棒地从 .mat 的结构体/数组里取字段。
    - 兼容 struct_as_record=False + squeeze_me=True 造成的不同形态
    - 若是 struct 数组，会遍历元素查找第一个带该字段的对象
    """
    if struct_name not in mat:         # 若根本没有该结构体
        return None
    obj = mat[struct_name]             # 取出对象（可能是 ndarray/对象）
    try:
        if hasattr(obj, field_name):   # 若对象直接有该字段
            return getattr(obj, field_name)
    except Exception:
        pass
    if isinstance(obj, np.ndarray):    # 若是 numpy 数组，可能包着一个 struct
        if obj.size == 1:              # 单元素数组：取标量后再查字段
            obj = obj.reshape(-1)[0]
            if hasattr(obj, field_name):
                return getattr(obj, field_name)
        else:
            for el in obj.flat:        # 多元素：逐个元素找第一个拥有该字段的
                if hasattr(el, field_name):
                    return getattr(el, field_name)
    return None                        # 都没找到则返回 None


def row_zscore(x):
    """对矩阵按行做 z-score 标准化，避免除零。
    x: [n_cells, n_time]
    返回同形矩阵
    """
    mu = np.nanmean(x, axis=1, keepdims=True)   # 每行均值（忽略 NaN）
    sd = np.nanstd(x, axis=1, keepdims=True)    # 每行标准差
    sd[sd < 1e-9] = 1.0                          # 防止除以接近 0
    return (x - mu) / sd                         # 标准化


def load_one_mouse(path):
    """读取单只小鼠的数据，尽力从不同位置找 Fs/frame_in/C/C_raw。
    返回 dict: {mouse, fs, frame_in, C_den, C_raw, mat, path}
    """
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)  # 读取 .mat，自动压缩维度
    keys = set(k for k in mat.keys() if not k.startswith("__"))       # 过滤掉 __header__ 等

    # Fs（采样率）
    fs = None
    if "Fs" in keys:
        fs = float(np.asarray(mat["Fs"]).squeeze())  # 直接取 Fs
    if fs is None:
        for s in ("results_final", "result_merged", "results_filtered", "results"):  # 依次在常见 struct 里找
            v = get_struct_field(mat, s, "Fs")
            if v is not None:
                fs = float(np.asarray(v).squeeze()); break
    if fs is None:
        raise RuntimeError(f"{os.path.basename(path)}: cannot find 'Fs'.")  # 仍找不到则报错

    # frame_in / time_in（事件帧或秒）
    frame_in = None
    if "frame_in" in keys:
        frame_in = int(np.asarray(mat["frame_in"]).squeeze())  # 直接取事件帧索引
    if frame_in is None and "time_in" in keys:
        frame_in = int(round(float(np.asarray(mat["time_in"]).squeeze()) * fs))  # 用时间×Fs 换算帧
    if frame_in is None:
        for s in ("results_final", "result_merged", "results_filtered", "results"):
            v = get_struct_field(mat, s, "frame_in")
            if v is not None:
                frame_in = int(np.asarray(v).squeeze()); break
        if frame_in is None:
            for s in ("results_final", "result_merged", "results_filtered", "results"):
                v = get_struct_field(mat, s, "time_in")
                if v is not None:
                    frame_in = int(round(float(np.asarray(v).squeeze()) * fs)); break
    if frame_in is None:
        raise RuntimeError(f"{os.path.basename(path)}: cannot find 'frame_in' or 'time_in'.")

    # C_final (denoised) and C_raw_final (raw)
    C_den, C_raw = None, None
    for s in ("results_final", "result_merged", "results_filtered", "results"):
        v = get_struct_field(mat, s, "C")
        if v is not None and np.asarray(v).ndim == 2:
            C_den = np.asarray(v, dtype=float); break       # 去噪痕迹
    for s in ("results_final", "result_merged", "results_filtered", "results"):
        v = get_struct_field(mat, s, "C_raw")
        if v is not None and np.asarray(v).ndim == 2:
            C_raw = np.asarray(v, dtype=float); break       # 原始痕迹（可缺）

    if C_den is None:
        if "traces_denoised" in keys:
            C_den = np.asarray(mat["traces_denoised"], dtype=float)
        elif "traces_zscored" in keys:
            C_den = np.asarray(mat["traces_zscored"], dtype=float)
    if C_den is None:
        raise RuntimeError(f"{os.path.basename(path)}: cannot find C_final (denoised traces).")

    # Ensure [cells, time]
    if C_den.shape[0] > C_den.shape[1]:
        C_den = C_den.T   # 若是 [time, cells] 则转置成 [cells, time]
    if C_raw is not None and C_raw.shape[0] > C_raw.shape[1]:
        C_raw = C_raw.T

    return {
        "mouse": parse_mouse_tag(path),  # 标签
        "fs": fs,                        # 采样率
        "frame_in": frame_in,            # 事件帧索引
        "C_den": C_den,                  # 去噪痕迹（必有）
        "C_raw": C_raw,                  # 原始痕迹（可空）
        "mat": mat,                      # 原始 mat 字典（后续取变量）
        "path": path,                    # 文件路径
    }


def match_tz_rows_in_C(tz, C_den):
    """将 traces_zscored(tz) 的每一行与 C_den 的某一行做一一匹配（最大余弦相似度）。
    - 先按行去均值/除标准差，再做 L2 归一化，计算相似度矩阵 S=A@B^T
    - 用匈牙利算法找到最优匹配，返回长度为 tz 行数的索引数组 tz_to_c
    - 返回 None 表示无法匹配（输入不合规等）
    """
    if not (isinstance(tz, np.ndarray) and tz.ndim == 2 and tz.shape[0] > 0):
        return None
    A = np.asarray(tz, dtype=float)
    B = np.asarray(C_den, dtype=float)
    nA, tA = A.shape; nB, tB = B.shape
    L = min(tA, tB)                 # 取两者最短时间长度对齐
    A = A[:, :L]; B = B[:, :L]

    def norm_rows(X):               # 行向标准化 + L2 归一
        X = X - X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True); sd[sd < 1e-9] = 1.0
        X = X / sd
        nrm = np.linalg.norm(X, axis=1, keepdims=True); nrm[nrm < 1e-9] = 1.0
        return X / nrm

    A = norm_rows(A); B = norm_rows(B)   # 归一化后计算相似度
    S = A @ B.T                          # 余弦相似度矩阵（因为都单位范数）
    r, c = linear_sum_assignment(-S)     # 最大化 S 等价于最小化 -S
    tz_to_c = np.full(nA, -1, dtype=int) # 预设为 -1（未匹配）
    tz_to_c[r] = c                       # 把行 r 对应的列 c 作为映射
    return tz_to_c


def normalize_cells_idx_to_mask(vals, n_cells):
    """把各种索引格式（可能 0/1 基）归一化为 C 空间的布尔掩码。
    - 支持浮点/整数数组；若 1..n 之间则视作 1 基并减 1
    - 越界则返回 None
    """
    a = np.array(vals).astype(float).reshape(-1)
    idx = np.rint(a).astype(int)        # 四舍五入转 int
    if idx.size == 0:
        return None
    if idx.min() >= 1 and idx.max() <= n_cells:
        idx = idx - 1                   # 转为 0 基
    if idx.min() < 0 or idx.max() >= n_cells:
        return None
    m = np.zeros(n_cells, dtype=bool); m[idx] = True
    return m


def make_cells_union_mask(cells_var, n_cells, tz=None, tz_to_c=None):
    """根据 mat['cells'] 构造 C 空间的布尔掩码；自动识别多种格式：
    - C 空间布尔/索引；或 TZ 空间布尔/索引（需要 tz_to_c 映射）
    返回 (mask, 描述字符串)；若无法识别返回 (None, 原因)
    """
    if cells_var is None:
        return None, "cells absent"
    a = np.asarray(cells_var)

    # 尝试把 object 数组拍平并转成数值
    if a.dtype == object:
        try:
            a = np.array([float(x) for x in a.reshape(-1)], dtype=float)
        except Exception:
            a = a.reshape(-1)
    else:
        a = a.reshape(-1)

    # Case A: C 空间布尔掩码
    if a.dtype == bool and a.size == n_cells:
        return a.copy(), "cells -> C bool"

    # Case B: C 空间数字索引
    if np.issubdtype(a.dtype, np.number) and a.size >= 1:
        m = normalize_cells_idx_to_mask(a, n_cells)
        if m is not None:
            return m, "cells -> C idx"

    # Case C: TZ 空间（需要 tz_to_c）
    if tz is not None and tz_to_c is not None and isinstance(tz, np.ndarray) and tz.ndim == 2:
        tz_n = tz.shape[0]
        if a.dtype == bool and a.size == tz_n:     # TZ 布尔
            chosen = np.where(a)[0]
            c_idx = tz_to_c[chosen]
            m = np.zeros(n_cells, dtype=bool); m[c_idx] = True
            return m, "cells(TZ bool) -> C"
        if np.issubdtype(a.dtype, np.number) and a.size >= 1:   # TZ 索引
            idx = np.rint(a).astype(int)
            if idx.min() >= 1 and idx.max() <= tz_n:
                idx = idx - 1
            if idx.min() >= 0 and idx.max() < tz_n:
                c_idx = tz_to_c[idx]
                m = np.zeros(n_cells, dtype=bool); m[c_idx] = True
                return m, "cells(TZ idx) -> C"

    return None, "cells unrecognized"  # 识别失败


def extract_window(traces, frame_in, fs, pre_s, post_s, zscore=True):
    """围绕 frame_in 截取 [ -pre_s, +post_s ] 的时间窗；
    - 可选择对窗口内按行 z-score（作图更直观）
    返回: (win, t, zero_idx)
    """
    pre_frames  = int(round(pre_s * fs))         # 前窗帧数
    post_frames = int(round(post_s * fs))        # 后窗帧数
    start = frame_in - pre_frames                # 左边界（含）
    end   = frame_in + post_frames               # 右边界（不含）
    if start < 0 or end > traces.shape[1]:       # 越界则返回 None
        return None, None, None
    win = traces[:, start:end]                   # 切片
    if zscore:
        win = row_zscore(win)                    # 按行标准化（用于热图）
    t = np.arange(-pre_frames, post_frames, dtype=float) / fs  # 相对时间轴（秒）
    return win, t, pre_frames                    # pre_frames 即 zero_idx（0 秒位置）


def get_auc_signal(C_den_sel, C_raw_sel, fs):
    """根据配置选择用于 AUC 的信号：raw / raw_smooth / denoised。
    - raw_smooth: 对 C_raw 做 1D 高斯滤波（σ=SMOOTH_SEC*Fs, 下限0.5帧）
    - C_raw 不可用时回退到 C_den
    """
    if AUC_SIGNAL == "raw" and C_raw_sel is not None:
        return C_raw_sel
    if AUC_SIGNAL == "raw_smooth" and C_raw_sel is not None:
        sigma = max(0.5, float(SMOOTH_SEC) * float(fs))
        return gaussian_filter1d(C_raw_sel, sigma=sigma, axis=1, mode="nearest")
    if AUC_SIGNAL == "denoised":
        return C_den_sel
    return C_raw_sel if C_raw_sel is not None else C_den_sel


def auc_pre_vs_post(win, zero_idx, fs, post_eval_s=300.0):
    """对每个细胞计算 AUC：比较后段[0, post_eval_s] 与 前段[-pre, 0)。
    - 使用秩转换 rankdata + U 统计的闭式表达，避免显式排序带来的并列问题
    返回: 每个细胞的 AUC 数组
    """
    n_cells, _ = win.shape
    post_eval_frames = int(round(post_eval_s * fs))
    a_post = win[:, zero_idx:zero_idx + post_eval_frames]  # 正样本：事件后
    b_pre  = win[:, :zero_idx]                             # 负样本：事件前
    aucs = np.zeros(n_cells, dtype=float)
    for i in range(n_cells):
        a = a_post[i]; b = b_pre[i]
        g = np.concatenate([a, b])               # 合并
        r = rankdata(g, method="average")       # 秩（平均处理并列）
        n1 = a.size; n0 = b.size
        r1_sum = np.sum(r[:n1])                  # 正样本秩和
        U = r1_sum - n1*(n1+1)/2.0              # U 统计（正样本）
        aucs[i] = U/(n0*n1)                      # AUC 与 U 的换算
    return aucs


def classify_cells(aucs, p_reu, p_iso):
    """根据阈值/显著性把细胞分为 Isolation / Reunion / Neutral。
    - 固定模式: AUC ≤ thr_lo → Iso；AUC ≥ thr_hi → Reu
    - 分位数模式: 用分位数切分两端
    - 若 USE_MWU_SIGNIF=True，则分别要求对应单侧检验 p<ALPHA
    返回: (idx_iso, idx_reu, idx_neu, (thr_lo, thr_hi))
    """
    thr_lo, thr_hi = AUC_ISOLATION_THR, AUC_REUNION_THR
    if AUC_THR_MODE == "quantile":
        thr_lo, thr_hi = np.quantile(aucs, [AUC_Q, 1.0 - AUC_Q])
    if USE_MWU_SIGNIF:
        idx_reu = np.where((aucs >= thr_hi) & (p_reu < ALPHA))[0]
        idx_iso = np.where((aucs <= thr_lo) & (p_iso < ALPHA))[0]
    else:
        idx_reu = np.where(aucs >= thr_hi)[0]
        idx_iso = np.where(aucs <= thr_lo)[0]
    idx_neu = np.setdiff1d(np.arange(aucs.size), np.concatenate([idx_iso, idx_reu]))
    return idx_iso, idx_reu, idx_neu, (thr_lo, thr_hi)


def sort_by_peak(iso_mat, reu_mat, t, zero_idx):
    """对两类细胞各自在更相关的时间片段内按峰出现时间排序：
    - Isolation：看事件前 300s 内的峰位置
    - Reunion：看事件后 600s 内的峰位置
    返回两个重排索引（升序：峰更早→排更前）
    """
    fs = 1.0/(t[1]-t[0])
    pre_start = zero_idx - int(round(300.0*fs))
    post_end  = zero_idx + int(round(600.0*fs))
    def order_by_peak(mat, start, end):
        if mat.size == 0: return np.arange(0)
        seg = mat[:, start:end]
        seg = np.where(np.isfinite(seg), seg, -np.inf)  # NaN 置为 -inf 避免干扰 argmax
        pk  = np.argmax(seg, axis=1)                    # 每行峰位置
        pk[np.all(~np.isfinite(seg), axis=1)] = 0       # 全 -inf 时兜底为 0
        return np.argsort(pk)                           # 按峰位置升序
    iso_ord = order_by_peak(iso_mat, pre_start, zero_idx) if iso_mat.size else np.arange(0)
    reu_ord = order_by_peak(reu_mat, zero_idx, post_end) if reu_mat.size else np.arange(0)
    return iso_ord, reu_ord

# ===== Plot =====
def _set_heatmap_axes_style(ax, t):
    """统一热图 x 轴范围/刻度/标签。"""
    ax.set_xlim(t[0], t[-1])
    ax.set_xticks([-300, 0, 300, 600])
    ax.set_xlabel("Time from reunion (s)")
    ax.set_ylabel("Neurons")


def _add_partner_marks(ax):
    """在坐标轴顶部添加 Partner in/out 的小三角与文字标注。"""
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)  # x 用数据坐标，y 用轴坐标
    ax.plot([0],   [1.02], marker="v", color="black", markersize=6, transform=trans, clip_on=False)
    ax.plot([300], [1.02], marker="v", color="black", markersize=6, transform=trans, clip_on=False)
    ax.text(0,   1.07, "Partner in",  transform=trans, ha="center", va="bottom", fontsize=10, color="black")
    ax.text(300, 1.07, "Partner out", transform=trans, ha="center", va="bottom", fontsize=10, color="black")


def _title_with_colored_sup(ax, left_text, sup_text, right_text, color):
    """绘制标题：主体黑色，中心点添加彩色上标（\n类似 MPN^{Isolation} neurons）。"""
    ax.set_title(left_text + right_text, fontsize=12, color="black")
    ax.text(0.5, 1.0, r"$^{\mathrm{" + sup_text + "}}$", color=color,
            transform=ax.transAxes, ha="center", va="bottom", fontsize=12)


def plot_panel(ax_heat, ax_avg, data, t, group_name, color):
    """绘制一个类别（Isolation/Reunion）的上热图+下均值曲线面板。
    - data: [n_cells, n_time]（已 z-score）
    - 返回 im（用于共享色条）
    """
    shown = np.clip(data, VMIN, VMAX) if data.size else np.zeros((1, len(t)))  # 空时放 1 行 0
    im = ax_heat.imshow(
        shown, vmin=VMIN, vmax=VMAX, cmap=CMAP, aspect="auto",
        extent=[t[0], t[-1], 1, max(1, shown.shape[0])], origin="lower",
    )
    ax_heat.axvline(0,   color="white", lw=1.8)         # 0 s 实线
    ax_heat.axvline(300, color="white", lw=1.2, ls="--")# 300 s 虚线
    _set_heatmap_axes_style(ax_heat, t)
    _add_partner_marks(ax_heat)
    _title_with_colored_sup(ax_heat, "MPN", group_name, " neurons", color)

    nrows = data.shape[0]
    if nrows > 0:
        mid = int(round(nrows/2.0/10)*10) if nrows >= 80 else max(1, nrows//2)  # 中间刻度更友好
        ax_heat.set_yticks([1, mid, nrows]); ax_heat.set_yticklabels([1, mid, nrows])

    mean = np.nanmean(data, axis=0) if data.size else np.zeros_like(t)  # 均值
    sem = (np.nanstd(data, axis=0, ddof=1) / np.sqrt(data.shape[0])) if data.shape[0] > 1 else np.zeros_like(mean)  # SEM
    ax_avg.plot(t, mean, color=color, lw=2)
    if data.shape[0] > 1:
        ax_avg.fill_between(t, mean-sem, mean+sem, color=color, alpha=0.20, lw=0)

    ax_avg.axvline(0,   color="black", lw=1.2)
    ax_avg.axvline(300, color="black", lw=1.0, ls="--")
    ax_avg.axhline(0, color="#888", lw=0.8, ls=":")
    ax_avg.set_xlim(t[0], t[-1]); ax_avg.set_xticks([-300, 0, 300, 600])
    ax_avg.set_xlabel("Time from reunion (s)")

    yl = ax_avg.get_ylim(); yr = yl[1]-yl[0]
    x2 = 600 - 20; x1 = x2 - 100; y0 = yl[0] + 0.10*yr  # 比例尺位置
    ax_avg.plot([x1, x2], [y0, y0], color="black", lw=2)                 # 100 s 水平标尺
    ax_avg.text((x1+x2)/2, y0 - 0.05*yr, "100 s", ha="center", va="top", fontsize=10, color="black")
    ax_avg.plot([x2+15, x2+15], [y0, y0+1.0], color="black", lw=2)       # 1σ 竖线标注
    ax_avg.text(x2+19, y0+1.0, r"$\\sigma$", ha="left", va="center", fontsize=11, color="black")

    return im

# ===== Main =====
def main():
    os.makedirs(OUT_DIR, exist_ok=True)  # 创建输出目录（若存在不报错）

    combined_iso, combined_reu = [], []   # 收集所有鼠的 Isolation/Reunion 窗口（z-score）
    summary_lines = []                    # 汇总文本的每鼠一行
    select_lines = []                     # 细胞选择策略记录

    total_cells_all_mice = 0              # 所有鼠的细胞总数累计
    total_selected_all_mice = 0           # 所有鼠的“选中”细胞数累计（分母，期望≈714）

    t_ref = None                          # 参考时间轴（第一只鼠）
    zero_idx_ref = None                   # 参考 0 秒索引

    for f in INPUT_FILES:                 # 遍历每个输入文件
        d = load_one_mouse(f)             # 读取一只鼠
        mouse = d["mouse"]; fs = d["fs"]; frame_in = d["frame_in"]
        C_den = d["C_den"]; C_raw = d["C_raw"]; mat = d["mat"]
        n_cells = C_den.shape[0]
        total_cells_all_mice += n_cells

        # Selection mask 从全 True 开始（先假定全部选中）
        sel_mask = np.ones(n_cells, dtype=bool)
        sel_src_notes = []                 # 记录选择来源说明

        # 若存在 TZ（traces_zscored），构建 TZ→C 的行匹配
        tz = mat.get("traces_zscored", None)
        tz_to_c = match_tz_rows_in_C(tz, C_den) if isinstance(tz, np.ndarray) and tz.ndim == 2 and tz.shape[0] > 0 else None

        # 1) 排除被 TZ 匹配到的 C 行（实现 973 - 129 - 131 ... 的扣除）
        if EXCLUDE_TZ_ROWS and tz_to_c is not None:
            sel_mask[tz_to_c] = False
            sel_src_notes.append(f"exclude tz={tz_to_c.size}")

        # 2) 将 mat['cells'] 指定的细胞集合并回（支持 C/TZ 空间布尔或索引）
        if UNION_CELLS_IDX and ("cells" in mat):
            m_cells, src = make_cells_union_mask(mat["cells"], n_cells, tz=tz, tz_to_c=tz_to_c)
            if m_cells is not None:
                before = sel_mask.sum()
                sel_mask = np.logical_or(sel_mask, m_cells)  # 并集
                plus = int(sel_mask.sum() - before)
                sel_src_notes.append(f"union cells +{plus} ({src})")
            else:
                sel_src_notes.append("cells present but not recognized")

        n_sel = int(sel_mask.sum())
        total_selected_all_mice += n_sel
        sel_src = "; ".join(sel_src_notes) if sel_src_notes else "keep_all"
        select_lines.append(f"{mouse}: total={n_cells}, selected={n_sel}, source={sel_src}")

        # 根据 sel_mask 子集化信号
        C_den_sel = C_den[sel_mask, :]
        C_raw_sel = C_raw[sel_mask, :] if C_raw is not None else None

        # 截窗：用于作图的是 z-score 后的去噪；用于 AUC 的是选定信号（不 z-score）
        win_den_z, t, zero_idx = extract_window(C_den_sel, frame_in, fs, PRE_SEC, POST_SEC, zscore=True)
        X_auc = get_auc_signal(C_den_sel, C_raw_sel, fs)
        win_auc, _, _ = extract_window(X_auc, frame_in, fs, PRE_SEC, POST_SEC, zscore=False)

        # 计算 AUC（后窗仅取 AUC_POST_SEC 秒）
        aucs = auc_pre_vs_post(win_auc, zero_idx, fs, post_eval_s=AUC_POST_SEC)

        # 可选显著性：对每个细胞做单侧 U 检验（后>前；前>后）
        if USE_MWU_SIGNIF:
            post_frames = int(round(AUC_POST_SEC * fs))
            a_post = win_auc[:, zero_idx:zero_idx + post_frames]
            b_pre  = win_auc[:, :zero_idx]
            p_reu = np.array([mannwhitneyu(a_post[i], b_pre[i], alternative="greater").pvalue for i in range(win_auc.shape[0])])
            p_iso = np.array([mannwhitneyu(b_pre[i], a_post[i], alternative="greater").pvalue for i in range(win_auc.shape[0])])
        else:
            p_reu = np.ones(win_auc.shape[0]); p_iso = np.ones(win_auc.shape[0])

        # 分类
        idx_iso, idx_reu, idx_neu, thr_pair = classify_cells(aucs, p_reu, p_iso)

        # 收集合并（用于总体热图/曲线）——注意使用 z-score 后的窗口
        if idx_iso.size: combined_iso.append(win_den_z[idx_iso])
        if idx_reu.size: combined_reu.append(win_den_z[idx_reu])

        # 汇总文本：每只鼠的计数与阈值记录
        summary_lines.append(
            f"{mouse}: cells(all)={n_cells}, cells(selected)={n_sel}, "
            f"iso={len(idx_iso)}, reunion={len(idx_reu)}, neutral={len(idx_neu)}, "
            f"thr=({thr_pair[0]:.3f},{thr_pair[1]:.3f})"
        )

        # 保存第一只鼠的时间轴作为参考，保证两类拼接时一致
        if t_ref is None:
            t_ref = t; zero_idx_ref = zero_idx

    # 若两类都空，说明没有可用数据（可能窗口越界或键缺失）
    if not combined_iso and not combined_reu:
        raise RuntimeError("No usable data; check window or keys.")

    iso_mat = np.vstack(combined_iso) if combined_iso else np.zeros((0, len(t_ref)))
    reu_mat = np.vstack(combined_reu) if combined_reu else np.zeros((0, len(t_ref)))

    # 按峰排序
    iso_ord, reu_ord = sort_by_peak(iso_mat, reu_mat, t_ref, zero_idx_ref)
    iso_sorted = iso_mat[iso_ord] if iso_mat.size else iso_mat
    reu_sorted = reu_mat[reu_ord] if reu_mat.size else reu_mat

    # 作图布局：2x2（上两块热图，下两块均值曲线）
    fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, height_ratios=[4.0, 1.3], hspace=0.30, wspace=0.30)
    ax_h_iso = plt.subplot(gs[0, 0]); ax_h_reu = plt.subplot(gs[0, 1])
    ax_a_iso = plt.subplot(gs[1, 0]); ax_a_reu = plt.subplot(gs[1, 1])

    im1 = plot_panel(ax_h_iso, ax_a_iso, iso_sorted, t_ref, "Isolation", GREEN)
    im2 = plot_panel(ax_h_reu, ax_a_reu, reu_sorted, t_ref, "Reunion",  PURPLE)

    # 在左上热图左边嵌入一个共享色条（使用 im2，二者 vmin/vmax 一致）
    cax = inset_axes(ax_h_iso, width="2.0%", height="65%", loc="center left",
                     bbox_to_anchor=(-0.12, 0.0, 1, 1), bbox_transform=ax_h_iso.transAxes, borderpad=0.0)
    cb  = fig.colorbar(im2, cax=cax)
    cb.set_label("Activity\n(Z score)")
    cb.set_ticks([0, 1, 2, 3])

    # 在均值轴上写入计数（分子=该类细胞数；分母=所有选中细胞数）
    ax_a_iso.text(-285, ax_a_iso.get_ylim()[0] + 0.85*(ax_a_iso.get_ylim()[1]-ax_a_iso.get_ylim()[0]),
                  f"{iso_sorted.shape[0]}/{total_selected_all_mice} neurons", fontsize=11, color="black")
    ax_a_reu.text(-285, ax_a_reu.get_ylim()[0] + 0.85*(ax_a_reu.get_ylim()[1]-ax_a_reu.get_ylim()[0]),
                  f"{reu_sorted.shape[0]}/{total_selected_all_mice} neurons", fontsize=11, color="black")

    out_png = os.path.join(OUT_DIR, "fig2c_combined_paperstyle.png")  # 输出图路径
    plt.savefig(out_png, dpi=DPI); plt.close(fig)                       # 保存并关闭图

    # 写入汇总文本文件
    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as fw:
        fw.write("\n".join(summary_lines) + "\n\n")
        fw.write("Selection summary:\n")
        fw.write("\n".join(select_lines) + "\n\n")
        fw.write(f"Total cells (all mice): {total_cells_all_mice}\n")
        fw.write(f"Total cells (selected): {total_selected_all_mice}\n")
        fw.write(f"Total Isolation neurons: {iso_sorted.shape[0]}\n")
        fw.write(f"Total Reunion neurons: {reu_sorted.shape[0]}\n")

    print(f"Saved: {out_png}")                                        # 控制台提示图已保存
    print(f"Summary -> {os.path.join(OUT_DIR, 'summary.txt')}")        # 控制台提示汇总路径


if __name__ == "__main__":   # 作为脚本运行时执行 main()
    main()

```

