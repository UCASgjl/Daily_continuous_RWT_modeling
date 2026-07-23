import pandas as pd
import os
from scipy.optimize import curve_fit
import numpy as np


# ==================================  改进  ==========================================
# ========================================  添加rmse  =================================
import pandas as pd
import os
from scipy.optimize import curve_fit
import numpy as np
from sklearn.metrics import mean_squared_error


# 1. 逻辑回归模型（输出NumPy数组，避免形状问题）
def logistic_model(T_a, a, b, c, d):
    """逻辑回归模型：输入输出均为NumPy数组"""
    return a + (b - a) / (1 + np.exp(c * (d - T_a)))


# 2. 加权滞后气温计算（核心优化：让f可识别，输出NumPy数组）
def calculate_weighted_Ta(AT_mean_array, lag, f):
    """
    计算加权滞后气温（修复版）
    输入：AT_mean_array（NumPy数组，无索引）、lag（滞后天数）、f（衰减系数）
    输出：weighted_Ta（NumPy数组，与输入同长度）
    """
    n = len(AT_mean_array)
    weighted_Ta = np.zeros(n)  # 初始化输出数组（固定形状，避免不均匀）

    for i in range(n):
        # 确定窗口范围：i-lag 到 i（确保不越界）
        start_idx = max(0, i - lag)
        window = AT_mean_array[start_idx:i + 1]  # 窗口内的气温数据（NumPy数组）
        window_len = len(window)

        # 核心优化：让f可识别——权重随滞后天数线性增加j（而非从0开始）
        # j=1（当前时刻）, 2（滞后1天）, ..., window_len（滞后lag天）
        j = np.arange(1, window_len + 1)
        # 权重计算：当前时刻权重最大，滞后越久权重越小（f越大，衰减越快）
        wt = np.exp(-f * (j - 1))  # j-1确保当前时刻（j=1）权重为1
        wt_sum = wt.sum()

        # 归一化权重（避免除以零）
        if wt_sum < 1e-10:  # 防止数值下溢
            w = np.ones(window_len) / window_len
        else:
            w = wt / wt_sum

        # 计算加权气温（标量）
        weighted_Ta[i] = np.dot(w, window)

    return weighted_Ta  # 输出为NumPy数组（形状均匀）


# 3. 完整模型函数（适配curve_fit要求：输入输出均为NumPy数组）
def model_for_fitting(AT_mean_array, a, b, c, d, f, lag):
    """
    用于拟合的完整模型（修复版）
    注意：curve_fit要求“自变量”必须是第一个参数，“待拟合参数”在后
    这里通过lambda函数将lag固定，仅让a/b/c/d/f作为待拟合参数
    """
    weighted_Ta = calculate_weighted_Ta(AT_mean_array, lag, f)
    return logistic_model(weighted_Ta, a, b, c, d)


# 4. 数据准备（输出NumPy数组，避免索引问题）
def prepare_data(df, comid):
    """准备数据（修复版）：输出用于拟合的NumPy数组"""
    # 基础清洗：移除缺失值和无穷值
    df_cleaned = df.dropna(subset=['AT_mean', 'DOY', 'temp']).copy()
    df_cleaned = df_cleaned[
        (np.isfinite(df_cleaned['AT_mean'])) &
        (np.isfinite(df_cleaned['temp']))
        ]

    # 异常值处理（3σ原则）
    # 水温异常值
    temp_mean, temp_std = df_cleaned['temp'].mean(), df_cleaned['temp'].std()
    temp_outlier = (df_cleaned['temp'] < temp_mean - 3 * temp_std) | (df_cleaned['temp'] > temp_mean + 3 * temp_std)
    # 气温异常值
    at_mean, at_std = df_cleaned['AT_mean'].mean(), df_cleaned['AT_mean'].std()
    at_outlier = (df_cleaned['AT_mean'] < at_mean - 3 * at_std) | (df_cleaned['AT_mean'] > at_mean + 3 * at_std)
    # 移除异常值
    df_cleaned = df_cleaned[~temp_outlier & ~at_outlier]

    # 统计信息
    initial_count = df.dropna(subset=['temp']).shape[0]
    cleaned_count = df_cleaned.shape[0]
    print(f"[{comid}] 数据清洗：原始{initial_count}个有效水温 → 保留{cleaned_count}个")

    # 输出用于拟合的NumPy数组（无索引，避免形状问题）
    AT_fit = df_cleaned['AT_mean'].values  # 自变量：气温
    temp_fit = df_cleaned['temp'].values  # 因变量：水温
    df_cleaned['used_for_fitting'] = True  # 标记用于拟合的行

    return df_cleaned, AT_fit, temp_fit


# 5. 模型拟合（修复输入格式，优化参数约束）
def fit_model(AT_fit, temp_fit, lag, comid):
    """拟合模型（修复版）：处理NumPy数组，优化f参数识别"""
    n = len(AT_fit)
    if n < 15:
        print(f"[{comid}] 拟合失败：有效数据仅{n}个（需≥15）")
        return None

    # 初始参数猜测（基于数据特征，让f有初始影响力）
    temp_min, temp_max = temp_fit.min(), temp_fit.max()
    at_mean = AT_fit.mean()
    initial_guess = [
        temp_min * 0.9,  # a：水温最小值（略小，避免卡边界）
        temp_max * 1.1,  # b：水温最大值（略大，避免卡边界）
        0.2,  # c：曲线斜率（逻辑回归常用范围）
        at_mean,  # d：中点（与气温均值对齐，合理初始值）
        0.5  # f：衰减系数（初始值0.5，确保有衰减效应）
    ]

    # 参数边界（宽松且合理，避免限制f的优化）
    bounds = (
        [temp_min * 0.5 if temp_min != 0 else -5,  # a下限
         temp_min * 1.0,  # b下限（不低于实际最小值）
         0.01,  # c下限（避免斜率为0）
         AT_fit.min(),  # d下限（不低于最低气温）
         0.01],  # f下限（避免f=0，确保可识别）
        [temp_max * 0.6,  # a上限（不超过最大值的60%）
         temp_max * 1.5,  # b上限（不超过最大值的1.5倍）
         1.0,  # c上限（避免曲线过陡）
         AT_fit.max(),  # d上限（不超过最高气温）
         5.0]  # f上限（衰减不过快）
    )

    try:
        # 关键：通过lambda函数固定lag，仅拟合a/b/c/d/f
        # 输入AT_fit（NumPy数组），输出预测值（NumPy数组）
        popt, pcov = curve_fit(
            lambda x, a, b, c, d, f: model_for_fitting(x, a, b, c, d, f, lag),
            xdata=AT_fit,  # 自变量：气温（NumPy数组）
            ydata=temp_fit,  # 因变量：水温（NumPy数组）
            p0=initial_guess,  # 初始参数
            bounds=bounds,  # 参数边界
            maxfev=200000,  # 增加迭代次数，确保收敛
            method="dogbox",  # 适合有边界约束的优化
            ftol=1e-7,  # 收敛阈值（平衡精度与速度）
            xtol=1e-7
        )

        # 诊断f参数敏感性（验证f是否可识别）
        a, b, c, d, f_fit = popt
        # 计算f变化10%对预测值的影响
        f_varied = f_fit * 1.1
        pred_original = model_for_fitting(AT_fit, a, b, c, d, f_fit, lag)
        pred_varied = model_for_fitting(AT_fit, a, b, c, d, f_varied, lag)
        # 预测值变化率（绝对值平均）
        pred_change = np.mean(np.abs((pred_varied - pred_original) / (pred_original + 1e-10))) * 100
        print(f"[{comid}] f敏感性：f变化10% → 预测值变化{pred_change:.2f}%")
        if pred_change < 0.1:
            print(f"[{comid}] 警告：f敏感性较低（<0.1%），建议检查滞后天数lag")

        # 诊断权重分布（验证f的实际效果）
        sample_idx = min(100, n - 1)  # 取第100个数据点的窗口权重
        start_idx = max(0, sample_idx - lag)
        window = AT_fit[start_idx:sample_idx + 1]
        j = np.arange(1, len(window) + 1)
        wt = np.exp(-f_fit * (j - 1))
        w = wt / wt.sum()
        print(f"[{comid}] 样本窗口（第{sample_idx}天）权重：{w.round(4)}")

        # 边界检查（提示参数是否接近边界）
        param_names = ['a', 'b', 'c', 'd', 'f']
        for i, (param, lower, upper) in enumerate(zip(popt, bounds[0], bounds[1])):
            if param - lower < 1e-4:
                print(f"[{comid}] 警告：参数{param_names[i]}={param:.4f}接近下限{lower:.4f}")
            if upper - param < 1e-4:
                print(f"[{comid}] 警告：参数{param_names[i]}={param:.4f}接近上限{upper:.4f}")

        return popt

    except Exception as e:
        print(f"[{comid}] 拟合失败：{str(e)[:100]}")  # 打印错误前100字符（避免过长）
        return None


# 6. 主函数（完整流程：读取→清洗→拟合→保存）
def process_data(input_folder, output_folder, lag=3):
    os.makedirs(output_folder, exist_ok=True)
    catchment_params = []  # 存储所有流域的参数

    # 遍历所有Excel文件
    for file_name in os.listdir(input_folder):
        if not file_name.endswith('.xlsx'):
            continue
        comid = file_name.split('.')[0]
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        try:
            print(f"\n===== 开始处理：{file_name}（COMID：{comid}）=====")
            # 读取原始数据
            df = pd.read_excel(input_path)
            # 检查必要列
            required_cols = ['temp', 'AT_mean', 'DOY']
            if not all(col in df.columns for col in required_cols):
                print(f"[{comid}] 跳过：缺少必要列（需{required_cols}）")
                continue

            # 准备数据（输出清洗后的数据、气温数组、水温数组）
            df_cleaned, AT_fit, temp_fit = prepare_data(df, comid)
            if len(AT_fit) < 15:
                print(f"[{comid}] 跳过：有效数据不足")
                continue

            # 拟合模型
            params = fit_model(AT_fit, temp_fit, lag, comid)
            if params is None:
                print(f"[{comid}] 跳过：模型拟合失败")
                continue

            # 计算所有日期的预测值（包括未用于拟合的行）
            a, b, c, d, f_fit = params
            # 对全量气温数据计算加权滞后气温（注意：用清洗后的全量数据）
            AT_full = df_cleaned['AT_mean'].values
            df_cleaned['weighted_Ta'] = calculate_weighted_Ta(AT_full, lag, f_fit)
            # 计算预测水温
            df_cleaned['predicted_temp'] = logistic_model(df_cleaned['weighted_Ta'].values, a, b, c, d)

            # 合并回原始数据（保留所有原始行）
            df['used_for_fitting'] = False  # 初始标记为未使用
            df['weighted_Ta'] = np.nan  # 初始为NaN
            df['predicted_temp'] = np.nan  # 初始为NaN
            # 用DOY匹配，将结果写入原始数据
            doy_mapping = df_cleaned.set_index('DOY')[['used_for_fitting', 'weighted_Ta', 'predicted_temp']]
            df.update(doy_mapping)

            # 计算并添加RMSE列 - 只针对有原始水温数据的行
            valid_mask = df['temp'].notna() & df['predicted_temp'].notna()
            if valid_mask.sum() > 0:
                # 计算整体RMSE值
                rmse_value = np.sqrt(mean_squared_error(
                    df.loc[valid_mask, 'temp'],
                    df.loc[valid_mask, 'predicted_temp']
                ))
                print(f"[{comid}] 预测水温与原始水温的RMSE: {rmse_value:.4f}")

                # 添加RMSE列，所有行都显示相同的RMSE值
                df['RMSE'] = rmse_value
            else:
                df['RMSE'] = np.nan
                print(f"[{comid}] 没有足够的有效数据计算RMSE")

            # 保存结果
            df.to_excel(output_path, index=False)
            print(f"\n[{comid}] 处理完成！最终参数：")
            print(f"  a={a:.4f}, b={b:.4f}, c={c:.4f}, d={d:.4f}, f={f_fit:.4f}")

            # 记录参数（用于后续分析）
            catchment_params.append({
                'COMID': comid,
                'a': a, 'b': b, 'c': c, 'd': d, 'f': f_fit,
                'lag': lag,
                'valid_data_count': len(AT_fit),
                'total_days': len(df_cleaned),
                'RMSE': rmse_value if valid_mask.sum() > 0 else np.nan
            })

        except Exception as e:
            print(f"[{comid}] 处理出错：{str(e)[:100]}")
            continue

    # 保存所有流域的参数汇总
    if catchment_params:
        params_df = pd.DataFrame(catchment_params)
        params_path = os.path.join(output_folder, 'catchment_params_final.xlsx')
        params_df.to_excel(params_path, index=False)
        print(f"\n===== 所有流域参数已保存到：{params_path} =====")
        # 打印参数统计（验证f是否不同）
        f_values = [p['f'] for p in catchment_params if not np.isnan(p['f'])]
        rmse_values = [p['RMSE'] for p in catchment_params if not np.isnan(p['RMSE'])]

        print(
            f"===== f参数统计：均值={np.mean(f_values):.4f}, 标准差={np.std(f_values):.4f}, 范围=[{min(f_values):.4f}, {max(f_values):.4f}] =====")
        print(
            f"===== RMSE统计：均值={np.mean(rmse_values):.4f}, 标准差={np.std(rmse_values):.4f}, 范围=[{min(rmse_values):.4f}, {max(rmse_values):.4f}] =====")
    else:
        print("\n===== 无成功拟合的流域，未生成参数文件 =====")


# 执行主函数
if __name__ == "__main__":
    # 文件夹路径（根据实际情况修改）
    INPUT_FOLDER = r"E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\Statistical regression\\COMID1"
    OUTPUT_FOLDER = r"E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\Statistical regression\\output1"
    LAG_DAYS = 2  # 滞后天数（可尝试2或4天，观察f敏感性变化）

    # 启动处理
    # process_data(INPUT_FOLDER, OUTPUT_FOLDER, lag=LAG_DAYS)



# ===================================   多个备选滞后天数  ===============================================
import pandas as pd
import os
from scipy.optimize import curve_fit
import numpy as np
from sklearn.metrics import mean_squared_error


# 1. 逻辑回归模型
def logistic_model(T_a, a, b, c, d):
    return a + (b - a) / (1 + np.exp(c * (d - T_a)))


# 2. 加权滞后气温计算（弱化过度衰减，增强f的作用）
def calculate_weighted_Ta(AT_mean_series, lag, f):
    AT_clean = AT_mean_series.dropna()
    if AT_clean.empty:
        return pd.Series(np.nan, index=AT_mean_series.index)

    AT_array = AT_clean.values
    original_indices = AT_clean.index.tolist()
    n = len(AT_array)
    weighted_Ta_array = np.zeros(n)

    for i in range(n):
        if lag == 0:
            window = AT_array[i:i + 1]
        else:
            start_pos = max(0, i - lag)
            window = AT_array[start_pos:i + 1]

        window_len = len(window)
        # 弱化衰减加速，让f的影响更显著：j为滞后阶数（0=当前，1=滞后1天...）
        j = np.arange(window_len)  # 恢复为线性滞后阶数
        wt = np.exp(-f * j)  # 基础指数衰减，f的作用更直接
        wt_sum = wt.sum()

        if wt_sum < 1e-10:
            w = np.ones(window_len) / window_len
        else:
            w = wt / wt_sum

        weighted_Ta_array[i] = np.dot(w, window)

    weighted_Ta = pd.Series(index=AT_mean_series.index, dtype=float)
    weighted_Ta.loc[original_indices] = weighted_Ta_array
    return weighted_Ta


# 3. 完整模型函数
def model_for_fitting(AT_mean_series, a, b, c, d, f, lag):
    weighted_Ta = calculate_weighted_Ta(AT_mean_series, lag, f)
    predicted_temp = pd.Series(np.nan, index=AT_mean_series.index)
    valid_mask = ~pd.isna(weighted_Ta)
    predicted_temp[valid_mask] = logistic_model(weighted_Ta[valid_mask].values, a, b, c, d)
    return predicted_temp


# 4. 数据准备
def prepare_data(df, comid):
    df_full = df.copy()
    df_full['is_fit_valid'] = (
            df_full['temp'].notna() &
            df_full['AT_mean'].notna() &
            np.isfinite(df_full['temp']) &
            np.isfinite(df_full['AT_mean'])
    )

    if df_full['is_fit_valid'].sum() > 0:
        temp_valid = df_full.loc[df_full['is_fit_valid'], 'temp']
        temp_mean, temp_std = temp_valid.mean(), temp_valid.std()
        temp_outlier = (temp_valid < temp_mean - 3 * temp_std) | (temp_valid > temp_mean + 3 * temp_std)

        at_valid = df_full.loc[df_full['is_fit_valid'], 'AT_mean']
        at_mean, at_std = at_valid.mean(), at_valid.std()
        at_outlier = (at_valid < at_mean - 3 * at_std) | (at_valid > at_mean + 3 * at_std)

        df_full.loc[df_full['is_fit_valid'], 'is_fit_valid'] = ~temp_outlier & ~at_outlier

    initial_fit_count = df_full['temp'].notna().sum()
    final_fit_count = df_full['is_fit_valid'].sum()
    print(f"[{comid}] 拟合数据：原始{initial_fit_count} → 保留{final_fit_count}")

    fit_data = df_full[df_full['is_fit_valid']]
    AT_fit = fit_data['AT_mean'].values
    temp_fit = fit_data['temp'].values

    return df_full, AT_fit, temp_fit


# 5. 单滞后值拟合（调整f的边界）
def fit_single_lag(AT_fit, temp_fit, lag, comid):
    n = len(AT_fit)
    if n < 15:
        return None, np.inf

    temp_min, temp_max = temp_fit.min(), temp_fit.max()
    at_mean = AT_fit.mean()
    initial_guess = [
        temp_min * 0.9,
        temp_max * 1.1,
        0.2,
        at_mean,
        0.2  # 提高f的初始猜测，远离下限
    ]

    # 调整f的边界：降低下限，扩大优化空间
    bounds = (
        [temp_min * 0.5 if temp_min != 0 else -5,
         temp_min * 1.0,
         0.01,
         AT_fit.min(),
         0.01],  # f下限从0.05降回0.01
        [temp_max * 0.6,
         temp_max * 1.5,
         1.0,
         AT_fit.max(),
         2.0]  # 适当降低上限，避免过度衰减
    )

    try:
        popt, _ = curve_fit(
            lambda x, a, b, c, d, f: logistic_model(
                calculate_weighted_Ta(pd.Series(x), lag, f).values,
                a, b, c, d
            ),
            xdata=AT_fit,
            ydata=temp_fit,
            p0=initial_guess,
            bounds=bounds,
            maxfev=200000,
            method="dogbox"
        )

        # 计算f敏感性（判断f是否可识别）
        a, b, c, d, f_fit = popt
        f_varied = f_fit * 1.1  # 增加10%
        pred_original = logistic_model(
            calculate_weighted_Ta(pd.Series(AT_fit), lag, f_fit).values,
            a, b, c, d
        )
        pred_varied = logistic_model(
            calculate_weighted_Ta(pd.Series(AT_fit), lag, f_varied).values,
            a, b, c, d
        )
        pred_change = np.mean(np.abs((pred_varied - pred_original) / (pred_original + 1e-10))) * 100

        rmse = np.sqrt(mean_squared_error(temp_fit, pred_original))
        return popt, rmse, pred_change  # 返回f敏感性

    except Exception as e:
        print(f"[{comid}] lag={lag} 拟合失败：{str(e)[:50]}")
        return None, np.inf, 0.0


# 6. 带f敏感性约束的滞后优选（核心改进）
def select_best_lag(AT_fit, temp_fit, comid, candidate_lags=[0, 1, 2, 3, 4, 5, 6]):
    print(f"\n[{comid}] 滞后优选（候选：{candidate_lags}）")
    lag_results = []

    # 计算所有候选lag的拟合结果（包含f敏感性）
    for lag in candidate_lags:
        popt, rmse, f_sensitivity = fit_single_lag(AT_fit, temp_fit, lag, comid)
        if popt is not None:
            lag_results.append((lag, popt, rmse, f_sensitivity))

    if not lag_results:
        print(f"[{comid}] 所有lag拟合失败")
        return None, None, np.inf

    # 按RMSE升序排序
    lag_results.sort(key=lambda x: x[2])
    lag_rmse = [(lr[0], lr[2]) for lr in lag_results]

    # 调整边际效益阈值为0.5%（适应数据特性）
    valid_lags = []
    min_rmse = lag_rmse[0][1]
    for i, (lag, rmse) in enumerate(lag_rmse):
        if i == 0:
            valid_lags.append((lag, rmse, 100.0))
            continue

        prev_rmse = lag_rmse[i - 1][1]
        if prev_rmse == 0:
            下降率 = 0.0
        else:
            下降率 = (prev_rmse - rmse) / prev_rmse * 100

        # 降低阈值到0.5%，接受更小的RMSE改善
        if 下降率 >= 0.5:
            valid_lags.append((lag, rmse, 下降率))
        else:
            print(f"[{comid}] lag={lag} 边际效益不足（下降率{下降率:.2f}% < 0.5%），排除")

    if not valid_lags:
        return None, None, np.inf

    # 从有效lag中筛选f敏感性足够的（>0.3%）
    sensitive_lags = []
    for lag, rmse, _ in valid_lags:
        # 找到对应的f敏感性
        lr = next(lr for lr in lag_results if lr[0] == lag)
        f_sensitivity = lr[3]
        if f_sensitivity > 0.3:  # f变化10%，预测值变化>0.3%才算敏感
            sensitive_lags.append((lag, rmse, f_sensitivity))
        else:
            print(f"[{comid}] lag={lag} f敏感性不足（{f_sensitivity:.2f}% < 0.3%），排除")

    # 若没有敏感lag，退而求其次选择RMSE最小的
    if not sensitive_lags:
        print(f"[{comid}] 所有有效lag的f敏感性不足，选择RMSE最小的lag")
        sensitive_lags = [(lag, rmse, 0.0) for lag, rmse, _ in valid_lags]

    # 优先选择短lag（物理合理性）
    sensitive_lags.sort(key=lambda x: x[0])
    best_lag = sensitive_lags[0][0]
    best_popt = next(lr[1] for lr in lag_results if lr[0] == best_lag)
    best_rmse = sensitive_lags[0][1]
    best_f_sensitivity = sensitive_lags[0][2]

    f_best = best_popt[4]
    print(f"[{comid}] 最优lag：{best_lag}（RMSE：{best_rmse:.4f}，f={f_best:.4f}，f敏感性：{best_f_sensitivity:.2f}%）")
    return best_lag, best_popt, best_rmse


# 7. 主函数
def process_data(input_folder, output_folder, candidate_lags=[0, 1, 2, 3, 4, 5, 6]):
    os.makedirs(output_folder, exist_ok=True)
    catchment_params = []

    for file_name in os.listdir(input_folder):
        if not file_name.endswith('.xlsx'):
            continue
        comid = file_name.split('.')[0]
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        try:
            print(f"\n" + "=" * 50)
            print(f"处理流域：{file_name}（COMID：{comid}）")
            print("=" * 50)

            df = pd.read_excel(input_path)
            required_cols = ['temp', 'AT_mean', 'DOY']
            if not all(col in df.columns for col in required_cols):
                print(f"[{comid}] 跳过：缺少列{required_cols}")
                continue

            df_full, AT_fit, temp_fit = prepare_data(df, comid)
            if len(AT_fit) < 15:
                print(f"[{comid}] 跳过：有效数据不足")
                continue

            best_lag, best_popt, best_rmse = select_best_lag(AT_fit, temp_fit, comid, candidate_lags)
            if best_popt is None:
                print(f"[{comid}] 跳过：无有效参数")
                continue

            a, b, c, d, f_best = best_popt
            df_full['predicted_temp'] = model_for_fitting(
                AT_mean_series=df_full['AT_mean'],
                a=a, b=b, c=c, d=d, f=f_best, lag=best_lag
            )

            df_full['best_lag'] = best_lag
            df_full['fit_RMSE'] = best_rmse
            df_full['f_value'] = f_best  # 记录f值
            df_full['used_for_fitting'] = df_full['is_fit_valid']
            df_full['weighted_Ta'] = calculate_weighted_Ta(df_full['AT_mean'], best_lag, f_best)
            df_output = df_full.drop(columns=['is_fit_valid'])

            df_output.to_excel(output_path, index=False)
            print(f"[{comid}] 结果保存：{output_path}（{len(df_output)}行）")

            catchment_params.append({
                'COMID': comid,
                'best_lag': best_lag,
                'a': a, 'b': b, 'c': c, 'd': d, 'f': f_best,
                'fit_RMSE': best_rmse,
                'valid_fit_rows': len(AT_fit),
                'total_days': len(df_output)
            })

        except Exception as e:
            print(f"[{comid}] 处理出错：{str(e)}")
            continue

    if catchment_params:
        params_df = pd.DataFrame(catchment_params)
        params_path = os.path.join(output_folder, 'catchment_best_params_final.xlsx')
        params_df.to_excel(params_path, index=False)
        print(f"\n参数汇总：{params_path}")
        print(f"最优lag分布：{params_df['best_lag'].value_counts().sort_index().to_dict()}")
        print(f"f参数范围：[{params_df['f'].min():.4f}, {params_df['f'].max():.4f}]，均值：{params_df['f'].mean():.4f}")
    else:
        print("\n无成功处理的流域")


if __name__ == "__main__":
    INPUT_FOLDER = r"E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\Statistical regression\\COMID1"
    OUTPUT_FOLDER = r"E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\Statistical regression\\output_full_days"
    CANDIDATE_LAGS = [0, 1, 2, 3, 4, 5, 6]

    # process_data(INPUT_FOLDER, OUTPUT_FOLDER, candidate_lags=CANDIDATE_LAGS)






# =========================== yiqian  ===============================================================================
import pandas as pd
import os
from scipy.optimize import curve_fit
import numpy as np
from sklearn.metrics import mean_squared_error


# 1. 逻辑回归模型（输出NumPy数组）
def logistic_model(T_a, a, b, c, d):
    return a + (b - a) / (1 + np.exp(c * (d - T_a)))


# 2. 加权滞后气温计算（支持全量日期，适配不同lag）
def calculate_weighted_Ta(AT_mean_series, lag, f):
    """
    支持全量日期的加权滞后气温计算
    输入：AT_mean_series（Pandas Series，含所有日期的气温，允许NaN但会跳过）
    输出：weighted_Ta（Pandas Series，与输入同索引，仅对非NaN气温计算）
    """
    # 复制输入序列，避免修改原始数据
    AT_clean = AT_mean_series.dropna()  # 先移除气温为NaN的行（无法计算加权值）
    if AT_clean.empty:
        return pd.Series(np.nan, index=AT_mean_series.index)  # 无有效气温时全为NaN

    # 转换为NumPy数组（用于计算）和索引映射（用于后续对齐）
    AT_array = AT_clean.values
    original_indices = AT_clean.index.tolist()
    n = len(AT_array)
    weighted_Ta_array = np.zeros(n)

    # 计算每个有效气温日期的加权值
    for i in range(n):
        if lag == 0:
            # 滞后0天：仅用当天气温
            window = AT_array[i:i+1]
        else:
            # 滞后≥1天：用当天+前lag天（基于原始索引的位置，确保时间顺序正确）
            # 找到当前日期在原始序列中的位置，向前追溯lag天
            current_original_idx = original_indices[i]
            # 向前取lag天的索引（注意：原始数据可能有缺失，需用有效气温的索引）
            # 找到当前有效气温在AT_clean中的前lag个有效索引
            start_pos = max(0, i - lag)
            window = AT_array[start_pos:i+1]

        window_len = len(window)
        # 权重计算：当前时刻（最后1个元素）权重最大
        j = np.arange(window_len)
        wt = np.exp(-f * j[::-1])  # 反转j，当前时刻j=0（权重=1）
        wt_sum = wt.sum()

        if wt_sum < 1e-10:
            w = np.ones(window_len) / window_len
        else:
            w = wt / wt_sum

        weighted_Ta_array[i] = np.dot(w, window)

    # 构建全量日期的加权值Series（非有效气温日期为NaN）
    weighted_Ta = pd.Series(
        index=AT_mean_series.index,
        dtype=float
    )
    # 将计算好的加权值赋值到对应索引（确保全量日期覆盖）
    weighted_Ta.loc[original_indices] = weighted_Ta_array

    return weighted_Ta


# 3. 完整模型函数（适配全量预测）
def model_for_fitting(AT_mean_series, a, b, c, d, f, lag):
    """
    对全量日期生成预测值
    输入：AT_mean_series（全量日期的气温Series）
    输出：Logist_RWT（全量日期的预测水温Series，非有效气温日期为NaN）
    """
    # 先计算全量加权气温
    weighted_Ta = calculate_weighted_Ta(AT_mean_series, lag, f)
    # 仅对有加权值的日期计算预测水温
    Logist_RWT = pd.Series(np.nan, index=AT_mean_series.index)
    valid_mask = ~pd.isna(weighted_Ta)  # 筛选有加权值的日期
    Logist_RWT[valid_mask] = logistic_model(weighted_Ta[valid_mask].values, a, b, c, d)

    return Logist_RWT


# 4. 数据准备（保留全量日期，仅筛选拟合用数据）
def prepare_data(df, comid):
    """
    保留原始数据的所有日期，仅筛选“有水温+气温有效”的行用于拟合
    返回：
        - df_full：保留全量日期的DataFrame（未删除任何行）
        - AT_fit：拟合用的气温数组（仅有效行）
        - temp_fit：拟合用的水温数组（仅有效行）
    """
    # 复制原始数据，保留所有日期
    df_full = df.copy()

    # 标记“有效拟合行”：水温非NaN+气温非NaN+无无穷值
    df_full['is_fit_valid'] = (
        df_full['temp'].notna() &
        df_full['AT_mean'].notna() &
        np.isfinite(df_full['temp']) &
        np.isfinite(df_full['AT_mean'])
    )

    # 对有效拟合行进行异常值处理（3σ原则）
    if df_full['is_fit_valid'].sum() > 0:
        # 水温异常值
        temp_valid = df_full.loc[df_full['is_fit_valid'], 'temp']
        temp_mean, temp_std = temp_valid.mean(), temp_valid.std()
        temp_outlier = (temp_valid < temp_mean - 3*temp_std) | (temp_valid > temp_mean + 3*temp_std)
        # 气温异常值
        at_valid = df_full.loc[df_full['is_fit_valid'], 'AT_mean']
        at_mean, at_std = at_valid.mean(), at_valid.std()
        at_outlier = (at_valid < at_mean - 3*at_std) | (at_valid > at_mean + 3*at_std)
        # 更新有效拟合行：排除异常值
        df_full.loc[df_full['is_fit_valid'], 'is_fit_valid'] = ~temp_outlier & ~at_outlier

    # 统计清洗结果
    initial_fit_count = df_full['temp'].notna().sum()  # 原始有水温的行数
    final_fit_count = df_full['is_fit_valid'].sum()    # 最终用于拟合的行数
    print(f"[{comid}] 拟合数据筛选：原始{initial_fit_count}个有水温行 → 保留{final_fit_count}个有效拟合行")

    # 提取拟合用数组（仅有效行）
    fit_data = df_full[df_full['is_fit_valid']]
    AT_fit = fit_data['AT_mean'].values
    temp_fit = fit_data['temp'].values

    return df_full, AT_fit, temp_fit


# 5. 单滞后值拟合（返回参数+RMSE）
def fit_single_lag(AT_fit, temp_fit, lag, comid):
    n = len(AT_fit)
    if n < 15:
        return None, np.inf

    # 初始参数猜测
    temp_min, temp_max = temp_fit.min(), temp_fit.max()
    at_mean = AT_fit.mean()
    initial_guess = [
        temp_min * 0.9,
        temp_max * 1.1,
        0.2,
        at_mean,
        0.3
    ]

    # 参数边界
    bounds = (
        [temp_min * 0.5 if temp_min != 0 else -5,
         temp_min * 1.0,
         0.01,
         AT_fit.min(),
         0.01],
        [temp_max * 0.6,
         temp_max * 1.5,
         1.0,
         AT_fit.max(),
         5.0]
    )

    try:
        # 拟合（仅用有效拟合行的数据）
        popt, _ = curve_fit(
            lambda x, a, b, c, d, f: logistic_model(
                calculate_weighted_Ta(pd.Series(x), lag, f).values,  # 拟合时也用相同的加权逻辑
                a, b, c, d
            ),
            xdata=AT_fit,
            ydata=temp_fit,
            p0=initial_guess,
            bounds=bounds,
            maxfev=200000,
            method="dogbox",
            ftol=1e-7,
            xtol=1e-7
        )

        # 计算拟合RMSE
        pred_temp = logistic_model(
            calculate_weighted_Ta(pd.Series(AT_fit), lag, popt[4]).values,
            *popt[:4]
        )
        rmse = np.sqrt(mean_squared_error(temp_fit, pred_temp))

        print(f"[{comid}] lag={lag} → f={popt[4]:.4f}, RMSE={rmse:.4f}")
        return popt, rmse

    except Exception as e:
        print(f"[{comid}] lag={lag} → 拟合失败：{str(e)[:50]}")
        return None, np.inf


# 6. 滞后天数优选
def select_best_lag(AT_fit, temp_fit, comid, candidate_lags=[0,1,2,3,4]):
    print(f"\n[{comid}] 滞后天数优选（候选：{candidate_lags}）")
    lag_results = []

    for lag in candidate_lags:
        popt, rmse = fit_single_lag(AT_fit, temp_fit, lag, comid)
        if popt is not None:
            lag_results.append((lag, popt, rmse))

    if not lag_results:
        print(f"[{comid}] 所有lag拟合失败")
        return None, None, np.inf

    # 按RMSE排序选最优
    lag_results.sort(key=lambda x: x[2])
    best_lag, best_popt, best_rmse = lag_results[0]
    a, b, c, d, f_best = best_popt
    print(f"[{comid}] 最优lag：{best_lag}（RMSE：{best_rmse:.4f}），参数：a={a:.4f}, f={f_best:.4f}")

    return best_lag, best_popt, best_rmse


# 7. 主函数（核心：全日期预测）
def process_data(input_folder, output_folder, candidate_lags=[0,1,2,3,4]):
    os.makedirs(output_folder, exist_ok=True)
    catchment_params = []

    for file_name in os.listdir(input_folder):
        if not file_name.endswith('.xlsx'):
            continue
        comid = file_name.split('.')[0]
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        try:
            print(f"\n" + "="*50)
            print(f"处理流域：{file_name}（COMID：{comid}）")
            print("="*50)

            # 1. 读取原始数据（保留所有日期）
            df = pd.read_excel(input_path)
            required_cols = ['temp', 'AT_mean', 'DOY']
            if not all(col in df.columns for col in required_cols):
                print(f"[{comid}] 跳过：缺少列{required_cols}")
                continue

            # 2. 数据准备（保留全量日期，筛选拟合行）
            df_full, AT_fit, temp_fit = prepare_data(df, comid)
            if len(AT_fit) < 15:
                print(f"[{comid}] 跳过：有效拟合行不足15个")
                continue

            # 3. 优选滞后天数和参数
            best_lag, best_popt, best_rmse = select_best_lag(AT_fit, temp_fit, comid, candidate_lags)
            if best_popt is None:
                print(f"[{comid}] 跳过：无有效参数")
                continue

            # 4. 全日期预测（核心修复：对所有日期生成predicted_temp=Logist_RWT）
            a, b, c, d, f_best = best_popt
            # 对原始数据的全量气温计算预测值（包括无水温的日期）
            df_full['Logist_RWT'] = model_for_fitting(
                AT_mean_series=df_full['AT_mean'],  # 全量日期的气温
                a=a, b=b, c=c, d=d, f=f_best, lag=best_lag
            )

            # 5. 补充其他辅助列
            df_full['best_lag'] = best_lag  # 该流域的最优滞后值
            df_full['fit_RMSE'] = best_rmse  # 拟合精度
            # 标记“是否用于拟合”（方便后续查看）
            df_full['used_for_fitting'] = df_full['is_fit_valid']
            # 计算加权滞后气温（全量日期，可选保留）
            df_full['weighted_Ta'] = calculate_weighted_Ta(df_full['AT_mean'], best_lag, f_best)

            # 6. 删除临时列，整理输出
            df_output = df_full.drop(columns=['is_fit_valid'])  # 删除临时筛选列

            # 7. 保存全量结果（所有日期都包含predicted_temp=Logist_RWT）
            df_output.to_excel(output_path, index=False)
            print(f"\n[{comid}] 结果保存：{output_path}（共{len(df_output)}行，全量日期覆盖）")

            # 8. 记录最优参数
            catchment_params.append({
                'COMID': comid,
                'best_lag': best_lag,
                'a': a, 'b': b, 'c': c, 'd': d, 'f': f_best,
                'fit_RMSE': best_rmse,
                'valid_fit_rows': len(AT_fit),
                'total_days': len(df_output)  # 全量日期数
            })

        except Exception as e:
            print(f"[{comid}] 处理出错：{str(e)}")
            continue

    # 保存参数汇总
    if catchment_params:
        params_df = pd.DataFrame(catchment_params)
        params_path = os.path.join(output_folder, 'catchment_best_params_full_days.xlsx')
        params_df.to_excel(params_path, index=False)
        print(f"\n" + "="*50)
        print(f"参数汇总保存：{params_path}")
        print("="*50)

        # 统计信息
        print(f"\n全量日期预测统计：")
        total_days = sum([p['total_days'] for p in catchment_params])
        print(f"  所有流域总日期数：{total_days}")
        print(f"  最优lag分布：{params_df['best_lag'].value_counts().sort_index().to_dict()}")
        print(f"  平均拟合RMSE：{params_df['fit_RMSE'].mean():.4f}")

    else:
        print(f"\n无成功处理的流域")


# 执行
if __name__ == "__main__":
    INPUT_FOLDER = r"E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\Statistical regression\\COMID_AT_ATC_temp"
    OUTPUT_FOLDER = r"E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\Statistical regression\\output_full_new"
    CANDIDATE_LAGS = [0, 1, 2, 3]  # 候选滞后天数

    # process_data(INPUT_FOLDER, OUTPUT_FOLDER, candidate_lags=CANDIDATE_LAGS)










# =====================================   处理  COMID_AT_ATC_temp  文件，设置和air2stream一样的人工数据缺失  ==================
import os
import pandas as pd
from pathlib import Path


def remove_matched_temperatures(modified_records_file, xlsx_folder):
    """
    根据修改记录文件，在汇流区xlsx文件中匹配并删除对应temp值

    参数:
    modified_records_file: 之前生成的修改记录Excel文件路径
    xlsx_folder: 包含汇流区xlsx文件的文件夹路径
    """
    # 读取修改记录文件
    try:
        modified_df = pd.read_excel(modified_records_file)
        print(f"成功读取修改记录文件，共 {len(modified_df)} 条记录")
    except Exception as e:
        print(f"读取修改记录文件失败: {e}")
        return

    # 确保COMID是字符串类型，避免数字匹配问题
    modified_df['汇流区ID'] = modified_df['汇流区ID'].astype(str)
    # 确保日期格式一致
    modified_df['date'] = pd.to_datetime(modified_df['日期']).dt.date

    # 按汇流区ID分组，方便后续匹配
    grouped = modified_df.groupby('汇流区ID')

    # 遍历文件夹中的所有xlsx文件
    for filename in os.listdir(xlsx_folder):
        if filename.endswith('.xlsx') and filename.split('.')[0].isdigit():
            comid = filename.split('.')[0]
            file_path = os.path.join(xlsx_folder, filename)

            # 检查该COMID是否在修改记录中
            if comid not in grouped.groups:
                print(f"文件 {filename} 没有匹配的修改记录，跳过")
                continue

            try:
                # 读取汇流区数据文件
                df = pd.read_excel(file_path)
                print(f"处理文件: {filename}，共 {len(df)} 行数据")

                # 确保日期格式一致
                df['date'] = pd.to_datetime(df['date']).dt.date

                # 获取该COMID对应的所有修改记录日期
                comid_records = grouped.get_group(comid)
                dates_to_remove = set(comid_records['date'])

                # 找到需要删除temp的行
                mask = df['date'].isin(dates_to_remove)
                rows_removed = sum(mask)

                if rows_removed > 0:
                    # 将匹配行的temp设置为空
                    df.loc[mask, 'temp'] = None

                    # 保存修改后的文件
                    df.to_excel(file_path, index=False)
                    print(f"  已在 {filename} 中删除 {rows_removed} 行的temp值")
                else:
                    print(f"  在 {filename} 中未找到匹配的记录")

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")


if __name__ == "__main__":
    # 请修改为实际的文件路径
    modified_records_path = "E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\Air2stream\\Modified_999_records.xlsx"  # 之前生成的修改记录文件
    xlsx_files_folder = "E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\Statistical regression\\COMID_AT_ATC_temp"  # 包含汇流区xlsx文件的文件夹

    # 执行处理
    # remove_matched_temperatures(modified_records_path, xlsx_files_folder)
    # print("处理完成")
