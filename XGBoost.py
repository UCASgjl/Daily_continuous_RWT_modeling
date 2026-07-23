# =============================   Dail RWT reconstruction -XGBoost =========================================
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from itertools import product
import os
import shap
import matplotlib.pyplot as plt
import openpyxl
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter


def ensure_features(X, features):
    """确保测试数据包含所有特征列（含缺失值指示列）"""
    for col in features:
        if col not in X.columns:
            X[col] = np.nan
        if f'{col}_missing' not in X.columns:
            X[f'{col}_missing'] = X[col].isnull().astype(int)
    return X[features + [f'{c}_missing' for c in features]]


def create_param_grid():
    """返回参数搜索网格"""
    return {
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [150, 200, 300, 400],
        'min_child_weight': [3, 5, 7, 10],
        'missing': [np.nan],
        'tree_method': ['hist'],
        'enable_categorical': [True]
    }


def calc_metrics(y_true, y_pred):
    """计算整体 RMSE, MAE, R2, ME(Bias)"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    me = np.mean(y_pred - y_true)
    return rmse, mae, r2, me


def calc_median_metrics(df, comid_col, target_col, pred_col):
    """
    按COMID分组计算 Median RMSE 和 Median AE
    """
    df = df.copy()
    df['sq_error'] = (df[target_col] - df[pred_col]) ** 2
    df['abs_error'] = np.abs(df[target_col] - df[pred_col])

    grouped = df.groupby(comid_col)
    comid_rmse = np.sqrt(grouped['sq_error'].mean())
    comid_mae = grouped['abs_error'].mean()

    median_rmse = comid_rmse.median() if len(comid_rmse) > 0 else np.nan
    median_ae = comid_mae.median() if len(comid_mae) > 0 else np.nan
    return median_rmse, median_ae


def calc_comid_metrics(df, comid_col, target_col, pred_col):
    """
    计算每个COMID（站点）的 RMSE 和 MAE
    """
    df = df.copy()
    df['sq_error'] = (df[target_col] - df[pred_col]) ** 2
    df['abs_error'] = np.abs(df[target_col] - df[pred_col])

    grouped = df.groupby(comid_col)
    result = pd.DataFrame({
        'RMSE': np.sqrt(grouped['sq_error'].mean()),
        'MAE': grouped['abs_error'].mean(),
        'n_samples': grouped.size()
    }).reset_index()
    return result


# ========================== 新增：站点级 SHAP 特征重要性 ==========================

def calc_comid_shap_importance(model, df, comid_col, feature_cols,
                                sample_per_comid=None, random_state=42):
    """
    计算每个站点每个特征的 SHAP 平均绝对值（站点级特征重要性）。

    Parameters:
        model: 训练好的 XGBoost 模型
        df: 包含特征和 COMID 的数据框
        comid_col: 站点列名（如 'COMID'）
        feature_cols: 模型输入的特征列名列表
        sample_per_comid: 每个站点最多采样样本数，None 表示不采样
        random_state: 随机种子

    Returns:
        DataFrame: 行索引为 COMID，列为各特征的平均绝对 SHAP 值
    """
    np.random.seed(random_state)
    X_full = df[feature_cols].copy()
    comids = df[comid_col].values

    # 若数据量大，对每个站点分别采样以加速 SHAP 计算
    if sample_per_comid is not None:
        selected_idx = []
        for comid in np.unique(comids):
            mask = comids == comid
            idx = np.where(mask)[0]
            if len(idx) > sample_per_comid:
                idx = np.random.choice(idx, sample_per_comid, replace=False)
            selected_idx.extend(idx)
        selected_idx = np.array(selected_idx)
        X = X_full.iloc[selected_idx]
        comids_sub = comids[selected_idx]
    else:
        X = X_full
        comids_sub = comids

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # 构建 DataFrame 并按 COMID 聚合（平均绝对 SHAP 值）
    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    shap_df[comid_col] = comids_sub

    importance = shap_df.groupby(comid_col)[feature_cols].apply(
        lambda x: np.abs(x).mean()
    )

    # importance = shap_df.groupby(comid_col)[feature_cols].mean()

    return importance


# ========================== 通用参数搜索 ==========================

def param_search(param_grid, X, y, cv_func):
    """
    通用参数搜索框架
    cv_func: 接收 (param_dict, X, y) -> (avg_val_rmse, fold_results) 的函数
    """
    param_combinations = list(product(*param_grid.values()))
    best_val_rmse = float('inf')
    best_params = None
    all_results = []

    for idx, params in enumerate(param_combinations, 1):
        param_dict = dict(zip(param_grid.keys(), params))
        avg_val_rmse, fold_results = cv_func(param_dict, X, y)

        all_results.extend(fold_results)

        if avg_val_rmse < best_val_rmse:
            best_val_rmse = avg_val_rmse
            best_params = param_dict

        print(f"  [{idx}/{len(param_combinations)}] 参数: {param_dict} -> 平均验证RMSE: {avg_val_rmse:.4f}")

    return best_params, best_val_rmse, all_results


# ========================== Space CV ==========================

def space_cv_func(param_dict, X, y, train_data, comid_splits):
    """空间交叉验证：5折（基于COMID划分）"""
    train_val_comids = np.concatenate([comid_splits[i] for i in [0, 1, 2, 4, 5]])
    train_val_splits = np.array_split(train_val_comids, 5)

    val_rmse_list = []
    fold_results = []

    for val_fold in range(5):
        val_comids = train_val_splits[val_fold]
        train_comids = np.concatenate([train_val_splits[i] for i in range(5) if i != val_fold])

        val_idx = train_data['COMID'].isin(val_comids)
        train_idx = train_data['COMID'].isin(train_comids)

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = XGBRegressor(**param_dict)
        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_rmse_list.append(val_rmse)

        fold_results.append({
            '参数组合': str(param_dict),
            '折数': val_fold + 1,
            '验证集RMSE': val_rmse
        })

    avg_val_rmse = np.mean(val_rmse_list)
    fold_results.append({
        '参数组合': str(param_dict),
        '折数': '平均',
        '验证集RMSE': avg_val_rmse
    })

    return avg_val_rmse, fold_results


def run_space_cv(train_data, X, y, features, output_folder):
    """执行空间交叉验证全流程"""
    print("\n" + "=" * 60)
    print("【空间交叉验证】")
    print("=" * 60)

    # 划分COMID
    np.random.seed(42)
    comids = train_data['COMID'].unique()
    np.random.shuffle(comids)
    comid_splits = np.array_split(comids, 6)

    test_comids = comid_splits[3]
    train_val_comids = np.concatenate([comid_splits[i] for i in [0, 1, 2, 4, 5]])

    test_idx = train_data['COMID'].isin(test_comids)
    train_val_idx = train_data['COMID'].isin(train_val_comids)

    X_test, y_test = X[test_idx], y[test_idx]
    X_train_val, y_train_val = X[train_val_idx], y[train_val_idx]

    # 参数搜索
    param_grid = create_param_grid()
    best_params, best_val_rmse, all_results = param_search(
        param_grid, X, y,
        lambda p, X_, y_: space_cv_func(p, X_, y_, train_data, comid_splits)
    )

    print(f"\n最优参数: {best_params}")
    print(f"最优平均验证RMSE: {best_val_rmse:.4f}")

    # 最优参数下重新跑5折CV并保存每折预测 + 站点特征重要性
    train_val_splits = np.array_split(train_val_comids, 5)
    fold_metrics_list = []
    all_folds_list = []
    all_fold_comid_importance = []  # 收集每折的站点特征重要性

    # 所有模型输入特征（含 missing 指示列）
    all_feature_cols = features + [f'{c}_missing' for c in features]

    for val_fold in range(5):
        val_comids = train_val_splits[val_fold]
        train_comids = np.concatenate([train_val_splits[i] for i in range(5) if i != val_fold])

        val_idx = train_data['COMID'].isin(val_comids)
        train_idx = train_data['COMID'].isin(train_comids)

        X_train_fold, y_train_fold = X[train_idx], y[train_idx]
        X_val_fold, y_val_fold = X[val_idx], y[val_idx]

        model = XGBRegressor(**best_params)
        model.fit(X_train_fold, y_train_fold)

        y_val_pred = model.predict(X_val_fold)

        # 保存该折验证集预测
        val_original = train_data[val_idx].copy().reset_index(drop=True)
        val_original['Predicted_RWT'] = y_val_pred
        val_original['fold'] = val_fold + 1
        val_path = os.path.join(output_folder, f'Space_最优参数_第{val_fold + 1}折验证集预测.xlsx')
        val_original.to_excel(val_path, index=False)
        print(f"  ✓ 已保存第{val_fold + 1}折验证集预测: {val_path}")

        # 计算该折整体精度
        rmse_f, mae_f, r2_f, me_f = calc_metrics(y_val_fold, y_val_pred)
        median_rmse_f, median_ae_f = calc_median_metrics(val_original, 'COMID', 'temp', 'Predicted_RWT')

        fold_metrics_list.append({
            'fold': val_fold + 1,
            'RMSE': rmse_f,
            'MAE': mae_f,
            'R2': r2_f,
            'ME': me_f,
            'Median_RMSE': median_rmse_f,
            'Median_AE': median_ae_f,
            'n_samples': len(val_original)
        })

        # 计算并保存该折每个站点的精度
        comid_metrics = calc_comid_metrics(val_original, 'COMID', 'temp', 'Predicted_RWT')
        comid_metrics_path = os.path.join(output_folder, f'Space_最优参数_第{val_fold + 1}折_各站点精度.xlsx')
        comid_metrics.to_excel(comid_metrics_path, index=False)
        print(f"  ✓ 已保存第{val_fold + 1}折各站点精度: {comid_metrics_path}")

        # ========== 新增：① 该折验证集每个站点的特征重要性（SHAP） ==========
        print(f"  计算第{val_fold + 1}折验证集站点特征重要性（SHAP）...")
        fold_comid_imp = calc_comid_shap_importance(
            model, val_original, 'COMID', all_feature_cols
        )
        fold_imp_path = os.path.join(output_folder, f'Space_第{val_fold + 1}折_验证集站点特征重要性.xlsx')
        fold_comid_imp.to_excel(fold_imp_path)
        print(f"  ✓ 已保存第{val_fold + 1}折验证集站点特征重要性: {fold_imp_path}")

        # 记录该折重要性，用于后续汇总
        fold_comid_imp_long = fold_comid_imp.reset_index().melt(
            id_vars=['COMID'], var_name='Feature', value_name='MeanAbsSHAP'
        )
        fold_comid_imp_long['fold'] = val_fold + 1
        all_fold_comid_importance.append(fold_comid_imp_long)

        all_folds_list.append(val_original)

    # 合并所有折验证集
    all_folds_df = pd.concat(all_folds_list, ignore_index=True)
    merge_path = os.path.join(output_folder, 'Space_最优参数_所有折验证集合并.xlsx')
    all_folds_df.to_excel(merge_path, index=False)
    print(f"\n已合并所有折验证集: {merge_path} ({len(all_folds_df)} 条)")

    # 保存每折精度汇总
    fold_metrics_df = pd.DataFrame(fold_metrics_list)
    fold_metrics_path = os.path.join(output_folder, 'Space_最优参数_各折验证集精度汇总.xlsx')
    fold_metrics_df.to_excel(fold_metrics_path, index=False)
    print(f"已保存各折精度汇总: {fold_metrics_path}")

    # 计算5折验证集整体精度（基于合并后数据）
    rmse_val, mae_val, r2_val, me_val = calc_metrics(
        all_folds_df['temp'].values, all_folds_df['Predicted_RWT'].values
    )
    median_rmse_val, median_ae_val = calc_median_metrics(all_folds_df, 'COMID', 'temp', 'Predicted_RWT')
    print(f"\n5折验证集整体精度:")
    print(f"  RMSE={rmse_val:.4f}, MAE={mae_val:.4f}, R2={r2_val:.4f}, ME={me_val:.4f}")
    print(f"  Median_RMSE={median_rmse_val:.4f}, Median_AE={median_ae_val:.4f}")

    # 汇总所有折的站点特征重要性（长格式）
    if all_fold_comid_importance:
        all_fold_imp_df = pd.concat(all_fold_comid_importance, ignore_index=True)
        all_fold_imp_path = os.path.join(output_folder, 'Space_所有折_验证集站点特征重要性汇总.xlsx')
        all_fold_imp_df.to_excel(all_fold_imp_path, index=False)
        print(f"已保存所有折验证集站点特征重要性汇总（长格式）: {all_fold_imp_path}")

    # 训练最终模型（全部train_val数据）
    final_model = XGBRegressor(**best_params)
    final_model.fit(X_train_val, y_train_val)

    # 独立测试集评估
    y_test_pred = final_model.predict(X_test)
    rmse_test, mae_test, r2_test, me_test = calc_metrics(y_test, y_test_pred)

    test_df = train_data[test_idx].copy().reset_index(drop=True)
    test_df['Predicted_RWT'] = y_test_pred
    median_rmse_test, median_ae_test = calc_median_metrics(test_df, 'COMID', 'temp', 'Predicted_RWT')

    # 保存独立测试集（含预测值）
    test_path = os.path.join(output_folder, 'Space_独立测试集样本.xlsx')
    test_df.to_excel(test_path, index=False)
    print(f"\n已导出独立测试集(含预测值): {test_path}")

    # 计算并保存独立测试集每个站点的精度
    test_comid_metrics = calc_comid_metrics(test_df, 'COMID', 'temp', 'Predicted_RWT')
    test_comid_path = os.path.join(output_folder, 'Space_独立测试集_各站点精度.xlsx')
    test_comid_metrics.to_excel(test_comid_path, index=False)
    print(f"已保存独立测试集各站点精度: {test_comid_path}")

    # ========== 新增：② 最终模型 - 独立测试集站点特征重要性 ==========
    print("  计算最终模型独立测试集站点特征重要性（SHAP）...")
    test_comid_imp = calc_comid_shap_importance(
        final_model, test_df, 'COMID', all_feature_cols
    )
    test_imp_path = os.path.join(output_folder, 'Space_最终模型_测试集站点特征重要性.xlsx')
    test_comid_imp.to_excel(test_imp_path)
    print(f"  ✓ 已保存最终模型测试集站点特征重要性: {test_imp_path}")

    # 训练集回带
    y_train_pred = final_model.predict(X_train_val)
    rmse_train, mae_train, r2_train, me_train = calc_metrics(y_train_val, y_train_pred)

    train_df = train_data[train_val_idx].copy().reset_index(drop=True)
    train_df['Predicted_RWT'] = y_train_pred
    median_rmse_train, median_ae_train = calc_median_metrics(train_df, 'COMID', 'temp', 'Predicted_RWT')

    # ========== 新增：② 最终模型 - 训练集站点特征重要性 ==========
    print("  计算最终模型训练集站点特征重要性（SHAP）...")
    train_comid_imp = calc_comid_shap_importance(
        final_model, train_df, 'COMID', all_feature_cols
    )
    train_imp_path = os.path.join(output_folder, 'Space_最终模型_训练集站点特征重要性.xlsx')
    train_comid_imp.to_excel(train_imp_path)
    print(f"  ✓ 已保存最终模型训练集站点特征重要性: {train_imp_path}")

    print(f"\n最终模型精度:")
    print(f"  训练集: RMSE={rmse_train:.4f}, MAE={mae_train:.4f}, R2={r2_train:.4f}, ME={me_train:.4f}, Median_RMSE={median_rmse_train:.4f}, Median_AE={median_ae_train:.4f}")
    print(f"  验证集: RMSE={rmse_val:.4f}, MAE={mae_val:.4f}, R2={r2_val:.4f}, ME={me_val:.4f}, Median_RMSE={median_rmse_val:.4f}, Median_AE={median_ae_val:.4f}")
    print(f"  测试集: RMSE={rmse_test:.4f}, MAE={mae_test:.4f}, R2={r2_test:.4f}, ME={me_test:.4f}, Median_RMSE={median_rmse_test:.4f}, Median_AE={median_ae_test:.4f}")

    # 保存结果
    results_df = pd.DataFrame(all_results)
    final_results = pd.DataFrame({
        '评估数据集': ['训练集(回带)', '验证集(5折CV)', '独立测试集'],
        'RMSE': [rmse_train, rmse_val, rmse_test],
        'MAE': [mae_train, mae_val, mae_test],
        'R2': [r2_train, r2_val, r2_test],
        'ME': [me_train, me_val, me_test],
        'Median_RMSE': [median_rmse_train, median_rmse_val, median_rmse_test],
        'Median_AE': [median_ae_train, median_ae_val, median_ae_test]
    })

    excel_path = os.path.join(output_folder, 'Space_模型评估结果.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='所有参数结果', index=False)
        final_results.to_excel(writer, sheet_name='最优参数最终评估', index=False)
        pd.DataFrame([best_params]).to_excel(writer, sheet_name='最优参数', index=False)
        fold_metrics_df.to_excel(writer, sheet_name='各折验证集精度', index=False)

    print(f"\n结果已保存: {excel_path}")

    return final_model, best_params, features


# ========================== Time CV ==========================

def time_cv_func(param_dict, X, y, train_data):
    """时间交叉验证：3折（基于日期划分）"""
    date_splits = [
        (train_data['date'] >= '2021-01-01') & (train_data['date'] <= '2021-06-30'),
        (train_data['date'] >= '2021-07-01') & (train_data['date'] <= '2021-12-31'),
        (train_data['date'] >= '2022-01-01') & (train_data['date'] <= '2022-06-30'),
        (train_data['date'] >= '2022-07-01') & (train_data['date'] <= '2022-12-31')
    ]

    test_idx = date_splits[3]
    train_val_dates = [date_splits[i] for i in range(3)]

    X_test, y_test = X[test_idx], y[test_idx]

    val_rmse_list = []
    fold_results = []

    for val_fold in range(3):
        val_dates = train_val_dates[val_fold]
        train_dates = [train_val_dates[i] for i in range(3) if i != val_fold]
        train_idx = train_dates[0] | train_dates[1]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_dates], y[val_dates]

        model = XGBRegressor(**param_dict)
        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_rmse_list.append(val_rmse)

        fold_results.append({
            '参数组合': str(param_dict),
            '折数': val_fold + 1,
            '验证集RMSE': val_rmse
        })

    avg_val_rmse = np.mean(val_rmse_list)
    fold_results.append({
        '参数组合': str(param_dict),
        '折数': '平均',
        '验证集RMSE': avg_val_rmse
    })

    return avg_val_rmse, fold_results, test_idx


def run_time_cv(train_data, X, y, features, output_folder):
    """执行时间交叉验证全流程"""
    print("\n" + "=" * 60)
    print("【时间交叉验证】")
    print("=" * 60)

    all_feature_cols = features + [f'{c}_missing' for c in features]

    # 参数搜索
    param_grid = create_param_grid()
    best_params, best_val_rmse, all_results, test_idx = None, float('inf'), [], None

    param_combinations = list(product(*param_grid.values()))
    for idx, params in enumerate(param_combinations, 1):
        param_dict = dict(zip(param_grid.keys(), params))
        avg_val_rmse, fold_results, test_idx = time_cv_func(param_dict, X, y, train_data)
        all_results.extend(fold_results)

        if avg_val_rmse < best_val_rmse:
            best_val_rmse = avg_val_rmse
            best_params = param_dict

        print(f"  [{idx}/{len(param_combinations)}] 参数: {param_dict} -> 平均验证RMSE: {avg_val_rmse:.4f}")

    print(f"\n最优参数: {best_params}")
    print(f"最优平均验证RMSE: {best_val_rmse:.4f}")

    # 训练最终模型（前3个时间段）
    train_val_idx = (train_data['date'] >= '2019-01-01') & (train_data['date'] <= '2020-06-30')
    X_train_val, y_train_val = X[train_val_idx], y[train_val_idx]

    final_model = XGBRegressor(**best_params)
    final_model.fit(X_train_val, y_train_val)

    # 独立测试集评估（2020下半年）
    X_test, y_test = X[test_idx], y[test_idx]
    y_test_pred = final_model.predict(X_test)
    rmse_test, mae_test, r2_test, me_test = calc_metrics(y_test, y_test_pred)

    test_df = train_data[test_idx].copy().reset_index(drop=True)
    test_df['Predicted_RWT'] = y_test_pred
    test_path = os.path.join(output_folder, 'Time_独立测试集样本.xlsx')
    test_df.to_excel(test_path, index=False)
    print(f"已导出独立测试集(含预测值): {test_path}")

    # ========== 新增：最终模型 - 测试集站点特征重要性 ==========
    print("  计算最终模型测试集站点特征重要性（SHAP）...")
    test_comid_imp = calc_comid_shap_importance(
        final_model, test_df, 'COMID', all_feature_cols
    )
    test_imp_path = os.path.join(output_folder, 'Time_最终模型_测试集站点特征重要性.xlsx')
    test_comid_imp.to_excel(test_imp_path)
    print(f"  ✓ 已保存最终模型测试集站点特征重要性: {test_imp_path}")

    # 训练集回带
    y_train_pred = final_model.predict(X_train_val)
    rmse_train, mae_train, r2_train, me_train = calc_metrics(y_train_val, y_train_pred)

    train_df = train_data[train_val_idx].copy().reset_index(drop=True)
    train_df['Predicted_RWT'] = y_train_pred

    # ========== 新增：最终模型 - 训练集站点特征重要性 ==========
    print("  计算最终模型训练集站点特征重要性（SHAP）...")
    train_comid_imp = calc_comid_shap_importance(
        final_model, train_df, 'COMID', all_feature_cols
    )
    train_imp_path = os.path.join(output_folder, 'Time_最终模型_训练集站点特征重要性.xlsx')
    train_comid_imp.to_excel(train_imp_path)
    print(f"  ✓ 已保存最终模型训练集站点特征重要性: {train_imp_path}")

    print(f"最终模型 - 训练集RMSE: {rmse_train:.4f}")
    print(f"最终模型 - 验证集RMSE: {best_val_rmse:.4f}")
    print(f"最终模型 - 独立测试集RMSE: {rmse_test:.4f}")

    # 保存结果
    results_df = pd.DataFrame(all_results)
    final_results = pd.DataFrame({
        '评估数据集': ['训练集(回带)', '验证集(3折CV)', '独立测试集'],
        'RMSE': [rmse_train, best_val_rmse, rmse_test],
        'MAE': [mae_train, np.nan, mae_test],
        'R2': [r2_train, np.nan, r2_test],
        'ME': [me_train, np.nan, me_test]
    })

    excel_path = os.path.join(output_folder, 'Time_模型评估结果.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='所有参数结果', index=False)
        final_results.to_excel(writer, sheet_name='最优参数最终评估', index=False)
        pd.DataFrame([best_params]).to_excel(writer, sheet_name='最优参数', index=False)

    print(f"结果已保存: {excel_path}")

    return final_model, best_params, features


# ========================== Random CV ==========================

def random_cv_func(param_dict, X, y):
    """随机交叉验证：5折"""
    val_rmse_list = []
    fold_results = []

    for val_fold in range(5):
        # 随机划分20%验证集
        np.random.seed(42 + val_fold)
        val_idx = np.random.choice(len(X), int(0.2 * len(X)), replace=False)
        val_mask = np.isin(np.arange(len(X)), val_idx)

        X_val, y_val = X[val_mask], y[val_mask]
        train_mask = ~val_mask
        X_train, y_train = X[train_mask], y[train_mask]

        model = XGBRegressor(**param_dict)
        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_rmse_list.append(val_rmse)

        fold_results.append({
            '参数组合': str(param_dict),
            '折数': val_fold + 1,
            '验证集RMSE': val_rmse
        })

    avg_val_rmse = np.mean(val_rmse_list)
    fold_results.append({
        '参数组合': str(param_dict),
        '折数': '平均',
        '验证集RMSE': avg_val_rmse
    })

    return avg_val_rmse, fold_results


def run_random_cv(train_data, X, y, features, output_folder):
    """执行随机交叉验证全流程"""
    print("\n" + "=" * 60)
    print("【随机交叉验证】")
    print("=" * 60)

    all_feature_cols = features + [f'{c}_missing' for c in features]

    # 参数搜索
    param_grid = create_param_grid()
    best_params, best_val_rmse, all_results = param_search(
        param_grid, X, y,
        lambda p, X_, y_: random_cv_func(p, X_, y_)
    )

    print(f"\n最优参数: {best_params}")
    print(f"最优平均验证RMSE: {best_val_rmse:.4f}")

    # 随机划分20%测试集
    np.random.seed(42)
    test_idx = np.random.choice(len(X), int(0.2 * len(X)), replace=False)
    test_mask = np.isin(np.arange(len(X)), test_idx)

    X_test, y_test = X[test_mask], y[test_mask]
    train_val_mask = ~test_mask
    X_train_val, y_train_val = X[train_val_mask], y[train_val_mask]

    # 训练最终模型
    final_model = XGBRegressor(**best_params)
    final_model.fit(X_train_val, y_train_val)

    # 测试集评估
    y_test_pred = final_model.predict(X_test)
    rmse_test, mae_test, r2_test, me_test = calc_metrics(y_test, y_test_pred)

    test_df = train_data.iloc[test_idx].copy().reset_index(drop=True)
    test_df['Predicted_RWT'] = y_test_pred
    test_path = os.path.join(output_folder, 'Random_独立测试集样本.xlsx')
    test_df.to_excel(test_path, index=False)
    print(f"已导出独立测试集(含预测值): {test_path}")

    # ========== 新增：最终模型 - 测试集站点特征重要性 ==========
    print("  计算最终模型测试集站点特征重要性（SHAP）...")
    test_comid_imp = calc_comid_shap_importance(
        final_model, test_df, 'COMID', all_feature_cols
    )
    test_imp_path = os.path.join(output_folder, 'Random_最终模型_测试集站点特征重要性.xlsx')
    test_comid_imp.to_excel(test_imp_path)
    print(f"  ✓ 已保存最终模型测试集站点特征重要性: {test_imp_path}")

    # 训练集回带
    y_train_pred = final_model.predict(X_train_val)
    rmse_train, mae_train, r2_train, me_train = calc_metrics(y_train_val, y_train_pred)

    train_df = train_data.iloc[train_val_mask].copy().reset_index(drop=True)
    train_df['Predicted_RWT'] = y_train_pred

    # ========== 新增：最终模型 - 训练集站点特征重要性 ==========
    print("  计算最终模型训练集站点特征重要性（SHAP）...")
    train_comid_imp = calc_comid_shap_importance(
        final_model, train_df, 'COMID', all_feature_cols
    )
    train_imp_path = os.path.join(output_folder, 'Random_最终模型_训练集站点特征重要性.xlsx')
    train_comid_imp.to_excel(train_imp_path)
    print(f"  ✓ 已保存最终模型训练集站点特征重要性: {train_imp_path}")

    print(f"最终模型 - 训练集RMSE: {rmse_train:.4f}")
    print(f"最终模型 - 验证集RMSE: {best_val_rmse:.4f}")
    print(f"最终模型 - 测试集RMSE: {rmse_test:.4f}")

    # 保存结果
    results_df = pd.DataFrame(all_results)
    final_results = pd.DataFrame({
        '评估数据集': ['训练集', '验证集(5折CV)', '测试集'],
        'RMSE': [rmse_train, best_val_rmse, rmse_test],
        'MAE': [mae_train, np.nan, mae_test],
        'R2': [r2_train, np.nan, r2_test],
        'ME': [me_train, np.nan, me_test]
    })

    excel_path = os.path.join(output_folder, 'Random_模型评估结果.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='所有参数结果', index=False)
        final_results.to_excel(writer, sheet_name='最优参数最终评估', index=False)
        pd.DataFrame([best_params]).to_excel(writer, sheet_name='最优参数', index=False)

    print(f"结果已保存: {excel_path}")

    return final_model, best_params, features


# ========================== 批量预测 ==========================

def batch_predict(test_folder_path, output_folder_path, models_dict, features):
    """根据选中的模型进行批量预测"""
    print("\n" + "=" * 60)
    print("【批量预测日期文件夹】")
    print("=" * 60)

    test_files = [f for f in os.listdir(test_folder_path) if f.endswith('.xlsx')]
    if not test_files:
        print(f"警告: 文件夹 {test_folder_path} 中没有找到 .xlsx 文件")
        return

    for file in test_files:
        try:
            test_data = pd.read_excel(os.path.join(test_folder_path, file))

            # 确保特征完整
            X_test = ensure_features(test_data, features)

            # 只预测选中的模型
            for model_name, model_info in models_dict.items():
                if model_info is None:
                    continue
                model = model_info['model']
                pred_col = f'Predicted_RWT_{model_name}'
                test_data[pred_col] = model.predict(X_test)
                print(f"  {model_name} 预测完成: {file}")

            output_path = os.path.join(output_folder_path, file)
            test_data.to_excel(output_path, index=False)

        except Exception as e:
            print(f"处理文件 {file} 时出错: {str(e)}")


# ========================== SHAP 特征贡献度（全局） ==========================

def calculate_feature_contributions(X, models_dict, output_folder):
    """计算选中模型的SHAP特征贡献度"""
    print("\n" + "=" * 60)
    print("【特征贡献度分析】")
    print("=" * 60)

    # 采样加速
    sample_idx = np.random.choice(len(X), min(1000, len(X)), replace=False)
    X_sample = X.iloc[sample_idx] if isinstance(X, pd.DataFrame) else X[sample_idx]

    contributions_list = []
    valid_models = []

    for model_name, model_info in models_dict.items():
        if model_info is None:
            print(f"  {model_name} 未运行，跳过SHAP计算")
            continue

        model = model_info['model']
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            mean_abs = np.abs(shap_values).mean(axis=0)
            mean_abs = shap_values.mean(axis=0)
            direction = np.sign(shap_values).mean(axis=0)

            contributions_list.append((model_name, mean_abs, direction))
            valid_models.append(model_name)

            print(f"  {model_name} SHAP计算完成")

        except Exception as e:
            print(f"  {model_name} SHAP计算失败: {str(e)}")

    if not valid_models:
        print("没有可用的模型进行SHAP分析")
        return

    # 构建结果表
    feature_names = X.columns if isinstance(X, pd.DataFrame) else [f'Feature_{i}' for i in range(X.shape[1])]
    result_df = pd.DataFrame({'Feature': feature_names})

    for model_name, mean_abs, direction in contributions_list:
        result_df[f'{model_name}_Mean_Abs_SHAP'] = mean_abs
        result_df[f'{model_name}_Rank'] = np.argsort(np.argsort(-mean_abs)) + 1
        result_df[f'{model_name}_Direction'] = direction

    # 计算综合贡献度
    mean_cols = [f'{m}_Mean_Abs_SHAP' for m in valid_models]
    result_df['Mean_Contribution'] = result_df[mean_cols].mean(axis=1)
    result_df['Overall_Rank'] = np.argsort(np.argsort(-result_df['Mean_Contribution'])) + 1
    total = result_df['Mean_Contribution'].sum()
    result_df['Contribution_Percentage'] = result_df['Mean_Contribution'] / total * 100

    # 保存
    output_path = os.path.join(output_folder, 'Feature_Contributions.xlsx')
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        result_df.to_excel(writer, sheet_name='Contributions', index=False)

        # 美化
        workbook = writer.book
        sheet = workbook['Contributions']
        header_font = Font(bold=True)
        for cell in sheet[1]:
            cell.font = header_font
        for col in sheet.columns:
            max_len = max(len(str(cell.value)) for cell in col) + 2
            sheet.column_dimensions[get_column_letter(col[0].column)].width = min(max_len, 30)

    print(f"特征贡献度已保存: {output_path}")

    # 绘图
    plot_folder = os.path.join(output_folder, 'Contribution_Plots')
    os.makedirs(plot_folder, exist_ok=True)

    top_n = min(20, len(result_df))
    top_features = result_df.sort_values('Mean_Contribution', ascending=False).head(top_n)

    # 综合图
    plt.figure(figsize=(10, 8))
    colors = np.where(top_features['Overall_Rank'] <= 10, '#3498db', '#95a5a6')
    plt.barh(top_features['Feature'], top_features['Mean_Contribution'], color=colors)
    plt.xlabel('平均SHAP值绝对值（贡献度）')
    plt.ylabel('特征')
    plt.title(f'特征总体贡献度排名（Top {top_n}）')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, 'Overall_Contribution.png'), dpi=300)
    plt.close()

    # 各模型分图
    for model_name in valid_models:
        plt.figure(figsize=(10, 8))
        colors = np.where(top_features[f'{model_name}_Direction'] > 0, '#3498db', '#e74c3c')
        plt.barh(top_features['Feature'], top_features[f'{model_name}_Mean_Abs_SHAP'], color=colors)
        plt.xlabel('平均SHAP值绝对值（贡献度）')
        plt.ylabel('特征')
        plt.title(f'{model_name}模型特征贡献度排名（Top {top_n}）')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_folder, f'{model_name}_Contribution.png'), dpi=600)
        plt.close()

    print(f"贡献度图已保存: {plot_folder}")


# ========================== 主流程 ==========================

def train_and_predict(train_file_path, test_folder_path, output_folder_path, run_models=None):
    """
    主流程入口

    参数:
        run_models: 列表，可选 'Space', 'Time', 'Random'
                   默认为 None，表示全部运行
    """
    if run_models is None:
        run_models = ['Space', 'Time', 'Random']

    # 统一大小写
    run_models = [m.capitalize() for m in run_models]
    valid_models = {'Space', 'Time', 'Random'}
    run_models = [m for m in run_models if m in valid_models]

    if not run_models:
        raise ValueError("run_models 必须包含 'Space', 'Time', 'Random' 中的至少一个")

    print(f"\n{'#' * 60}")
    print(f"# 将运行以下模型: {', '.join(run_models)}")
    print(f"{'#' * 60}")

    os.makedirs(output_folder_path, exist_ok=True)

    # 读取训练数据
    print(f"\n读取训练数据: {train_file_path}")
    train_data = pd.read_excel(train_file_path)
    train_data['date'] = pd.to_datetime(train_data['date'])

    # 特征列
    features = [col for col in train_data.columns if col not in ['COMID', 'date', 'temp']]

    # 创建缺失值指示特征
    for col in features:
        train_data[f'{col}_missing'] = train_data[col].isnull().astype(int)

    X = train_data[features + [f'{c}_missing' for c in features]]
    y = train_data['temp']

    # 存储各模型结果
    models_dict = {}

    # 运行选中的CV
    if 'Space' in run_models:
        space_model, space_params, _ = run_space_cv(train_data, X, y, features, output_folder_path)
        models_dict['Space'] = {'model': space_model, 'params': space_params}

    if 'Time' in run_models:
        time_model, time_params, _ = run_time_cv(train_data, X, y, features, output_folder_path)
        models_dict['Time'] = {'model': time_model, 'params': time_params}

    if 'Random' in run_models:
        random_model, random_params, _ = run_random_cv(train_data, X, y, features, output_folder_path)
        models_dict['Random'] = {'model': random_model, 'params': random_params}

    # 批量预测（只预测选中的模型）
    batch_predict(test_folder_path, output_folder_path, models_dict, features)

    # SHAP分析（只计算选中的模型）
    calculate_feature_contributions(X, models_dict, output_folder_path)

    print(f"\n{'=' * 60}")
    print("全部完成！")
    print(f"{'=' * 60}")


# ========================== 用户配置区 ==========================

if __name__ == "__main__":
    # 路径配置
    train_file_path = "D:\\huai_river\\Station_RWT.xlsx"  #所有训练+验证样本
    test_folder_path = 'D:\\huai_river\\Huairiver_X'  #用于预测RWT的 输入变量
    output_folder_path = "D:\huai_river\\Huairiver_RWT"


    # 选择要运行的模型
    run_models = ['Space']  # 选择模型
    train_and_predict(train_file_path, test_folder_path, output_folder_path, run_models=run_models)