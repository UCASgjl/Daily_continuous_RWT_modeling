import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
import time  # 用于计时


def load_cv_folds(cv_fold_paths):
    """
    加载5折交叉验证的训练集文件
    :param cv_fold_paths: 5个训练集文件路径列表（如["空间交叉验证_训练集样本_第1折.xlsx", ...]）
    :return: 折名-数据框字典（{fold_name: df, ...}）、全局特征均值（用于缺失值填充）
    """
    start_time = time.time()
    cv_folds = {}

    # 检查文件有效性
    for path in cv_fold_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"交叉验证文件不存在：{path}")
        fold_name = os.path.splitext(os.path.basename(path))[0]  # 提取折名（如"空间交叉验证_训练集样本_第1折"）

        # 读取数据并检查必要列
        df = pd.read_excel(path, engine='openpyxl')
        required_cols = ['COMID', 'DOY', 'lat', 'lon', 'Mean_Value', 'Slope', 'Aspect',
                         'AT_mean', 'Evaporation_mean', 'DSR', 'LWDN', 'LAI_mean', 'LST_mean', 'date', 'temp']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"{fold_name} 缺失必要列：{missing_cols}")

        # 数据预处理：统一格式、处理日期
        df['COMID'] = df['COMID'].astype(str)
        df['date'] = pd.to_datetime(df['date']).dt.date  # 统一日期格式
        df = df.dropna(subset=['temp'])  # 删除目标变量（水温）缺失的行

        cv_folds[fold_name] = df
        print(f"已加载 {fold_name}：{len(df)} 条有效数据")

    # 计算全局特征均值（用于所有数据的缺失值填充，避免数据泄露）
    all_features = pd.concat([fold[required_cols[:-1]] for fold in cv_folds.values()], ignore_index=True)  # 排除temp
    global_means = all_features.mean(numeric_only=True).to_dict()

    print(f"\n5折数据加载完成（耗时: {time.time() - start_time:.2f}秒）")
    print(f"全局特征均值计算完成，共涉及 {len(all_features)} 条特征数据")
    return cv_folds, global_means


def cross_validation_train(cv_folds, global_means, x_cols):
    """
    5折交叉验证训练：每次用4折训练，1折验证，筛选RMSE最低的最优模型
    :param cv_folds: 折名-数据框字典
    :param global_means: 全局特征均值（用于缺失值填充）
    :param x_cols: 特征列列表
    :return: 最优模型参数（intercept + 各特征系数）、各折验证结果
    """
    start_time = time.time()
    fold_names = list(cv_folds.keys())
    cv_results = []  # 存储各折验证结果
    best_model = None
    best_rmse = float('inf')  # 初始化为无穷大，找最小RMSE

    print(f"\n开始5折交叉验证训练（特征列：{x_cols}）")
    for val_fold_idx in range(5):
        # 1. 拆分训练集（4折）和验证集（1折）
        val_fold_name = fold_names[val_fold_idx]
        train_fold_names = [name for i, name in enumerate(fold_names) if i != val_fold_idx]

        # 合并4折训练数据
        train_df = pd.concat([cv_folds[name] for name in train_fold_names], ignore_index=True)
        # 提取验证数据
        val_df = cv_folds[val_fold_name].copy()

        print(f"\n--- 第{val_fold_idx + 1}折验证（训练集：{len(train_df)}条，验证集：{len(val_df)}条）---")

        # 2. 处理训练集特征（缺失值填充）
        X_train = train_df[x_cols].copy()
        for col in x_cols:
            if X_train[col].isna().any():
                X_train[col].fillna(global_means[col], inplace=True)
        # 删除仍有缺失值的行（极端情况）
        X_train = X_train.dropna()
        y_train = train_df.loc[X_train.index, 'temp']

        # 3. 处理验证集特征（缺失值填充）
        X_val = val_df[x_cols].copy()
        for col in x_cols:
            if X_val[col].isna().any():
                X_val[col].fillna(global_means[col], inplace=True)
        X_val = X_val.dropna()
        y_val = val_df.loc[X_val.index, 'temp']

        # 4. 训练线性回归模型
        model = LinearRegression(fit_intercept=True, n_jobs=-1)  # 用所有CPU加速
        model.fit(X_train, y_train)

        # 5. 验证集预测与指标计算
        y_val_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        r2 = r2_score(y_val, y_val_pred)

        # 6. 保存该折结果
        fold_result = {
            "fold_name": val_fold_name,
            "train_sample_count": len(X_train),
            "val_sample_count": len(X_val),
            "rmse": rmse,
            "r2": r2,
            "intercept": model.intercept_,
            "coefficients": dict(zip(x_cols, model.coef_))
        }
        cv_results.append(fold_result)
        print(f"第{val_fold_idx + 1}折结果：RMSE={rmse:.2f}, R²={r2:.2f}")

        # 7. 更新最优模型（RMSE最小）
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = {
                "intercept": model.intercept_,
                "coefficients": dict(zip(x_cols, model.coef_)),
                "used_x_cols": x_cols,
                "val_fold_name": val_fold_name,
                "val_rmse": rmse,
                "val_r2": r2
            }

    # 输出交叉验证汇总
    print(f"\n{'=' * 50}")
    print("5折交叉验证汇总：")
    for i, res in enumerate(cv_results, 1):
        print(f"第{i}折（验证集：{res['fold_name']}）：RMSE={res['rmse']:.2f}, R²={res['r2']:.2f}")
    print(f"\n最优模型：")
    print(f"- 对应验证集：{best_model['val_fold_name']}")
    print(f"- 验证RMSE：{best_model['val_rmse']:.2f}, R²：{best_model['val_r2']:.2f}")
    print(f"- 截距：{best_model['intercept']:.4f}")
    print(f"- 系数：{best_model['coefficients']}")
    print(f"{'=' * 50}")

    print(f"\n5折交叉验证完成（耗时: {time.time() - start_time:.2f}秒）")
    return best_model, cv_results


def predict_with_best_model(df, best_model, global_means, pred_col_name="multiple_RWT_spatem"):
    """
    用最优模型对数据进行预测
    :param df: 输入数据框（需包含所有特征列）
    :param best_model: 最优模型参数（cross_validation_train返回）
    :param global_means: 全局特征均值（用于缺失值填充）
    :param pred_col_name: 预测结果列名
    :return: 添加预测列后的数据框
    """
    df_copy = df.copy()
    x_cols = best_model["used_x_cols"]

    # 1. 检查特征列是否齐全
    missing_cols = [col for col in x_cols if col not in df_copy.columns]
    if missing_cols:
        raise ValueError(f"数据缺失特征列：{missing_cols}")

    # 2. 处理特征缺失值
    X = df_copy[x_cols].copy()
    for col in x_cols:
        if X[col].isna().any():
            X[col].fillna(global_means[col], inplace=True)

    # 3. 计算预测值（手动计算，避免模型对象依赖）
    intercept = best_model["intercept"]
    coefficients = best_model["coefficients"]
    prediction = intercept + sum(coefficients[col] * X[col] for col in x_cols)

    # 4. 赋值到结果列（缺失特征的行预测值为NaN）
    df_copy[pred_col_name] = prediction
    df_copy[pred_col_name] = df_copy[pred_col_name].where(X.notna().all(axis=1), np.nan)

    return df_copy


def save_cv_fold_results(cv_folds, best_model, global_means, output_folder):
    """
    保存5折交叉验证训练集的原始数据+最优模型预测结果
    :param cv_folds: 折名-数据框字典
    :param best_model: 最优模型参数
    :param global_means: 全局特征均值
    :param output_folder: 输出文件夹路径
    """
    start_time = time.time()
    os.makedirs(output_folder, exist_ok=True)

    print(f"\n开始保存5折训练集结果（输出路径：{output_folder}）")
    for fold_name, df in cv_folds.items():
        # 用最优模型预测
        df_with_pred = predict_with_best_model(df, best_model, global_means)
        # 保存文件
        output_path = os.path.join(output_folder, f"{fold_name}_带预测结果.xlsx")
        df_with_pred.to_excel(output_path, index=False, engine='openpyxl')
        print(f"已保存：{os.path.basename(output_path)}（{len(df_with_pred)}条数据）")

    print(f"5折训练集结果保存完成（耗时: {time.time() - start_time:.2f}秒）")


def save_independent_validation(indep_val_path, best_model, global_means, output_folder):
    """
    保存独立验证集的原始数据+最优模型预测结果
    :param indep_val_path: 独立验证集文件路径
    :param best_model: 最优模型参数
    :param global_means: 全局特征均值
    :param output_folder: 输出文件夹路径
    """
    start_time = time.time()
    os.makedirs(output_folder, exist_ok=True)

    # 加载独立验证集
    if not os.path.exists(indep_val_path):
        raise FileNotFoundError(f"独立验证文件不存在：{indep_val_path}")
    df = pd.read_excel(indep_val_path, engine='openpyxl')

    # 检查必要列
    required_cols = ['COMID', 'DOY', 'lat', 'lon', 'Mean_Value', 'Slope', 'Aspect',
                     'AT_mean', 'Evaporation_mean', 'DSR', 'LWDN', 'LAI_mean', 'LST_mean', 'date', 'temp']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"独立验证集缺失必要列：{missing_cols}")

    # 数据预处理
    df['COMID'] = df['COMID'].astype(str)
    df['date'] = pd.to_datetime(df['date']).dt.date
    df = df.dropna(subset=['temp'])  # 删除水温缺失行

    # 用最优模型预测
    df_with_pred = predict_with_best_model(df, best_model, global_means)

    # 计算独立验证指标
    valid_df = df_with_pred.dropna(subset=['temp', 'multiple_RWT_spatem'])
    if len(valid_df) > 0:
        rmse = np.sqrt(mean_squared_error(valid_df['temp'], valid_df['multiple_RWT_spatem']))
        r2 = r2_score(valid_df['temp'], valid_df['multiple_RWT_spatem'])
        # 添加指标汇总行
        summary_row = pd.Series({
            'COMID': '独立验证汇总',
            'date': '—',
            'temp': f'RMSE: {rmse:.2f}',
            'multiple_RWT_spatem': f'R²: {r2:.2f}',
            **{col: '—' for col in required_cols[:-1] if col not in ['COMID', 'date', 'temp']}
        })
        df_with_pred = pd.concat([df_with_pred, summary_row.to_frame().T], ignore_index=True)
        print(f"\n独立验证结果：RMSE={rmse:.2f}, R²={r2:.2f}（{len(valid_df)}条有效数据）")

    # 保存结果
    output_path = os.path.join(output_folder, "独立验证集_带预测结果.xlsx")
    df_with_pred.to_excel(output_path, index=False, engine='openpyxl')
    print(f"独立验证集结果保存完成（路径：{output_path}，耗时: {time.time() - start_time:.2f}秒）")


def reconstruct_spatiotemporal(input_folder, best_model, global_means, output_folder):
    """
    用最优模型对整个时空数据（按日期命名的xlsx文件）进行重建，添加预测列
    :param input_folder: 输入文件夹（含"YYYY-MM-DD.xlsx"文件）
    :param best_model: 最优模型参数
    :param global_means: 全局特征均值
    :param output_folder: 输出文件夹路径
    """
    start_time = time.time()
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有日期文件
    input_files = [f for f in os.listdir(input_folder) if f.endswith('.xlsx')]
    if not input_files:
        raise FileNotFoundError(f"时空数据文件夹 {input_folder} 中无xlsx文件")

    # 检查特征列
    x_cols = best_model["used_x_cols"]
    success_count = 0
    failure_count = 0
    failed_files = []

    print(f"\n开始整个时空数据重建（共{len(input_files)}个日期文件，特征列：{x_cols}）")
    for i, file in enumerate(input_files, 1):
        file_path = os.path.join(input_folder, file)
        file_name = os.path.basename(file)

        # 进度提示
        if i % 50 == 0:
            print(f"  进度：{i}/{len(input_files)} 文件（{i / len(input_files):.1%}）")

        try:
            # 1. 读取日期文件（需包含COMID和所有特征列）
            df = pd.read_excel(file_path, engine='openpyxl')

            # 2. 检查必要列
            required_cols = x_cols + ['COMID']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"缺失必要列：{missing_cols}")

            # 3. 数据预处理
            df['COMID'] = df['COMID'].astype(str)

            # 4. 用最优模型预测（添加multiple_RWT_spatem列）
            df_with_pred = predict_with_best_model(df, best_model, global_means)

            # 5. 保存结果
            output_path = os.path.join(output_folder, file_name)
            df_with_pred.to_excel(output_path, index=False, engine='openpyxl')

            success_count += 1
        except Exception as e:
            failure_count += 1
            failed_files.append({"file": file_name, "reason": str(e)})
            continue

    # 输出重建汇总
    print(f"\n{'=' * 50}")
    print("整个时空数据重建完成：")
    print(f"总文件数：{len(input_files)}")
    print(f"成功：{success_count} 个，失败：{failure_count} 个")
    if failed_files:
        print(f"失败文件示例（前10个）：")
        for item in failed_files[:10]:
            print(f"  - {item['file']}: {item['reason'][:50]}...")
    print(f"输出路径：{output_folder}")
    print(f"耗时: {time.time() - start_time:.2f}秒")
    print(f"{'=' * 50}")


def main():
    # 计时开始
    overall_start = time.time()

    # --------------------------
    # 1. 参数配置（请根据你的实际路径修改！）
    # --------------------------
    # 5折交叉验证训练集文件路径（5个文件）
    CV_FOLD_PATHS = [
        r"E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\old\\result-0805\\空间交叉验证_训练集样本_第1折.xlsx",
        r"E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\old\\result-0805\\空间交叉验证_训练集样本_第2折.xlsx",
        r"E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\old\\result-0805\\空间交叉验证_训练集样本_第3折.xlsx",
        r"E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\old\\result-0805\\空间交叉验证_训练集样本_第4折.xlsx",
        r"E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\old\\result-0805\\空间交叉验证_训练集样本_第5折.xlsx"
    ]

    # 独立验证集文件路径
    INDEPENDENT_VAL_PATH = r"E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\old\\result-0805\\空间交叉验证_验证集样本.xlsx"

    # 时空重建输入文件夹（按日期命名的xlsx文件）
    SPATIOTEMPORAL_INPUT_FOLDER = r"E:\huai_river\Huairiver_GEE_data\Daily_data\old\11-test-2(fr_remove)"

    # 输出文件夹配置
    OUTPUT_ROOT = r"E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\multiple regression\\results_spatiotemporal"
    CV_FOLD_OUTPUT = os.path.join(OUTPUT_ROOT, "5折训练集结果")  # 5折训练集带预测结果
    INDEPENDENT_VAL_OUTPUT = os.path.join(OUTPUT_ROOT, "独立验证结果")  # 独立验证集带预测结果
    SPATIOTEMPORAL_OUTPUT = os.path.join(OUTPUT_ROOT, "时空重建结果")  # 整个时空重建结果

    # 特征列定义（与数据中的列名对应）
    X_COLS = ['DOY', 'lat', 'lon', 'Mean_Value', 'Slope', 'Aspect',
              'AT_mean', 'Evaporation_mean', 'DSR', 'LWDN', 'LAI_mean', 'LST_mean']

    try:
        # --------------------------
        # 2. 加载5折交叉验证数据
        # --------------------------
        print("===== 步骤1/5：加载交叉验证数据 =====")
        cv_folds, global_means = load_cv_folds(CV_FOLD_PATHS)

        # --------------------------
        # 3. 5折交叉验证训练，选择最优模型
        # --------------------------
        print("\n===== 步骤2/5：交叉验证训练 =====")
        best_model, cv_results = cross_validation_train(cv_folds, global_means, X_COLS)

        # --------------------------
        # 4. 保存5折训练集的预测结果
        # --------------------------
        print("\n===== 步骤3/5：保存训练集结果 =====")
        save_cv_fold_results(cv_folds, best_model, global_means, CV_FOLD_OUTPUT)

        # --------------------------
        # 5. 处理独立验证集并保存结果
        # --------------------------
        print("\n===== 步骤4/5：处理独立验证集 =====")
        save_independent_validation(INDEPENDENT_VAL_PATH, best_model, global_means, INDEPENDENT_VAL_OUTPUT)

        # --------------------------
        # 6. 整个时空数据重建
        # --------------------------
        print("\n===== 步骤5/5：时空数据重建 =====")
        reconstruct_spatiotemporal(SPATIOTEMPORAL_INPUT_FOLDER, best_model, global_means, SPATIOTEMPORAL_OUTPUT)

        # 总耗时统计
        total_time = time.time() - overall_start
        print(f"\n{'=' * 70}")
        print(f"所有流程完成！总耗时: {total_time:.2f}秒 ({total_time / 60:.1f}分钟)")
        print(f"结果输出根目录：{OUTPUT_ROOT}")
        print(f"1. 5折训练集结果：{CV_FOLD_OUTPUT}")
        print(f"2. 独立验证集结果：{INDEPENDENT_VAL_OUTPUT}")
        print(f"3. 时空重建结果：{SPATIOTEMPORAL_OUTPUT}")
        print(f"最优模型验证RMSE：{best_model['val_rmse']:.2f}")
        print(f"{'=' * 70}")

    except Exception as e:
        print(f"\n程序执行出错：{str(e)}")
        import traceback
        print("错误详情：")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
