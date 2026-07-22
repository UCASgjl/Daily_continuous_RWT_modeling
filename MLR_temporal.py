
# ============================================  改进  ==============================================================
# ===================================     未建模成功的日期在所有建模结束后print出原因    ======================================
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta
import time  # 用于计时


def find_lwdn_column(columns):
    """查找包含"LWDN"的列名（不区分大小写）"""
    for col in columns:
        if 'lwdn' in col.lower():
            return col
    return None


def prepare_fitting_validation_data(main_data_path, val_comid_path):
    """1. 拆分拟合数据和独立验证数据，保留Predicted_RWT_Space列"""
    start_time = time.time()

    # 使用更高效的引擎读取Excel，确保读取Predicted_RWT_Space列
    main_df = pd.read_excel(main_data_path, engine='openpyxl')

    # 检查并确保Predicted_RWT_Space列存在，不存在则创建空列
    if 'Predicted_RWT_Space' not in main_df.columns:
        main_df['Predicted_RWT_Space'] = np.nan
        print("警告：主数据中未找到'Predicted_RWT_Space'列，已创建空列")

    val_comid_df = pd.read_excel(val_comid_path, engine='openpyxl')
    val_comids = val_comid_df['COMID'].astype(str).tolist()

    # 查找LWDN列
    lwdn_col = find_lwdn_column(main_df.columns)
    if not lwdn_col:
        raise ValueError("主数据中未找到包含'LWDN'的列")

    # 关键列检查（动态包含LWDN列和Predicted_RWT_Space列）
    required_cols = ['COMID', 'date', 'temp', 'DOY', 'lat', 'lon', 'Mean_Value',
                     'Slope', 'Aspect', 'AT_mean', 'Evaporation_mean', 'DSR',
                     lwdn_col, 'LAI_mean', 'LST_mean', 'Predicted_RWT_Space']
    missing_cols = [col for col in required_cols if col not in main_df.columns]
    if missing_cols:
        raise ValueError(f"主数据缺失必要列：{missing_cols}")

    # 数据格式统一（只转换必要的列）
    main_df['COMID'] = main_df['COMID'].astype(str)
    main_df['date'] = pd.to_datetime(main_df['date']).dt.date

    # 拆分数据（使用isin加速）
    val_mask = main_df['COMID'].isin(val_comids)
    fit_df = main_df[~val_mask].copy()
    val_df = main_df[val_mask].copy()

    # 释放内存
    del main_df, val_mask

    print(f"数据拆分完成（耗时: {time.time() - start_time:.2f}秒）：")
    print(f"- 总数据量：{len(fit_df) + len(val_df)} 条")
    print(f"- 拟合数据量：{len(fit_df)} 条（{fit_df['COMID'].nunique()} 个COMID）")
    print(f"- 独立验证数据量：{len(val_df)} 条（{val_df['COMID'].nunique()} 个COMID）")
    print(f"- 检测到LWDN列：{lwdn_col}")

    return fit_df, val_df, val_comids, lwdn_col


def daily_multiple_regression(fit_df, x_cols, lwdn_col, start_date_str="2019-01-01", end_date_str="2020-12-31"):
    """2. 按日多元回归拟合（修正日期统计逻辑，基于完整日期范围）"""
    start_time = time.time()
    # 替换x_cols中的LWDN为实际找到的列名
    adjusted_x_cols = [col if col != 'LWDN' else lwdn_col for col in x_cols]

    daily_models = {}
    failed_dates = []  # 记录未建模成功的日期及原因

    # 生成2019-2020年完整日期范围（核心修正点）
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    all_dates = []
    current_date = start_date
    while current_date <= end_date:
        all_dates.append(current_date)
        current_date += timedelta(days=1)
    total_dates = len(all_dates)  # 总日期为731天（2019-2020）
    print(f"待建模的完整日期范围：{start_date} 至 {end_date}，共 {total_dates} 天")

    # 提取拟合数据中实际存在的日期
    existing_dates = set(fit_df['date'].unique())

    # 预计算所有特征列的全局均值（避免重复计算）
    global_means = {col: fit_df[col].mean() for col in adjusted_x_cols}

    for i, date in enumerate(all_dates, 1):
        # 进度提示（每100天显示一次）
        if i % 100 == 0:
            print(f"  处理进度：{i}/{total_dates} 天 ({i / total_dates:.1%})")

        # 检查该日期是否在拟合数据中存在
        if date not in existing_dates:
            reason = "拟合数据中无该日期的记录"
            failed_dates.append({"date": date, "reason": reason})
            continue

        # 使用布尔索引快速筛选
        date_mask = fit_df['date'] == date
        daily_fit_df = fit_df.loc[date_mask]

        # 1. 初步过滤：样本量至少3个
        if len(daily_fit_df) < 3:
            reason = f"样本量不足（仅{len(daily_fit_df)}条）"
            failed_dates.append({"date": date, "reason": reason})
            continue

        # 2. 提取X和Y，先删除Y（temp）缺失的样本
        Y_daily = daily_fit_df['temp'].dropna()
        if len(Y_daily) < 3:
            reason = f"有效温度样本不足（仅{len(Y_daily)}条）"
            failed_dates.append({"date": date, "reason": reason})
            continue

        # 只选择需要的列并对齐
        X_daily = daily_fit_df.loc[Y_daily.index, adjusted_x_cols].copy()

        # 3. 处理X的缺失值（优化版）
        # 一次性计算所有列的缺失情况
        missing_counts = X_daily.isna().sum()
        total_count = len(X_daily)

        # 标记需要删除的列
        cols_to_drop = []
        for col in adjusted_x_cols:
            missing_count = missing_counts[col]
            if missing_count == 0:
                continue

            missing_ratio = missing_count / total_count

            # 情况1：该列全为NaN → 删除列
            if missing_count == total_count:
                cols_to_drop.append(col)
                continue

            # 情况2：缺失率>50% → 删除列
            if missing_ratio > 0.5:
                cols_to_drop.append(col)
                continue

            # 情况3：缺失率≤50% → 全局均值填充（使用预计算的均值）
            X_daily[col].fillna(global_means[col], inplace=True)

        # 执行列删除
        if cols_to_drop:
            X_daily.drop(columns=cols_to_drop, inplace=True)

        # 4. 二次过滤：无特征列或样本量不足3个
        if len(X_daily.columns) == 0:
            reason = "所有特征列因缺失值过多被删除"
            failed_dates.append({"date": date, "reason": reason})
            continue

        if len(X_daily) < 3:
            reason = f"处理后有效样本不足（仅{len(X_daily)}条）"
            failed_dates.append({"date": date, "reason": reason})
            continue

        # 5. 最终检查：X是否还有NaN
        if X_daily.isna().any().any():
            X_daily = X_daily.dropna()
            Y_daily = Y_daily.loc[X_daily.index]
            if len(X_daily) < 3:
                reason = f"删除NaN后样本不足（仅{len(X_daily)}条）"
                failed_dates.append({"date": date, "reason": reason})
                continue

        # 6. 建模并保存结果
        model = LinearRegression(fit_intercept=True, n_jobs=-1)  # 使用所有可用CPU
        model.fit(X_daily, Y_daily)

        # 计算拟合指标
        Y_pred = model.predict(X_daily)
        rmse = np.sqrt(mean_squared_error(Y_daily, Y_pred))
        r2 = r2_score(Y_daily, Y_pred)

        # 保存参数（只保存必要信息，不保存完整模型）
        params = {'intercept': model.intercept_}
        for col, coef in zip(X_daily.columns, model.coef_):
            params[col] = coef

        daily_models[date] = {
            'used_x_cols': X_daily.columns.tolist(),
            'params': params,
            'metrics': {'RMSE': rmse, 'R²': r2, 'sample_count': len(X_daily)}
        }

        # 减少打印频率
        if i % 50 == 0:
            print(
                f"{date} - 建模成功 | 样本数：{len(X_daily)} | 用X列数：{len(X_daily.columns)} | RMSE：{rmse:.2f} | R²：{r2:.2f}")

    # 打印未建模成功的日期及原因
    print(
        f"\n按日建模完成（耗时: {time.time() - start_time:.2f}秒）：共成功建模 {len(daily_models)} 天（总日期 {total_dates} 天）")
    print(f"未成功建模的日期共 {len(failed_dates)} 天，原因分类统计如下：")
    # 按原因分类统计
    reason_counts = {}
    for item in failed_dates:
        reason_counts[item['reason']] = reason_counts.get(item['reason'], 0) + 1
    for reason, count in reason_counts.items():
        print(f"  - {reason}：{count} 天")

    # 如需查看具体日期，可取消下面注释
    # print("\n未成功建模的具体日期及原因：")
    # for item in failed_dates:
    #     print(f"  - {item['date']}: {item['reason']}")

    return daily_models, adjusted_x_cols, failed_dates


def reconstruct_daily_rwt(daily_models, failed_dates, input_x_folder, output_rwt_folder, x_cols, fit_df, lwdn_col):
    """3. 每日水温重建（保留Predicted_RWT_Space列）"""
    start_time = time.time()
    os.makedirs(output_rwt_folder, exist_ok=True)

    # 获取并排序所有输入文件
    input_files = [f for f in os.listdir(input_x_folder) if f.endswith('.xlsx')]
    if not input_files:
        raise FileNotFoundError(f"输入X文件夹 {input_x_folder} 中未找到任何xlsx文件")

    # 预计算所有特征列的全局均值
    global_means = {col: fit_df[col].mean() for col in x_cols if col in fit_df.columns}

    success_count = 0
    failure_count = 0
    total_files = len(input_files)
    failed_files_reasons = []  # 记录处理失败的文件及原因

    for i, file in enumerate(input_files, 1):
        # 进度提示
        if i % 50 == 0:
            print(f"  重建进度：{i}/{total_files} 文件 ({i / total_files:.1%})")

        # 解析文件名中的日期
        try:
            file_date_str = os.path.splitext(file)[0]
            file_date = datetime.strptime(file_date_str, '%Y-%m-%d').date()
        except ValueError:
            reason = "文件名不符合'YYYY-MM-DD.xlsx'格式"
            failed_files_reasons.append({"file": file, "reason": reason})
            failure_count += 1
            continue

        # 检查该日是否有可用模型
        if file_date not in daily_models:
            # 查找未建模原因
            fail_reason = next((item['reason'] for item in failed_dates if item['date'] == file_date),
                               "无可用模型（原因未知）")
            reason = f"该日期未建模：{fail_reason}"
            failed_files_reasons.append({"file": file, "reason": reason})
            failure_count += 1
            continue

        try:
            # 读取当天的X数据（包含Predicted_RWT_Space列）
            input_file_path = os.path.join(input_x_folder, file)
            # 获取需要的列（包含Predicted_RWT_Space）
            model_info = daily_models[file_date]
            required_cols = model_info['used_x_cols'] + ['COMID', 'Predicted_RWT_Space']

            # 只读取必要的列，提高速度并减少内存使用
            x_df = pd.read_excel(input_file_path, usecols=required_cols, engine='openpyxl')
            x_df['COMID'] = x_df['COMID'].astype(str)

            # 检查缺失的列（允许Predicted_RWT_Space缺失，缺失时创建空列）
            missing_used_cols = [col for col in model_info['used_x_cols'] if col not in x_df.columns]
            if missing_used_cols:
                reason = f"缺失必要特征列：{missing_used_cols}"
                failed_files_reasons.append({"file": file, "reason": reason})
                failure_count += 1
                continue

            # 如果原始文件中没有Predicted_RWT_Space列，创建一个空列
            if 'Predicted_RWT_Space' not in x_df.columns:
                x_df['Predicted_RWT_Space'] = np.nan

            # 只保留模型训练时使用的特征列用于预测
            X_recon = x_df[model_info['used_x_cols']].copy()

            # 处理X数据的缺失值（使用预计算的均值）
            for col in model_info['used_x_cols']:
                if X_recon[col].isna().any() and col in global_means:
                    X_recon[col].fillna(global_means[col], inplace=True)

            # 最终检查：删除仍含NaN的行
            initial_count = len(X_recon)
            X_recon = X_recon.dropna()
            if len(X_recon) == 0:
                reason = "所有样本处理后仍含NaN"
                failed_files_reasons.append({"file": file, "reason": reason})
                failure_count += 1
                continue

            # 使用参数手动计算预测值
            params = model_info['params']
            intercept = params['intercept']
            prediction = intercept

            for col in model_info['used_x_cols']:
                if col in params:
                    prediction += params[col] * X_recon[col]

            # 保存结果（保留Predicted_RWT_Space列）
            x_df.loc[X_recon.index, 'Multiple_RWT'] = prediction
            x_df['Multiple_RWT'] = x_df['Multiple_RWT'].fillna(np.nan)

            # 保存重建结果，确保包含Predicted_RWT_Space列
            output_file_path = os.path.join(output_rwt_folder, file)
            x_df.to_excel(output_file_path, index=False, engine='openpyxl')
            success_count += 1

        except Exception as e:
            reason = f"处理错误：{str(e)}"
            failed_files_reasons.append({"file": file, "reason": reason})
            failure_count += 1
            continue

    print(
        f"\n每日水温重建完成（耗时: {time.time() - start_time:.2f}秒）：共处理 {total_files} 个文件，成功 {success_count} 个，失败 {failure_count} 个")

    # 打印文件处理失败的原因（如果有）
    if failed_files_reasons:
        print("处理失败的文件及原因：")
        for item in failed_files_reasons:
            print(f"  - {item['file']}: {item['reason']}")

    return success_count, failure_count, failed_files_reasons


def generate_summary_results(fit_df, val_df, daily_models, x_cols, output_summary_folder, lwdn_col):
    """4. 生成汇总结果（保留Predicted_RWT_Space列）"""
    start_time = time.time()
    os.makedirs(output_summary_folder, exist_ok=True)

    # 预计算全局均值
    global_means = {col: fit_df[col].mean() for col in x_cols if col in fit_df.columns}

    # --------------------------
    # 4.1 拟合数据汇总（原始+重建RWT，保留Predicted_RWT_Space列）
    # --------------------------
    # 复制原始数据，包含Predicted_RWT_Space列
    fit_recon_df = fit_df.copy()
    fit_recon_df['Multiple_RWT'] = np.nan  # 添加新的重建结果列

    # 按日期处理
    for date in daily_models.keys():
        date_mask = fit_recon_df['date'] == date
        if not date_mask.any():
            continue

        try:
            model_info = daily_models[date]
            used_x_cols = model_info['used_x_cols']

            # 提取当天有效样本
            Y_fit = fit_recon_df.loc[date_mask, 'temp'].dropna()
            if len(Y_fit) < 1:
                continue

            X_fit = fit_recon_df.loc[Y_fit.index, used_x_cols].copy()

            # 处理X缺失值
            for col in used_x_cols:
                if X_fit[col].isna().any() and col in global_means:
                    X_fit[col].fillna(global_means[col], inplace=True)

            # 过滤无效样本
            X_fit = X_fit.dropna()
            Y_fit = Y_fit.loc[X_fit.index]
            if len(X_fit) == 0:
                continue

            # 手动计算预测值
            params = model_info['params']
            intercept = params['intercept']
            prediction = intercept

            for col in used_x_cols:
                if col in params:
                    prediction += params[col] * X_fit[col]

            fit_recon_df.loc[X_fit.index, 'Multiple_RWT'] = prediction

        except Exception as e:
            print(f"处理拟合数据 {date} 时出错：{str(e)}")
            continue

    fit_summary_path = os.path.join(output_summary_folder, "拟合数据_原始+重建RWT.xlsx")
    fit_recon_df.to_excel(fit_summary_path, index=False, engine='openpyxl')
    print(f"已保存拟合数据汇总：{fit_summary_path}（{len(fit_recon_df)} 条记录，包含Predicted_RWT_Space列）")

    # --------------------------
    # 4.2 独立验证数据汇总（保留Predicted_RWT_Space列）
    # --------------------------
    # 复制原始数据，包含Predicted_RWT_Space列
    val_recon_df = val_df.copy()
    val_recon_df['Multiple_RWT'] = np.nan  # 添加新的重建结果列

    for date in daily_models.keys():
        date_mask = val_recon_df['date'] == date
        if not date_mask.any():
            continue

        try:
            model_info = daily_models[date]
            used_x_cols = model_info['used_x_cols']

            # 提取当天有效样本
            Y_val = val_recon_df.loc[date_mask, 'temp'].dropna()
            if len(Y_val) < 1:
                continue

            X_val = val_recon_df.loc[Y_val.index, used_x_cols].copy()

            # 处理X缺失值
            for col in used_x_cols:
                if X_val[col].isna().any() and col in global_means:
                    X_val[col].fillna(global_means[col], inplace=True)

            # 过滤无效样本
            X_val = X_val.dropna()
            Y_val = Y_val.loc[X_val.index]
            if len(X_val) == 0:
                continue

            # 手动计算预测值
            params = model_info['params']
            intercept = params['intercept']
            prediction = intercept

            for col in used_x_cols:
                if col in params:
                    prediction += params[col] * X_val[col]

            val_recon_df.loc[X_val.index, 'Multiple_RWT'] = prediction

        except Exception as e:
            print(f"处理验证数据 {date} 时出错：{str(e)}")
            continue

    # 计算验证指标
    val_valid_df = val_recon_df.dropna(subset=['temp', 'Multiple_RWT'])
    val_metrics = {"RMSE": np.nan, "R²": np.nan}
    if len(val_valid_df) > 0:
        val_metrics['RMSE'] = np.sqrt(mean_squared_error(val_valid_df['temp'], val_valid_df['Multiple_RWT']))
        val_metrics['R²'] = r2_score(val_valid_df['temp'], val_valid_df['Multiple_RWT'])
        print(
            f"\n独立验证结果：RMSE = {val_metrics['RMSE']:.2f}, R² = {val_metrics['R²']:.2f}（{len(val_valid_df)} 条有效数据）")

        # 添加指标汇总行，处理Predicted_RWT_Space列
        summary_row = pd.Series({
            'COMID': '验证汇总指标',
            'date': '—',
            'temp': f'RMSE: {val_metrics["RMSE"]:.2f}',
            'Multiple_RWT': f'R²: {val_metrics["R²"]:.2f}',
            'Predicted_RWT_Space': '—',
            **{col: '—' for col in x_cols if
               col in val_recon_df.columns and col not in ['COMID', 'date', 'temp', 'Multiple_RWT',
                                                           'Predicted_RWT_Space']}
        })
        val_recon_df = pd.concat([val_recon_df, summary_row.to_frame().T], ignore_index=True)

    val_summary_path = os.path.join(output_summary_folder, "独立验证数据_原始+重建RWT.xlsx")
    val_recon_df.to_excel(val_summary_path, index=False, engine='openpyxl')
    print(f"已保存独立验证数据汇总：{val_summary_path}（包含Predicted_RWT_Space列）")

    # --------------------------
    # 4.3 每日模型参数汇总
    # --------------------------
    model_summary = []
    for date, model_info in daily_models.items():
        row = {
            'date': date,
            'sample_count': model_info['metrics']['sample_count'],
            'used_x_cols': ', '.join(model_info['used_x_cols']),
            'RMSE': model_info['metrics']['RMSE'],
            'R²': model_info['metrics']['R²'], **model_info['params']
        }
        # 补全未使用的X列
        for col in x_cols:
            if col not in row:
                row[col] = np.nan
        model_summary.append(row)

    model_summary_df = pd.DataFrame(model_summary)
    # 调整列顺序
    base_cols = ['date', 'sample_count', 'used_x_cols', 'RMSE', 'R²', 'intercept']
    param_cols = [col for col in x_cols if col in model_summary_df.columns]
    model_summary_df = model_summary_df[base_cols + param_cols]

    model_summary_path = os.path.join(output_summary_folder, "每日多元回归模型参数.xlsx")
    model_summary_df.to_excel(model_summary_path, index=False, engine='openpyxl')
    print(f"已保存每日模型参数汇总：{model_summary_path}")
    print(f"汇总结果生成完成（耗时: {time.time() - start_time:.2f}秒）")




def main():
    # 计时开始
    overall_start = time.time()

    # 参数设置
    MAIN_DATA_PATH = r"E:\\huai_river\\Station_RWT\\Extracted_matching_with_temp.xlsx"
    VAL_COMID_PATH = r"E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\multiple regression\\results_test_validation\\validation_COMID.xlsx"
    INPUT_X_FOLDER = r"E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\old\\11-test-2(fr_remove)"
    OUTPUT_RWT_FOLDER = r"E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\multiple regression\\results_daily_1002-3"
    OUTPUT_SUMMARY_FOLDER = r"E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\multiple regression\\results_test_validation-3"

    # 定义X变量列
    X_COLS = ['DOY', 'lat', 'lon', 'Mean_Value', 'Slope', 'Aspect',
              'AT_mean', 'Evaporation_mean', 'DSR', 'LWDN', 'LAI_mean', 'LST_mean']

    try:
        # 1. 拆分拟合数据和独立验证数据
        print("1. 开始数据准备...")
        fit_df, val_df, val_comids, lwdn_col = prepare_fitting_validation_data(MAIN_DATA_PATH, VAL_COMID_PATH)

        # 2. 按日进行多元回归拟合
        print("\n2. 开始按日多元回归拟合...")
        daily_models, adjusted_x_cols, failed_dates = daily_multiple_regression(fit_df, X_COLS, lwdn_col)

        # 3. 重建每日所有COMID的水温
        print("\n3. 开始每日水温重建...")
        success_count, failure_count, failed_files_reasons = reconstruct_daily_rwt(
            daily_models, failed_dates, INPUT_X_FOLDER, OUTPUT_RWT_FOLDER, adjusted_x_cols, fit_df, lwdn_col)

        # 4. 生成汇总结果
        print("\n4. 开始生成汇总结果...")
        generate_summary_results(fit_df, val_df, daily_models, adjusted_x_cols, OUTPUT_SUMMARY_FOLDER, lwdn_col)

        # 总耗时
        total_time = time.time() - overall_start
        print("\n" + "=" * 70)
        print(f"所有流程执行完成！总耗时: {total_time:.2f}秒 ({total_time / 60:.1f}分钟)")
        print(f"1. 每日重建结果：{OUTPUT_RWT_FOLDER}")
        print(f"   - 成功生成 {success_count} 个文件")
        print(f"   - 失败 {failure_count} 个文件")
        print(f"2. 汇总结果：{OUTPUT_SUMMARY_FOLDER}")
        print(f"   - 拟合数据汇总包含Predicted_RWT_Space列")
        print(f"   - 验证数据汇总包含Predicted_RWT_Space列")
        print(f"3. 使用的LWDN列：{lwdn_col}")
        print("=" * 70)

    except Exception as e:
        print(f"\n程序执行出错：{str(e)}")
        import traceback
        print("错误详情：")
        print(traceback.format_exc())

#
# if __name__ == "__main__":
#     main()






# ======================================================= 多元回归-空间，改为交叉验证      ====================================
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta
import time  # 用于计时


def find_lwdn_column(columns):
    """查找包含"LWDN"的列名（不区分大小写）"""
    for col in columns:
        if 'lwdn' in col.lower():
            return col
    return None


def load_multiple_main_data(main_data_paths):
    """加载多个主数据Excel文件，合并为一个DataFrame（用于空间交叉验证）"""
    start_time = time.time()
    all_dfs = []
    file_names = [os.path.basename(path) for path in main_data_paths]

    for path, name in zip(main_data_paths, file_names):
        if not os.path.exists(path):
            raise FileNotFoundError(f"主数据文件不存在：{path}")

        # 读取单个文件，确保包含核心列
        df = pd.read_excel(path, engine='openpyxl')

        # 添加文件标识列，用于后续交叉验证分组
        df['file_identifier'] = name

        # 检查并添加Predicted_RWT_Space列（若缺失）
        if 'Predicted_RWT_Space' not in df.columns:
            df['Predicted_RWT_Space'] = np.nan
            print(f"警告：文件 {name} 中未找到'Predicted_RWT_Space'列，已创建空列")

        all_dfs.append(df)

    # 合并所有数据，重置索引
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # 统一关键列格式
    combined_df['COMID'] = combined_df['COMID'].astype(str)
    combined_df['date'] = pd.to_datetime(combined_df['date']).dt.date

    # 检查必要列（动态包含LWDN列）
    lwdn_col = find_lwdn_column(combined_df.columns)
    if not lwdn_col:
        raise ValueError("合并后的数据中未找到包含'LWDN'的列")

    required_cols = ['COMID', 'date', 'temp', 'DOY', 'lat', 'lon', 'Mean_Value',
                     'Slope', 'Aspect', 'AT_mean', 'Evaporation_mean', 'DSR',
                     lwdn_col, 'LAI_mean', 'LST_mean', 'Predicted_RWT_Space', 'file_identifier']
    missing_cols = [col for col in required_cols if col not in combined_df.columns]
    if missing_cols:
        raise ValueError(f"合并后的数据缺失必要列：{missing_cols}")

    print(f"加载并合并{len(main_data_paths)}个主数据文件完成（耗时: {time.time() - start_time:.2f}秒）：")
    print(f"- 合并后总数据量：{len(combined_df)} 条")
    print(f"- 包含日期范围：{combined_df['date'].min()} 至 {combined_df['date'].max()}")
    print(f"- 检测到LWDN列：{lwdn_col}")

    return combined_df, lwdn_col, file_names


def daily_spatial_cv_regression(combined_df, x_cols, lwdn_col, file_names,
                                start_date_str="2019-01-01", end_date_str="2020-12-31"):
    """按日执行空间交叉验证（5折：4个文件训练，1个文件测试），选择RMSE最优模型"""
    start_time = time.time()
    # 替换x_cols中的LWDN为实际找到的列名
    adjusted_x_cols = [col if col != 'LWDN' else lwdn_col for col in x_cols]

    daily_best_models = {}  # 存储每日最优模型
    failed_dates = []  # 记录未建模成功的日期及原因
    n_folds = len(file_names)  # 折数等于文件数量

    # 生成完整日期范围
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    all_dates = []
    current_date = start_date
    while current_date <= end_date:
        all_dates.append(current_date)
        current_date += timedelta(days=1)
    total_dates = len(all_dates)
    print(f"待建模的完整日期范围：{start_date} 至 {end_date}，共 {total_dates} 天")

    # 提取合并数据中实际存在的日期
    existing_dates = set(combined_df['date'].unique())
    # 预计算所有特征列的全局均值（用于缺失值填充）
    global_means = {col: combined_df[col].mean() for col in adjusted_x_cols}

    # 遍历每个日期执行空间交叉验证
    for i, date in enumerate(all_dates, 1):
        # 进度提示（每50天显示一次）
        if i % 50 == 0:
            print(f"  处理进度：{i}/{total_dates} 天 ({i / total_dates:.1%})")

        # 检查该日期是否存在数据
        if date not in existing_dates:
            reason = "合并数据中无该日期的记录"
            failed_dates.append({"date": date, "reason": reason})
            continue

        # 筛选该日期的所有数据
        date_mask = combined_df['date'] == date
        daily_all_data = combined_df.loc[date_mask].copy()
        if len(daily_all_data) < 3:
            reason = f"该日期总样本量不足（仅{len(daily_all_data)}条）"
            failed_dates.append({"date": date, "reason": reason})
            continue

        # 检查是否包含所有文件的数据
        present_files = daily_all_data['file_identifier'].unique()
        missing_files = [f for f in file_names if f not in present_files]
        if missing_files:
            reason = f"该日期缺少文件数据：{missing_files}"
            failed_dates.append({"date": date, "reason": reason})
            continue

        # 存储该日期交叉验证组合的结果
        cv_results = []
        for test_file in file_names:
            # 划分训练集（n-1个文件）和测试集（1个文件）
            train_files = [f for f in file_names if f != test_file]
            train_mask = daily_all_data['file_identifier'].isin(train_files)
            test_mask = daily_all_data['file_identifier'] == test_file

            train_data = daily_all_data.loc[train_mask].copy()
            test_data = daily_all_data.loc[test_mask].copy()

            # 过滤训练集：样本量≥3且temp非空
            train_data = train_data.dropna(subset=['temp'])
            if len(train_data) < 3:
                cv_results.append({
                    "train_files": train_files,
                    "test_file": test_file,
                    "train_sample_count": len(train_data),
                    "test_sample_count": len(test_data),
                    "success": False,
                    "reason": f"训练集样本量不足（仅{len(train_data)}条）",
                    "test_rmse": np.inf
                })
                continue

            # 过滤测试集：样本量≥1且temp非空
            test_data = test_data.dropna(subset=['temp'])
            if len(test_data) < 1:
                cv_results.append({
                    "train_files": train_files,
                    "test_file": test_file,
                    "train_sample_count": len(train_data),
                    "test_sample_count": len(test_data),
                    "success": False,
                    "reason": "测试集无有效样本",
                    "test_rmse": np.inf
                })
                continue

            # 处理训练集特征缺失值
            X_train = train_data[adjusted_x_cols].copy()
            Y_train = train_data['temp']
            # 按列处理缺失值（缺失率>50%删除列，否则全局均值填充）
            cols_to_drop = []
            for col in adjusted_x_cols:
                missing_count = X_train[col].isna().sum()
                if missing_count == len(X_train):  # 全缺失→删除
                    cols_to_drop.append(col)
                    continue
                if missing_count / len(X_train) > 0.5:  # 缺失率>50%→删除
                    cols_to_drop.append(col)
                    continue
                # 缺失率≤50%→全局均值填充
                X_train[col].fillna(global_means[col], inplace=True)
            if cols_to_drop:
                X_train.drop(columns=cols_to_drop, inplace=True)
            # 二次过滤：特征列为空或样本不足
            if len(X_train.columns) == 0 or len(X_train) < 3:
                cv_results.append({
                    "train_files": train_files,
                    "test_file": test_file,
                    "train_sample_count": len(train_data),
                    "test_sample_count": len(test_data),
                    "success": False,
                    "reason": "训练集特征列不足或样本量不足",
                    "test_rmse": np.inf
                })
                continue

            # 处理测试集特征缺失值（使用训练集的列和全局均值）
            X_test = test_data[X_train.columns].copy()
            Y_test = test_data['temp']
            for col in X_test.columns:
                if X_test[col].isna().any():
                    X_test[col].fillna(global_means[col], inplace=True)
            # 过滤测试集NaN
            X_test = X_test.dropna()
            Y_test = Y_test.loc[X_test.index]
            if len(X_test) < 1:
                cv_results.append({
                    "train_files": train_files,
                    "test_file": test_file,
                    "train_sample_count": len(train_data),
                    "test_sample_count": len(test_data),
                    "success": False,
                    "reason": "测试集处理后无有效样本",
                    "test_rmse": np.inf
                })
                continue

            # 训练模型并评估
            model = LinearRegression(fit_intercept=True, n_jobs=-1)
            model.fit(X_train, Y_train)

            # 计算训练集指标
            Y_train_pred = model.predict(X_train)
            train_rmse = np.sqrt(mean_squared_error(Y_train, Y_train_pred))
            train_r2 = r2_score(Y_train, Y_train_pred)

            # 计算测试集指标
            Y_test_pred = model.predict(X_test)
            test_rmse = np.sqrt(mean_squared_error(Y_test, Y_test_pred))
            test_r2 = r2_score(Y_test, Y_test_pred)

            # 保存该交叉验证组合的结果
            cv_results.append({
                "train_files": train_files,
                "test_file": test_file,
                "train_sample_count": len(X_train),
                "test_sample_count": len(X_test),
                "used_x_cols": X_train.columns.tolist(),
                "model_params": {
                    "intercept": model.intercept_,
                    **{col: coef for col, coef in zip(X_train.columns, model.coef_)}
                },
                "train_metrics": {"RMSE": train_rmse, "R²": train_r2},
                "test_metrics": {"RMSE": test_rmse, "R²": test_r2},
                "success": True
            })

        # 从交叉验证组合中选择最优模型（测试集RMSE最小）
        valid_cv = [res for res in cv_results if res["success"]]
        if not valid_cv:
            reason = f"{n_folds}个交叉验证组合均建模失败"
            failed_dates.append({"date": date, "reason": reason})
            continue

        # 选择测试集RMSE最小的组合作为最优模型
        best_idx = np.argmin([res["test_metrics"]["RMSE"] for res in valid_cv])
        best_res = valid_cv[best_idx]

        # 保存每日最优模型信息（包含交叉验证详情）
        daily_best_models[date] = {
            "cv_details": cv_results,  # 所有交叉验证结果（便于追溯）
            "best_cv_index": best_idx,  # 最优组合的索引
            "used_x_cols": best_res["used_x_cols"],
            "params": best_res["model_params"],
            "train_metrics": best_res["train_metrics"],
            "test_metrics": best_res["test_metrics"],
            "train_sample_count": best_res["train_sample_count"],
            "test_sample_count": best_res["test_sample_count"],
            "best_train_files": best_res["train_files"],
            "best_test_file": best_res["test_file"]
        }

        # 打印最优模型信息（每50天显示一次）
        if i % 50 == 0:
            print(f"{date} - 最优模型 | 测试RMSE：{best_res['test_metrics']['RMSE']:.2f} | "
                  f"测试文件：{best_res['test_file']}")

    # 打印建模总结
    print(f"\n按日空间交叉验证建模完成（耗时: {time.time() - start_time:.2f}秒）：")
    print(f"共成功建模 {len(daily_best_models)} 天（总日期 {total_dates} 天）")
    print(f"未成功建模的日期共 {len(failed_dates)} 天，原因分类统计如下：")
    reason_counts = {}
    for item in failed_dates:
        reason_counts[item['reason']] = reason_counts.get(item['reason'], 0) + 1
    for reason, count in reason_counts.items():
        print(f"  - {reason}：{count} 天")

    return daily_best_models, adjusted_x_cols, failed_dates


def reconstruct_daily_rwt(daily_models, failed_dates, input_x_folder, output_rwt_folder, x_cols, combined_df, lwdn_col):
    """每日水温重建（保留Predicted_RWT_Space列，使用最优模型参数）"""
    start_time = time.time()
    os.makedirs(output_rwt_folder, exist_ok=True)

    # 获取输入文件并排序
    input_files = [f for f in os.listdir(input_x_folder) if f.endswith('.xlsx')]
    if not input_files:
        raise FileNotFoundError(f"输入X文件夹 {input_x_folder} 中未找到任何xlsx文件")

    # 预计算全局均值（用于缺失值填充）
    global_means = {col: combined_df[col].mean() for col in x_cols if col in combined_df.columns}

    success_count = 0
    failure_count = 0
    failed_files_reasons = []

    for i, file in enumerate(input_files, 1):
        if i % 50 == 0:
            print(f"  重建进度：{i}/{len(input_files)} 文件 ({i / len(input_files):.1%})")

        # 解析文件名中的日期
        try:
            file_date_str = os.path.splitext(file)[0]
            file_date = datetime.strptime(file_date_str, '%Y-%m-%d').date()
        except ValueError:
            reason = "文件名不符合'YYYY-MM-DD.xlsx'格式"
            failed_files_reasons.append({"file": file, "reason": reason})
            failure_count += 1
            continue

        # 检查该日是否有可用模型
        if file_date not in daily_models:
            # 查找未建模原因
            fail_reason = next((item['reason'] for item in failed_dates if item['date'] == file_date),
                               "无可用模型（原因未知）")
            reason = f"该日期未建模：{fail_reason}"
            failed_files_reasons.append({"file": file, "reason": reason})
            failure_count += 1
            continue

        try:
            # 读取当天的X数据（包含Predicted_RWT_Space列）
            input_file_path = os.path.join(input_x_folder, file)
            model_info = daily_models[file_date]
            required_cols = model_info['used_x_cols'] + ['COMID', 'Predicted_RWT_Space']

            # 只读取必要的列，提高速度
            x_df = pd.read_excel(input_file_path, usecols=required_cols, engine='openpyxl')
            x_df['COMID'] = x_df['COMID'].astype(str)

            # 检查缺失的特征列
            missing_used_cols = [col for col in model_info['used_x_cols'] if col not in x_df.columns]
            if missing_used_cols:
                reason = f"缺失必要特征列：{missing_used_cols}"
                failed_files_reasons.append({"file": file, "reason": reason})
                failure_count += 1
                continue

            # 确保Predicted_RWT_Space列存在
            if 'Predicted_RWT_Space' not in x_df.columns:
                x_df['Predicted_RWT_Space'] = np.nan

            # 准备预测用的特征数据
            X_recon = x_df[model_info['used_x_cols']].copy()

            # 处理缺失值
            for col in model_info['used_x_cols']:
                if X_recon[col].isna().any() and col in global_means:
                    X_recon[col].fillna(global_means[col], inplace=True)

            # 最终过滤仍含NaN的行
            initial_count = len(X_recon)
            X_recon = X_recon.dropna()
            if len(X_recon) == 0:
                reason = "所有样本处理后仍含NaN"
                failed_files_reasons.append({"file": file, "reason": reason})
                failure_count += 1
                continue

            # 使用最优模型参数计算预测值
            params = model_info['params']
            intercept = params['intercept']
            prediction = intercept

            for col in model_info['used_x_cols']:
                if col in params:
                    prediction += params[col] * X_recon[col]

            # 保存结果
            x_df.loc[X_recon.index, 'Multiple_RWT'] = prediction
            x_df['Multiple_RWT'] = x_df['Multiple_RWT'].fillna(np.nan)

            # 输出到文件
            output_file_path = os.path.join(output_rwt_folder, file)
            x_df.to_excel(output_file_path, index=False, engine='openpyxl')
            success_count += 1

        except Exception as e:
            reason = f"处理错误：{str(e)}"
            failed_files_reasons.append({"file": file, "reason": reason})
            failure_count += 1
            continue

    print(f"\n每日水温重建完成（耗时: {time.time() - start_time:.2f}秒）：")
    print(f"共处理 {len(input_files)} 个文件，成功 {success_count} 个，失败 {failure_count} 个")

    if failed_files_reasons:
        print("处理失败的文件及原因：")
        for item in failed_files_reasons[:5]:  # 只显示前5个
            print(f"  - {item['file']}: {item['reason']}")
        if len(failed_files_reasons) > 5:
            print(f"  ... 还有 {len(failed_files_reasons) - 5} 个失败文件未显示")

    return success_count, failure_count, failed_files_reasons


def generate_summary_results(combined_df, daily_models, x_cols, output_summary_folder, lwdn_col, file_names):
    """生成汇总结果（包含交叉验证信息）"""
    start_time = time.time()
    os.makedirs(output_summary_folder, exist_ok=True)

    # 预计算全局均值
    global_means = {col: combined_df[col].mean() for col in x_cols if col in combined_df.columns}

    # 1. 拟合与验证数据汇总（包含所有日期的预测结果）
    all_results_df = combined_df.copy()
    all_results_df['Multiple_RWT'] = np.nan  # 模型预测值
    all_results_df['CV_Group'] = np.nan  # 交叉验证分组（训练/测试）

    for date in daily_models.keys():
        date_mask = all_results_df['date'] == date
        if not date_mask.any():
            continue

        try:
            model_info = daily_models[date]
            used_x_cols = model_info['used_x_cols']
            best_train_files = model_info['best_train_files']

            # 标记交叉验证分组
            all_results_df.loc[
                date_mask & all_results_df['file_identifier'].isin(best_train_files), 'CV_Group'] = 'Train'
            all_results_df.loc[
                date_mask & (all_results_df['file_identifier'] == model_info['best_test_file']), 'CV_Group'] = 'Test'

            # 提取当天有效样本
            Y_actual = all_results_df.loc[date_mask, 'temp'].dropna()
            if len(Y_actual) < 1:
                continue

            X_data = all_results_df.loc[Y_actual.index, used_x_cols].copy()

            # 处理缺失值
            for col in used_x_cols:
                if X_data[col].isna().any() and col in global_means:
                    X_data[col].fillna(global_means[col], inplace=True)

            # 过滤无效样本
            X_data = X_data.dropna()
            Y_actual = Y_actual.loc[X_data.index]
            if len(X_data) == 0:
                continue

            # 计算预测值
            params = model_info['params']
            intercept = params['intercept']
            prediction = intercept

            for col in used_x_cols:
                if col in params:
                    prediction += params[col] * X_data[col]

            all_results_df.loc[X_data.index, 'Multiple_RWT'] = prediction

        except Exception as e:
            print(f"处理汇总数据 {date} 时出错：{str(e)}")
            continue

    # 保存完整结果
    all_results_path = os.path.join(output_summary_folder, "所有数据_原始+预测+交叉验证分组.xlsx")
    all_results_df.to_excel(all_results_path, index=False, engine='openpyxl')
    print(f"已保存完整结果：{all_results_path}（{len(all_results_df)} 条记录）")

    # 2. 每日最优模型参数汇总
    model_summary = []
    for date, model_info in daily_models.items():
        # 基础信息
        row = {
            'date': date,
            'best_train_files': ', '.join(model_info['best_train_files']),
            'best_test_file': model_info['best_test_file'],
            'train_sample_count': model_info['train_sample_count'],
            'test_sample_count': model_info['test_sample_count'],
            'used_x_cols': ', '.join(model_info['used_x_cols']),
            'train_RMSE': model_info['train_metrics']['RMSE'],
            'train_R²': model_info['train_metrics']['R²'],
            'test_RMSE': model_info['test_metrics']['RMSE'],
            'test_R²': model_info['test_metrics']['R²'], **model_info['params']
        }
        # 补全未使用的X列
        for col in x_cols:
            if col not in row:
                row[col] = np.nan
        model_summary.append(row)

    model_summary_df = pd.DataFrame(model_summary)
    # 调整列顺序
    base_cols = ['date', 'best_train_files', 'best_test_file', 'train_sample_count',
                 'test_sample_count', 'used_x_cols', 'train_RMSE', 'train_R²',
                 'test_RMSE', 'test_R²', 'intercept']
    param_cols = [col for col in x_cols if col in model_summary_df.columns]
    model_summary_df = model_summary_df[base_cols + param_cols]

    model_summary_path = os.path.join(output_summary_folder, "每日最优模型参数及交叉验证结果.xlsx")
    model_summary_df.to_excel(model_summary_path, index=False, engine='openpyxl')
    print(f"已保存每日模型参数汇总：{model_summary_path}")

    # 3. 交叉验证详细结果汇总
    cv_details_list = []
    for date, model_info in daily_models.items():
        for cv_idx, cv_res in enumerate(model_info['cv_details']):
            cv_row = {
                'date': date,
                'cv_index': cv_idx,
                'is_best_model': (cv_idx == model_info['best_cv_index']),
                'train_files': ', '.join(cv_res['train_files']),
                'test_file': cv_res['test_file'],
                'train_sample_count': cv_res['train_sample_count'],
                'test_sample_count': cv_res['test_sample_count'],
                'success': cv_res['success'],
                'reason': cv_res.get('reason', ''),
                'train_RMSE': cv_res['train_metrics']['RMSE'] if cv_res['success'] else np.nan,
                'train_R²': cv_res['train_metrics']['R²'] if cv_res['success'] else np.nan,
                'test_RMSE': cv_res['test_metrics']['RMSE'] if cv_res['success'] else np.nan,
                'test_R²': cv_res['test_metrics']['R²'] if cv_res['success'] else np.nan,
                'used_x_cols': ', '.join(cv_res['used_x_cols']) if cv_res['success'] else ''
            }
            cv_details_list.append(cv_row)

    cv_details_df = pd.DataFrame(cv_details_list)
    cv_details_path = os.path.join(output_summary_folder, "交叉验证详细结果.xlsx")
    cv_details_df.to_excel(cv_details_path, index=False, engine='openpyxl')
    print(f"已保存交叉验证详情：{cv_details_path}")

    print(f"汇总结果生成完成（耗时: {time.time() - start_time:.2f}秒）")


def main():
    # 计时开始
    overall_start = time.time()

    # 参数设置 - 请修改为你的5个xlsx文件路径
    MAIN_DATA_PATHS = [
        r"E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\old\\result-0805\\空间交叉验证_训练集样本_第1折.xlsx",
        r"E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\old\\result-0805\\空间交叉验证_训练集样本_第2折.xlsx",
        r"E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\old\\result-0805\\空间交叉验证_训练集样本_第3折.xlsx",
        r"E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\old\\result-0805\\空间交叉验证_训练集样本_第4折.xlsx",
        r"E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\old\\result-0805\\空间交叉验证_训练集样本_第5折.xlsx"
    ]

    # 其他路径设置
    INPUT_X_FOLDER = r"E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\old\\11-test-2(fr_remove)"
    OUTPUT_RWT_FOLDER = r"E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\multiple regression\\results_spatial\\results_daily"
    OUTPUT_SUMMARY_FOLDER = r"E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\multiple regression\\results_spatial\\results_test_validation"

    # 定义X变量列
    X_COLS = ['DOY', 'lat', 'lon', 'Mean_Value', 'Slope', 'Aspect',
              'AT_mean', 'Evaporation_mean', 'DSR', 'LWDN', 'LAI_mean', 'LST_mean']

    try:
        # 1. 加载并合并多个主数据文件
        print("1. 开始加载并合并主数据文件...")
        combined_df, lwdn_col, file_names = load_multiple_main_data(MAIN_DATA_PATHS)

        # 2. 按日进行空间交叉验证多元回归拟合
        print("\n2. 开始按日空间交叉验证建模...")
        daily_models, adjusted_x_cols, failed_dates = daily_spatial_cv_regression(
            combined_df, X_COLS, lwdn_col, file_names)

        # 3. 重建每日所有COMID的水温
        print("\n3. 开始每日水温重建...")
        success_count, failure_count, failed_files_reasons = reconstruct_daily_rwt(
            daily_models, failed_dates, INPUT_X_FOLDER, OUTPUT_RWT_FOLDER, adjusted_x_cols, combined_df, lwdn_col)

        # 4. 生成汇总结果
        print("\n4. 开始生成汇总结果...")
        generate_summary_results(combined_df, daily_models, adjusted_x_cols,
                                 OUTPUT_SUMMARY_FOLDER, lwdn_col, file_names)

        # 总耗时
        total_time = time.time() - overall_start
        print("\n" + "=" * 70)
        print(f"所有流程执行完成！总耗时: {total_time:.2f}秒 ({total_time / 60:.1f}分钟)")
        print(f"1. 每日重建结果：{OUTPUT_RWT_FOLDER}")
        print(f"   - 成功生成 {success_count} 个文件")
        print(f"   - 失败 {failure_count} 个文件")
        print(f"2. 汇总结果：{OUTPUT_SUMMARY_FOLDER}")
        print(f"   - 包含所有数据的预测结果和交叉验证分组")
        print(f"   - 每日最优模型参数及交叉验证详情")
        print(f"3. 使用的LWDN列：{lwdn_col}")
        print("=" * 70)

    except Exception as e:
        print(f"\n程序执行出错：{str(e)}")
        import traceback
        print("错误详情：")
        print(traceback.format_exc())


# if __name__ == "__main__":
#     main()







# ================================   在  results_daily  中 导出 独立验证集  的数据    ===================================
import pandas as pd
import os
from datetime import datetime
from pathlib import Path
import pickle
import hashlib
from tqdm import tqdm


def create_file_index(folder_path, cache_file="file_index.cache"):
    """创建文件夹中Excel文件的索引，包含日期和其中的COMID数据"""
    # 计算文件夹内容的哈希值，用于判断是否需要重新生成索引
    folder_hash = hashlib.md5()
    for file in sorted(Path(folder_path).glob("*.xlsx")):
        if not file.name.startswith('~$'):
            folder_hash.update(str(file).encode())
            folder_hash.update(str(file.stat().st_mtime).encode())
    current_hash = folder_hash.hexdigest()

    # 检查缓存是否存在且有效
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached_hash, file_index = pickle.load(f)
            if cached_hash == current_hash:
                print("使用缓存的文件索引")
                return file_index
        except:
            pass  # 缓存损坏，重新生成

    # 创建新的文件索引
    file_index = {}
    excel_files = list(Path(folder_path).glob("*.xlsx"))

    print(f"正在创建文件索引，共发现 {len(excel_files)} 个Excel文件...")

    # 支持的日期格式列表，可根据实际情况调整
    date_formats = ['%Y%m%d', '%Y-%m-%d', '%m%d%Y', '%m-%d-%Y',
                    '%Y%m%d_%H%M%S', '%Y-%m-%d_%H-%M-%S']

    # 使用tqdm显示进度
    for file in tqdm(excel_files, desc="索引文件"):
        if file.name.startswith('~$'):  # 跳过临时文件
            continue

        # 提取文件名中的日期
        date_str = file.stem
        file_date = None

        for fmt in date_formats:
            try:
                file_date = datetime.strptime(date_str, fmt).date()
                break
            except ValueError:
                continue

        if file_date is None:
            continue  # 无法解析日期格式

        # 读取文件并提取必要数据
        try:
            # 只读取需要的列，提高速度
            # df = pd.read_excel(file, usecols=['COMID', 'Predicted_RWT_Space', 'Multiple_RWT'])

            df = pd.read_excel(file, usecols=['COMID', 'predicted_temp'])

            # 确保COMID是字符串类型，避免匹配问题
            df['COMID'] = df['COMID'].astype(str)

            # 创建COMID到数据的映射
            comid_map = {}
            for _, row in df.iterrows():
                comid = str(row['COMID'])
                comid_map[comid] = {
                    # 'Predicted_RWT_Space': row['Predicted_RWT_Space'],
                    # 'Multiple_RWT': row['Multiple_RWT'],
                    'predicted_temp':row['predicted_temp']
                }

            # 将此文件的数据添加到索引
            if file_date not in file_index:
                file_index[file_date] = {}
            file_index[file_date].update(comid_map)

        except Exception as e:
            print(f"处理文件 {file.name} 时出错: {str(e)}")
            continue

    # 保存索引到缓存
    with open(cache_file, 'wb') as f:
        pickle.dump((current_hash, file_index), f)

    return file_index


def merge_excel_data(folder_path, input_file, output_file, cache_file="file_index.cache"):
    """主函数：读取输入文件，匹配数据并导出结果"""
    # 创建文件索引（会使用缓存）
    file_index = create_file_index(folder_path, cache_file)

    # 读取包含COMID和date的Excel文件
    try:
        input_df = pd.read_excel(input_file)
    except Exception as e:
        print(f"无法读取输入文件: {str(e)}")
        return

    # 检查必要的列是否存在
    if 'COMID' not in input_df.columns or 'date' not in input_df.columns:
        print("输入文件必须包含'COMID'和'date'列")
        return

    # 初始化新列
    # input_df['Predicted_RWT_Space'] = None
    # input_df['Multiple_RWT'] = None

    input_df['predicted_temp'] = None


    # 处理日期列，转换为日期对象
    try:
        # 尝试直接转换
        input_df['date_parsed'] = pd.to_datetime(input_df['date']).dt.date
    except:
        # 如果直接转换失败，尝试多种格式
        date_formats = ['%Y-%m-%d', '%Y%m%d', '%m-%d-%Y', '%m/%d/%Y']
        input_df['date_parsed'] = None

        for fmt in date_formats:
            mask = input_df['date_parsed'].isna()
            if mask.sum() == 0:
                break
            input_df.loc[mask, 'date_parsed'] = pd.to_datetime(
                input_df.loc[mask, 'date'],
                format=fmt,
                errors='ignore'
            ).dt.date

    # 确保COMID是字符串类型
    input_df['COMID_str'] = input_df['COMID'].astype(str)

    # 定义匹配函数
    def get_matching_values(row):
        date = row['date_parsed']
        comid = row['COMID_str']

        if pd.isna(date) or not comid:
            return pd.Series([None, None])

        # 查找匹配的数据
        date_data = file_index.get(date, {})
        comid_data = date_data.get(comid, {})

        return pd.Series([
            # comid_data.get('Predicted_RWT_Space'),
            # comid_data.get('Multiple_RWT')

            comid_data.get('predicted_temp')
        ])

    # 匹配数据（使用tqdm显示进度）
    print("正在匹配数据...")
    # 使用tqdm包装迭代过程
    tqdm.pandas()
    result = input_df.progress_apply(get_matching_values, axis=1)
    # input_df['Predicted_RWT_Space'] = result[0]
    # input_df['Multiple_RWT'] = result[1]

    input_df['predicted_temp'] = result[0]


    # 删除临时列
    input_df.drop(columns=['date_parsed', 'COMID_str'], inplace=True)

    # 导出结果到新的Excel文件
    try:
        input_df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"成功导出结果到 {output_file}")
    except Exception as e:
        print(f"导出文件时出错: {str(e)}")




if __name__ == "__main__":
    # 配置参数 - 请根据实际情况修改以下路径
    # FOLDER_PATH = "E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\multiple regression\\results_spatial\\results_daily"  # 包含日期命名的Excel文件的文件夹路径
    # INPUT_FILE = "E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\old\\result-0805\\空间交叉验证_验证集样本.xlsx"  # 包含COMID和date的Excel文件路径
    # OUTPUT_FILE = "E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\multiple regression\\results_spatial\\results_test_validation\\single_accuracy_valida.xlsx"  # 输出结果的Excel文件路径


    # FOLDER_PATH = "E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\XGBoost_spatial\\results_daily"  # 包含日期命名的Excel文件的文件夹路径
    # INPUT_FILE = "E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\old\\result-0805\\空间交叉验证_验证集样本.xlsx"  # 包含COMID和date的Excel文件路径
    # OUTPUT_FILE = "E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\XGBoost_spatial\\results_daily\\results_valida_data.xlsx"  # 输出结果的Excel文件路径

    FOLDER_PATH = "E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\XGBoost_spatial\\results_daily"  # 包含日期命名的Excel文件的文件夹路径
    INPUT_FILE = "E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\XGBoost_spatial\\results_train_data_multi_space.xlsx"  # 包含COMID和date的Excel文件路径
    OUTPUT_FILE = "E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\XGBoost_spatial\\results_daily\\results_train_data.xlsx"  # 输出结果的Excel文件路径

    # 执行合并操作
    # merge_excel_data(FOLDER_PATH, INPUT_FILE, OUTPUT_FILE)






# ==========================================   生成 两年 均值的 shp文件  =============================================
import geopandas as gpd
import pandas as pd
import os

# -------------------------- 1. 配置参数（需手动修改！）--------------------------
# EXCEL_FOLDER = r"E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\multiple regression\\results_daily_1002-4"  # 存放所有日期Excel的文件夹
# SHP_INPUT = r"E:\\huai_river\\Huai_river_basin\\Huairiver_River_fr_MERIT.shp"  # 原始汇流区SHP文件
# SHP_OUTPUT = r"E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\multiple regression\\SHP\\Avg_RWT.shp"  # 输出含均值的新SHP文件
# EXCEL_EXT = ".xlsx"  # Excel文件后缀




# EXCEL_FOLDER = r"E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\multiple regression\\results_spatiotemporal\\results_daily"  # 存放所有日期Excel的文件夹
# SHP_INPUT = r"E:\\huai_river\\Huai_river_basin\\Huairiver_River_fr_MERIT.shp"  # 原始汇流区SHP文件
# SHP_OUTPUT = r"E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\multiple regression\\SHP\\Multi_spacetime.shp"  # 输出含均值的新SHP文件
# EXCEL_EXT = ".xlsx"  # Excel文件后缀


EXCEL_FOLDER = r"E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\XGBoost_spatial\\results_daily"  # 存放所有日期Excel的文件夹
SHP_INPUT = r"E:\\huai_river\\Huai_river_basin\\Huairiver_River_fr_MERIT.shp"  # 原始汇流区SHP文件
SHP_OUTPUT = r"E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\multiple regression\\SHP\\XGBoost_space.shp"  # 输出含均值的新SHP文件
EXCEL_EXT = ".xlsx"  # Excel文件后缀


# -------------------------- 2. 读取所有Excel，计算每个COMID的Multiple_RWT均值 --------------------------
comid_rwt_dict = {}

for filename in os.listdir(EXCEL_FOLDER):
    if filename.endswith(EXCEL_EXT):
        excel_path = os.path.join(EXCEL_FOLDER, filename)
        print(f"正在读取：{filename}")

        # 读取Excel并过滤无效值   Multiple_RWT    multiple_RWT_spatem    predicted_temp
        df = pd.read_excel(excel_path, usecols=["COMID", "predicted_temp"], engine="openpyxl")
        df = df.dropna(subset=["COMID", "predicted_temp"])
        df = df[pd.to_numeric(df["predicted_temp"], errors="coerce").notna()]

        # 计算当前Excel中各COMID的均值
        comid_avg = df.groupby("COMID")["predicted_temp"].mean().reset_index()

        # 累计到字典
        for _, row in comid_avg.iterrows():
            comid = int(row["COMID"])
            avg_rwt = row["predicted_temp"]
            comid_rwt_dict[comid] = comid_rwt_dict.get(comid, []) + [avg_rwt]

# 计算最终均值
df_avg = pd.DataFrame([
    {"COMID": comid, "Avg_RWT": sum(rwts) / len(rwts)}
    for comid, rwts in comid_rwt_dict.items()
])
print(f"\n共计算 {len(df_avg)} 个COMID的均值")

# -------------------------- 3. 关联SHP文件，添加均值字段并输出 --------------------------
# 读取SHP文件（GeoDataFrame格式）
gdf = gpd.read_file(SHP_INPUT)

# 确保SHP的COMID为整数（与Excel匹配）
gdf["COMID"] = gdf["COMID"].astype(int)

# 左连接：将均值关联到SHP（保留SHP中所有COMID，无均值的设为NaN）
gdf_merged = gdf.merge(
    df_avg,
    on="COMID",
    how="left"
)

# 保存为新SHP文件
gdf_merged.to_file(SHP_OUTPUT, driver="ESRI Shapefile")
print(f"\n任务完成！新SHP文件已保存至：{SHP_OUTPUT}")



# ====================================