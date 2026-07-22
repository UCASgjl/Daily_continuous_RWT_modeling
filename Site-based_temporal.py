import os
import pandas as pd
from glob import glob
from datetime import datetime


def merge_excel_data(comid_folder, date_folder, columns_to_add, output_folder=None):
    """
    合并两个文件夹中的Excel数据（修复版）

    参数:
    comid_folder: 包含以COMID命名的Excel文件的文件夹路径
    date_folder: 包含以日期命名的Excel文件的文件夹路径
    columns_to_add: 需要从日期文件夹添加到COMID文件夹的列名列表
    output_folder: 输出结果的文件夹路径，如果为None则覆盖原文件
    """
    # 创建输出文件夹（如果指定）
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # -------------- 预加载所有日期文件到缓存 --------------
    print("正在预加载所有日期文件...")
    date_files = glob(os.path.join(date_folder, "*.xlsx"))
    date_cache = {}  # 格式: {日期字符串: {COMID: {列名: 值}}}

    for date_file in date_files:
        # 提取日期字符串
        date_str = os.path.splitext(os.path.basename(date_file))[0]

        try:
            # 尝试解析日期格式，确保格式正确
            datetime.strptime(date_str, '%Y-%m-%d')

            # 读取日期文件
            date_df = pd.read_excel(date_file)

            # 检查必要的列
            if 'COMID' not in date_df.columns:
                print(f"警告: 日期文件 {date_file} 不包含'COMID'列，已跳过")
                continue

            # 检查需要添加的列
            missing_columns = [col for col in columns_to_add if col not in date_df.columns]
            if missing_columns:
                print(f"警告: 日期文件 {date_file} 缺少列: {missing_columns}，已跳过这些列")
                columns_available = [col for col in columns_to_add if col in date_df.columns]
                if not columns_available:
                    continue
            else:
                columns_available = columns_to_add

            # 转换COMID为字符串，确保匹配一致性
            date_df['COMID'] = date_df['COMID'].astype(str)

            # 创建该日期的缓存数据结构
            date_data = {}
            for _, row in date_df.iterrows():
                comid = row['COMID']
                date_data[comid] = {col: row[col] for col in columns_available}

            # 存入缓存
            date_cache[date_str] = (date_data, columns_available)
            print(f"已加载日期文件: {date_str}")

        except ValueError:
            print(f"警告: 文件名 {date_file} 不是有效的日期格式(YYYY-MM-DD)，已跳过")
        except Exception as e:
            print(f"读取日期文件 {date_file} 失败: {e}，已跳过")

    print(f"共加载 {len(date_cache)} 个有效的日期文件\n")

    # -------------- 处理所有COMID文件 --------------
    comid_files = glob(os.path.join(comid_folder, "*.xlsx"))
    print(f"找到 {len(comid_files)} 个COMID文件")

    # 处理每个COMID文件
    for comid_file in comid_files:
        # 提取COMID（文件名，不包含扩展名）
        comid = os.path.splitext(os.path.basename(comid_file))[0]
        print(f"处理COMID: {comid}")

        # 读取COMID文件
        try:
            comid_df = pd.read_excel(comid_file)
        except Exception as e:
            print(f"读取文件 {comid_file} 失败: {e}，已跳过")
            continue

        # 检查是否包含date列
        if 'date' not in comid_df.columns:
            print(f"文件 {comid_file} 不包含'date'列，已跳过")
            continue

        # 转换date列为 datetime 类型，确保格式一致
        try:
            comid_df['date'] = pd.to_datetime(comid_df['date'])
        except Exception as e:
            print(f"转换'date'列时出错: {e}，将尝试直接使用字符串匹配")

        # 为需要添加的列创建空列（如果不存在）
        for col in columns_to_add:
            if col not in comid_df.columns:
                comid_df[col] = pd.NA

        # 获取该COMID文件中的所有唯一日期
        unique_dates = comid_df['date'].unique()
        processed_dates = set()

        # 处理每个日期
        for date in unique_dates:
            # 转换为标准日期字符串
            if isinstance(date, pd.Timestamp):
                date_str = date.strftime('%Y-%m-%d')
            else:
                date_str = str(date)

            # 跳过已处理的日期
            if date_str in processed_dates:
                continue

            # 检查缓存中是否有该日期的数据
            if date_str not in date_cache:
                processed_dates.add(date_str)
                continue

            # 从缓存获取数据
            date_data, columns_available = date_cache[date_str]

            # 检查该日期中是否有当前COMID的数据
            if comid in date_data:
                # 批量更新该日期的所有行
                mask = comid_df['date'] == date
                for col in columns_available:
                    comid_df.loc[mask, col] = date_data[comid][col]

            processed_dates.add(date_str)

        # 保存结果
        if output_folder:
            output_file = os.path.join(output_folder, f"{comid}.xlsx")
        else:
            output_file = comid_file  # 覆盖原文件

        try:
            # 修复：移除了不支持的datetime_format参数
            # 使用ExcelWriter来控制日期格式
            with pd.ExcelWriter(output_file, engine='openpyxl', mode='w') as writer:
                comid_df.to_excel(writer, index=False)

                # 如果有日期列，设置日期格式
                if 'date' in comid_df.columns and pd.api.types.is_datetime64_any_dtype(comid_df['date']):
                    worksheet = writer.sheets['Sheet1']
                    date_col_idx = comid_df.columns.get_loc('date') + 1  # +1因为Excel列索引从1开始
                    for row in range(2, len(comid_df) + 2):  # 从第2行开始(跳过表头)
                        worksheet.cell(row=row, column=date_col_idx).number_format = 'yyyy-mm-dd'

            print(f"已保存结果到 {output_file}")
        except Exception as e:
            print(f"保存文件 {output_file} 失败: {e}")

    print("\n所有文件处理完成")




if __name__ == "__main__":
    # 配置参数 - 请根据实际情况修改以下路径和列名
    COMID_FOLDER = "E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\Statistical regression\\COMID_AT_ATC_temp"  # 包含以COMID命名的Excel文件的文件夹
    DATE_FOLDER = "E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\old\\result-0718"  # 包含以日期命名的Excel文件的文件夹
    COLUMNS_TO_ADD = ['lat',    'lon' ,   'Mean_Value',   'Slope',    'Aspect',
                      'Evaporation_mean',    'DSR',    'LWDN','LST_mean']  # 需要从日期文件添加的列名
    OUTPUT_FOLDER = "E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\XGBoost_temporal\\1-COMID_data"  # 输出文件夹，设为None则覆盖原文件

    # 调用函数执行合并
    # merge_excel_data(COMID_FOLDER, DATE_FOLDER, COLUMNS_TO_ADD, OUTPUT_FOLDER)







# ====================================   每个COMID站点拟合一组最优参数，实现时间序列的完整重建   ===============================
# ====================================   XG-Boost - 时间
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from itertools import product
import os
from datetime import datetime
import gc  # 垃圾回收模块


def load_comid_file(file_path):
    """加载单个COMID文件，保留所有列，仅检查必要的特征列和目标列"""
    # 必须存在的特征列和目标列
    required_features = ['DOY', 'AT_mean', 'LAI_mean', 'lat', 'lon', 'Mean_Value',
                         'Slope', 'Aspect', 'Evaporation_mean', 'DSR', 'LWDN', 'LST_mean']
    required_columns = required_features + ['temp', 'date']  # 确保包含date列

    try:
        # 读取文件的所有列（不限制usecols）
        df = pd.read_excel(file_path, engine='openpyxl')

        # 检查必要列是否存在
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺失必要列：{missing_cols}")

        # 提取有观测数据的样本（temp非空）用于建模
        modeled_df = df.dropna(subset=['temp']).reset_index(drop=True)
        if len(modeled_df) < 10:  # 至少需要10条有观测的数据才能建模
            raise ValueError(f"有效观测数据量不足（仅{len(modeled_df)}条，需至少10条）")

        return df, modeled_df, None
    except Exception as e:
        return None, None, str(e)


def fill_missing_features(df, features, train_mean=None):
    """填充特征缺失值（用训练集均值避免数据泄露）"""
    df_filled = df.copy()
    # 如果提供了训练集均值则使用，否则计算当前数据的均值
    if train_mean is None:
        train_mean = {col: df_filled[col].mean() for col in features if pd.api.types.is_numeric_dtype(df_filled[col])}

    for col in features:
        if col in train_mean and pd.api.types.is_numeric_dtype(df_filled[col]):
            df_filled[col].fillna(train_mean[col], inplace=True)
    return df_filled, train_mean


def get_param_combinations(param_grid):
    """生成所有参数组合"""
    param_tuples = list(product(*param_grid.values()))
    return [dict(zip(param_grid.keys(), tuple_)) for tuple_ in param_tuples]


def select_best_params(X_train, y_train, X_val, y_val, param_grid):
    """筛选最优参数：用训练集训练，验证集RMSE最小的为最优"""
    param_combinations = get_param_combinations(param_grid)
    best_rmse = np.inf
    best_params = None

    for params in param_combinations:
        try:
            model = XGBRegressor(
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                n_estimators=params['n_estimators'],
                missing=np.nan,
                objective='reg:squarederror',
                random_state=params['random_state'],
                eval_metric='rmse',
                n_jobs=-1
            )
            model.fit(X_train, y_train, verbose=False)

            # 验证集评估
            y_val_pred = model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_params = params.copy()
        except Exception as e:
            print(f"参数组合 {str(params)[:50]}... 训练失败：{str(e)}")
            continue

    if best_params is None:
        raise ValueError("所有参数组合均无效，无法筛选最优参数")
    return best_params, best_rmse


def train_final_model(X_train, y_train, best_params):
    """用最优参数训练最终模型"""
    model = XGBRegressor(
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        n_estimators=best_params['n_estimators'],
        missing=np.nan,
        objective='reg:squarederror',
        random_state=best_params['random_state'],
        eval_metric='rmse',
        n_jobs=-1
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def process_single_comid(input_file_path, output_folder, param_grid, features):
    """处理单个COMID文件：
    1. 用有观测数据的样本训练模型
    2. 对所有行（包括无观测数据）进行重建
    3. 保留所有原始列和行数，仅添加结果列
    """
    comid = os.path.splitext(os.path.basename(input_file_path))[0]
    print(f"\n{'=' * 40}")
    print(f"处理COMID：{comid}")
    print(f"{'=' * 40}")

    # 1. 加载文件（保留所有列和行）
    full_df, modeled_df, error = load_comid_file(input_file_path)
    if error:
        print(f"❌ 加载失败：{error}")
        return False, comid, error

    print(f"数据概况：总记录数 {len(full_df)} 条，其中有观测数据 {len(modeled_df)} 条")

    # 2. 划分训练集和验证集（仅用有观测的数据）
    train_df, val_df = train_test_split(modeled_df, test_size=0.2, random_state=42, shuffle=True)
    print(f"建模数据划分：训练集{len(train_df)}条，验证集{len(val_df)}条")

    # 3. 填充缺失值（用训练集均值，避免数据泄露）
    train_df_filled, train_mean = fill_missing_features(train_df, features)
    val_df_filled, _ = fill_missing_features(val_df, features, train_mean)  # 使用训练集均值

    # 4. 准备特征和标签
    X_train = train_df_filled[features]
    y_train = train_df_filled['temp']
    X_val = val_df_filled[features]
    y_val = val_df_filled['temp']

    # 5. 筛选最优参数
    try:
        best_params, best_val_rmse = select_best_params(X_train, y_train, X_val, y_val, param_grid)
        print(f"最优参数：{str(best_params)[:80]}...")
        print(f"最优参数验证集RMSE：{best_val_rmse:.4f}")
    except Exception as e:
        print(f"❌ 参数筛选失败：{str(e)}")
        return False, comid, str(e)

    # 6. 用最优参数训练最终模型
    final_model = train_final_model(X_train, y_train, best_params)

    # 7. 对所有数据（包括无观测的）进行预测
    # 用训练集的均值填充所有数据的缺失值（确保一致性）
    full_df_filled, _ = fill_missing_features(full_df, features, train_mean)
    X_full = full_df_filled[features]

    # 生成预测结果并添加到原始数据中
    full_df['XGBoost_time'] = final_model.predict(X_full)

    # 8. 保存结果到新文件夹（保留所有原始列和行数）
    try:
        # 确保结果列在最后
        cols = [col for col in full_df.columns if col != 'XGBoost_time'] + ['XGBoost_time']
        output_file_path = os.path.join(output_folder, os.path.basename(input_file_path))
        full_df[cols].to_excel(output_file_path, index=False, engine='openpyxl')
        print(f"✅ 处理完成：结果已保存至 {os.path.basename(output_file_path)}")
        print(f"   保留原始行数：{len(full_df)} 行，包含date列及所有原始列")

        # 清理内存
        del train_df_filled, val_df_filled, final_model, X_full, full_df_filled
        gc.collect()
        return True, comid, None
    except Exception as e:
        print(f"❌ 保存文件失败：{str(e)}")
        return False, comid, str(e)


def process_all_comids(input_folder, output_folder, param_grid):
    """批量处理所有COMID文件"""
    start_time = datetime.now()
    # 定义训练用的特征列
    features = ['DOY', 'AT_mean', 'LAI_mean', 'lat', 'lon', 'Mean_Value',
                'Slope', 'Aspect', 'Evaporation_mean', 'DSR', 'LWDN', 'LST_mean']

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有COMID文件
    comid_files = [f for f in os.listdir(input_folder) if f.endswith('.xlsx')]
    if not comid_files:
        print(f"⚠️ 在 {input_folder} 中未找到任何xlsx文件")
        return

    # 统计结果
    success_count = 0
    fail_count = 0
    fail_records = []

    # 遍历所有文件
    for idx, file_name in enumerate(comid_files, 1):
        input_file_path = os.path.join(input_folder, file_name)
        print(f"\n【{idx}/{len(comid_files)}】")
        success, comid, error = process_single_comid(
            input_file_path, output_folder, param_grid, features
        )

        if success:
            success_count += 1
        else:
            fail_count += 1
            fail_records.append({'COMID': comid, '原因': error})

    # 输出汇总
    print(f"\n{'=' * 50}")
    print(f"所有COMID处理完成（总耗时：{datetime.now() - start_time}）")
    print(f"成功：{success_count} 个COMID")
    print(f"失败：{fail_count} 个COMID")
    print(f"结果保存路径：{output_folder}")
    if fail_records:
        print("\n失败详情：")
        for record in fail_records:
            print(f"  COMID {record['COMID']}：{record['原因']}")
        # 保存失败记录
        fail_df = pd.DataFrame(fail_records)
        fail_df.to_excel(os.path.join(output_folder, "处理失败记录.xlsx"), index=False, engine='openpyxl')
    print(f"{'=' * 50}")



def main():
    # --------------------------
    # 配置参数（请根据你的路径修改）
    # --------------------------
    INPUT_FOLDER = r"E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\XGBoost_temporal\\1-COMID_data"  # 存放原始COMID文件的文件夹
    OUTPUT_FOLDER = r"E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\XGBoost_temporal\\2-results_daily"  # 新文件夹，用于保存带预测结果的文件
    # 模型参数网格（参数越少，运行越快；可根据需求调整）
    PARAM_GRID = {
        'max_depth': [3, 5],  # 树深度：小值避免过拟合
        'learning_rate': [0.1, 0.2],  # 学习率：控制步长
        'n_estimators': [100, 200],  # 树的数量：越多越耗时但可能更准
        'random_state': [42]  # 固定随机种子，确保结果可复现
    }

    # 检查输入文件夹是否存在
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ 输入文件夹 {INPUT_FOLDER} 不存在，请检查路径")
        return

    # 开始批量处理
    process_all_comids(INPUT_FOLDER, OUTPUT_FOLDER, PARAM_GRID)

#
# if __name__ == "__main__":
#     main()







# ========================================   提取  训练 和验证  数据   ===============================================
import pandas as pd
import os
import re
from datetime import datetime
from tqdm import tqdm


def ensure_directory_exists(path):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def load_master_file(master_file_path):
    """加载主文件，返回包含COMID、date和temp的DataFrame"""
    if not os.path.exists(master_file_path):
        raise FileNotFoundError(f"主文件不存在: {master_file_path}")

    # 读取需要的列，包括temp列
    df = pd.read_excel(master_file_path, engine='openpyxl', usecols=['COMID', 'date', 'temp'])

    # 标准化日期格式
    df['date'] = pd.to_datetime(df['date']).dt.date

    # 去除重复的COMID和date组合
    df = df.drop_duplicates(subset=['COMID', 'date'])

    print(f"主文件加载完成，包含 {len(df)} 条唯一的COMID-date记录")
    return df


def extract_comid_from_filename(filename):
    """从文件名中提取COMID（假设文件名包含数字形式的COMID）"""
    base_name = os.path.splitext(filename)[0]
    # 查找文件名中的数字序列作为COMID
    match = re.search(r'\d+', base_name)
    if match:
        return int(match.group())
    return None


def process_comid_files(master_df, comid_folder_path, output_file_path):
    """处理COMID文件夹中的所有文件，提取匹配的数据"""
    if not os.path.exists(comid_folder_path):
        raise FileNotFoundError(f"COMID文件夹不存在: {comid_folder_path}")

    # 获取文件夹中所有Excel文件
    excel_files = [f for f in os.listdir(comid_folder_path)
                   if f.endswith(('.xlsx', '.xls')) and not f.startswith('~$')]

    if not excel_files:
        raise ValueError(f"在COMID文件夹中未找到Excel文件: {comid_folder_path}")

    print(f"找到 {len(excel_files)} 个COMID相关Excel文件")

    # 按COMID和date分组主数据，提高查找效率
    comid_date_groups = master_df.set_index(['COMID', 'date'])
    all_matched_data = []

    # 遍历所有Excel文件
    for file in tqdm(excel_files, desc="处理文件"):
        file_path = os.path.join(comid_folder_path, file)

        # 从文件名提取COMID
        file_comid = extract_comid_from_filename(file)
        if file_comid is None:
            print(f"警告: 无法从文件名 {file} 中提取COMID，跳过该文件")
            continue

        # 检查该COMID是否存在于主文件中
        if file_comid not in master_df['COMID'].values:
            continue  # 主文件中没有这个COMID，跳过

        try:
            # 读取当前COMID文件
            comid_df = pd.read_excel(file_path, engine='openpyxl')

            # 检查是否包含date列
            if 'date' not in comid_df.columns:
                print(f"警告: 文件 {file} 中不包含'date'列，跳过该文件")
                continue

            # 标准化日期格式
            comid_df['date'] = pd.to_datetime(comid_df['date']).dt.date

            # 移除文件夹文件中的temp列（如果存在）
            if 'temp' in comid_df.columns:
                comid_df = comid_df.drop(columns=['temp'])

            # 获取该COMID在主文件中需要匹配的所有日期
            comid_mask = master_df['COMID'] == file_comid
            target_dates = set(master_df[comid_mask]['date'])

            # 筛选匹配的日期
            mask = comid_df['date'].isin(target_dates)
            matched_rows = comid_df[mask].copy()

            if len(matched_rows) > 0:
                # 添加COMID列（确保存在）
                if 'COMID' not in matched_rows.columns:
                    matched_rows['COMID'] = file_comid

                # 合并主文件中的temp列
                matched_rows = matched_rows.set_index(['COMID', 'date'])
                # 只合并主文件中的temp列
                matched_rows = matched_rows.join(
                    comid_date_groups[['temp']],
                    how='left',
                    on=['COMID', 'date']
                )
                # 重置索引
                matched_rows = matched_rows.reset_index()

                all_matched_data.append(matched_rows)
                # 打印进度信息
                tqdm.write(f"文件 {file} 中找到 {len(matched_rows)} 条匹配记录")

        except Exception as e:
            print(f"处理文件 {file} 时出错: {str(e)}")
            continue

    if not all_matched_data:
        print("未找到任何匹配的记录")
        return None

    # 合并所有匹配的数据
    result_df = pd.concat(all_matched_data, ignore_index=True)

    # 按照COMID和date排序
    result_df = result_df.sort_values(by=['COMID', 'date'])

    # 保存结果
    ensure_directory_exists(os.path.dirname(output_file_path))
    result_df.to_excel(output_file_path, index=False, engine='openpyxl')
    print(f"处理完成，共找到 {len(result_df)} 条匹配记录，已保存至 {output_file_path}")

    return result_df


def main():
    # 配置文件路径（请根据实际情况修改）
    MASTER_FILE_PATH = r"E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\Air2stream\\Modified_999_records.xlsx"  # 包含COMID和date的主文件
    COMID_FOLDER_PATH = r"E:\\huai_river\Huairiver_GEE_data\\Daily_data\\XGBoost_temporal\\2-results_daily"  # 以COMID命名的Excel文件所在文件夹
    OUTPUT_FILE_PATH = r"E:\\huai_river\Huairiver_GEE_data\\Daily_data\\XGBoost_temporal\\results_valida_data.xlsx"  # 输出文件路径

    try:
        # 加载主文件
        master_df = load_master_file(MASTER_FILE_PATH)

        # 处理COMID文件并生成结果
        result_df = process_comid_files(master_df, COMID_FOLDER_PATH, OUTPUT_FILE_PATH)

    except Exception as e:
        print(f"程序出错: {str(e)}")
        import traceback
        print(traceback.format_exc())

#
# if __name__ == "__main__":
#     main()