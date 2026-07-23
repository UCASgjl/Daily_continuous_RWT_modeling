# ==========================================   每个catchment一个模型，一组拟合出来的最优参数  ==============================
import pandas as pd
import os
import pandas as pd
from tqdm import tqdm  # 显示处理进度条
import warnings

warnings.filterwarnings('ignore')  # 忽略Excel读取的无关警告


def add_year_doy2_to_catchment(excel_folder, save_mode="new_folder", new_folder_suffix="_processed"):
    """
    为指定文件夹内所有Catchment的Excel文件添加Year和DOY2列
    适配2020年闰年特性，分别计算两年基于春分日的DOY2

    参数:
        excel_folder: 存储Catchment Excel文件的文件夹路径
        save_mode: 保存模式——"overwrite"（覆盖原文件）、"new_folder"（保存到新文件夹，默认）
        new_folder_suffix: 新文件夹的后缀名，默认"_processed"
    """
    # 1. 验证文件夹路径
    if not os.path.exists(excel_folder):
        print(f"错误：文件夹路径不存在 → {excel_folder}")
        return

    # 2. 创建新文件夹（若选择"new_folder"模式）
    if save_mode == "new_folder":
        processed_folder = os.path.join(os.path.dirname(excel_folder),
                                        f"{os.path.basename(excel_folder)}{new_folder_suffix}")
        os.makedirs(processed_folder, exist_ok=True)
        print(f"所有处理后的文件将保存到 → {processed_folder}")
    elif save_mode != "overwrite":
        print(f"警告：未知保存模式'{save_mode}'，自动切换为'new_folder'模式")
        processed_folder = os.path.join(os.path.dirname(excel_folder),
                                        f"{os.path.basename(excel_folder)}_processed")
        os.makedirs(processed_folder, exist_ok=True)
        print(f"所有处理后的文件将保存到 → {processed_folder}")

    # 3. 获取文件夹内所有Excel文件（仅.xlsx格式）
    excel_files = [f for f in os.listdir(excel_folder) if f.endswith(".xlsx") and not f.startswith("~$")]
    if not excel_files:
        print(f"提示：在文件夹'{excel_folder}'中未找到任何.xlsx文件")
        return
    print(f"共找到 {len(excel_files)} 个Excel文件，开始处理...")

    # 4. 批量处理每个Excel文件
    for file in tqdm(excel_files, desc="处理进度"):
        file_path = os.path.join(excel_folder, file)
        try:
            # 读取Excel文件（保留所有原有列）
            df = pd.read_excel(file_path, engine="openpyxl")

            # 检查是否存在"DOY"列
            if "DOY" not in df.columns:
                print(f"跳过文件'{file}'：未找到'DOY'列")
                continue

            # 确保DOY列为数值类型（避免字符串干扰）
            df["DOY"] = pd.to_numeric(df["DOY"], errors="coerce")  # 无法转换的值设为NaN
            if df["DOY"].isnull().any():
                print(f"警告：文件'{file}'中有 {df['DOY'].isnull().sum()} 个DOY值无法转换为数值，已设为NaN")

            # -------------------------- 添加Year列 --------------------------
            # 逻辑：DOY∈[1,365] → 2019；DOY∈[366,731] → 2020（2020年366天）
            df["Year"] = pd.Series(dtype="int64")  # 初始化Year列为整数类型
            df.loc[df["DOY"].between(1, 365, inclusive="both"), "Year"] = 2019
            df.loc[df["DOY"].between(366, 731, inclusive="both"), "Year"] = 2020

            # -------------------------- 添加DOY2列（基于春分日，适配闰年） --------------------------
            # 春分日：3月20日（两年分别计算本地DOY）
            EQUINOX_2019 = 79  # 2019平年：3月20日 → 1月31+2月28+20=79
            EQUINOX_2020 = 80  # 2020闰年：3月20日 → 1月31+2月29+20=80
            df["DOY2"] = pd.Series(dtype="int64")  # 初始化DOY2列为整数类型

            # 2019年DOY2：原始DOY - 2019年春分日DOY
            mask_2019 = df["Year"] == 2019
            df.loc[mask_2019, "DOY2"] = df.loc[mask_2019, "DOY"] - EQUINOX_2019

            # 2020年DOY2：(原始DOY-365)转换为本地DOY后 - 2020年春分日DOY
            mask_2020 = df["Year"] == 2020
            df.loc[mask_2020, "DOY2"] = (df.loc[mask_2020, "DOY"] - 365) - EQUINOX_2020

            # -------------------------- 保存文件 --------------------------
            if save_mode == "overwrite":
                save_path = file_path  # 覆盖原文件
            else:
                save_path = os.path.join(processed_folder, file)  # 保存到新文件夹

            # 保存Excel（不保留索引）
            df.to_excel(save_path, index=False, engine="openpyxl")

        except Exception as e:
            print(f"处理文件'{file}'时出错：{str(e)}，已跳过该文件")
            continue

    print(f"\n处理完成！共处理 {len(excel_files)} 个文件（部分可能因错误跳过）")


if __name__ == "__main__":
    # -------------------------- 请根据实际情况修改以下参数 --------------------------
    # 1. Excel文件所在文件夹路径
    CATCHMENT_FOLDER = r"E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\EATC\\COMID"

    # 2. 保存模式："overwrite"（覆盖原文件）或 "new_folder"（保存到新文件夹，推荐）
    SAVE_MODE = "new_folder"

    # 3. 新文件夹后缀（仅当SAVE_MODE="new_folder"时生效）
    NEW_FOLDER_SUFFIX = "_with_year_doy2"
    # -----------------------------------------------------------------------------------

    # 执行处理
    # add_year_doy2_to_catchment(
    #     excel_folder=CATCHMENT_FOLDER,
    #     save_mode=SAVE_MODE,
    #     new_folder_suffix=NEW_FOLDER_SUFFIX
    # )





# ==============================================   ATC拟合气温 =========================================================
import pandas as pd
import numpy as np
from scipy.optimize import leastsq
import os
import shutil


# 定义ATC模型拟合函数（支持单个年份的动态N值）
def fitfunc_atc(p, x, year):
    T0, A, theta = p
    # 对单个年份判断平年(365)或闰年(366)
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        N = 366
    else:
        N = 365
    return T0 + A * np.sin(2 * np.pi * x / N + theta)


# 定义残差函数（支持单个年份的动态N值）
def error_atc(p, x, y_obs, year):
    return fitfunc_atc(p, x, year) - y_obs


# 处理单个catchment的气温ATC拟合
def process_atc_fit(file_path, output_folder):
    # 读取流域数据
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"读取文件 {os.path.basename(file_path)} 出错: {str(e)}，跳过该文件！")
        return

    # 检查必要列是否存在
    required_cols = ['DOY2', 'AT_mean', 'Year']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"文件 {os.path.basename(file_path)} 缺少必要列：{missing_cols}，跳过该文件！")
        return

    # 提取用于拟合的数据
    x = df['DOY2'].values  # 年积日（相对于春分日）
    y_obs = df['AT_mean'].values  # 实际气温
    years = df['Year'].values  # 年份数组

    # 拟合参数初值
    temp_mean = np.nanmean(y_obs)
    temp_std = np.nanstd(y_obs)
    p0 = [temp_mean, temp_std if temp_std > 0 else 10, 0]

    # 过滤无效数据
    valid_mask = ~np.isnan(y_obs)
    if np.sum(valid_mask) < 3:
        print(f"文件 {os.path.basename(file_path)} 有效气温数据不足，无法进行ATC拟合！")
        df['AT_ATC'] = np.nan
    else:
        # 获取有效数据的年份
        valid_years = years[valid_mask]
        if len(valid_years) == 0:
            print(f"文件 {os.path.basename(file_path)} 有效年份数据不足，无法进行ATC拟合！")
            df['AT_ATC'] = np.nan
        else:
            # 使用有效数据中的第一个年份进行拟合
            fit_year = valid_years[0]

            # 最小二乘法拟合
            try:
                p_fit, _ = leastsq(
                    error_atc,
                    p0,
                    args=(x[valid_mask], y_obs[valid_mask], fit_year)
                )
                T0, A, theta = p_fit

                # 计算所有日期的ATC拟合值
                at_atc = []
                for i in range(len(df)):
                    doy_val = df['DOY2'].iloc[i]
                    year_val = df['Year'].iloc[i]
                    atc_val = fitfunc_atc(p_fit, doy_val, year_val)
                    at_atc.append(atc_val)
                df['AT_ATC'] = at_atc
            except Exception as e:
                print(f"文件 {os.path.basename(file_path)} 拟合过程出错: {str(e)}，AT_ATC列将设为NaN！")
                df['AT_ATC'] = np.nan

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 构建输出文件路径（保持原文件名）
    file_name = os.path.basename(file_path)
    output_path = os.path.join(output_folder, file_name)

    # 保存结果（覆盖同名文件）
    try:
        df.to_excel(output_path, index=False)
        print(f"已完成 {file_name} 的ATC拟合，结果保存至：{output_path}")
    except Exception as e:
        print(f"保存文件 {file_name} 出错: {str(e)}！")

    return df


# 主函数：批量处理所有catchment文件
def main():
    # 配置路径
    catchments_folder = r'E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\EATC\\COMID_with_year_doy2'  # 输入文件夹路径
    output_folder = r'E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\EATC\\COMID_AT_ATC'  # 输出文件夹路径

    # 检查输入文件夹是否存在
    if not os.path.exists(catchments_folder):
        print(f"错误：输入文件夹 {catchments_folder} 不存在！")
        return

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历文件夹中的每个catchment文件
    for file_name in os.listdir(catchments_folder):
        # 仅处理xlsx文件
        if file_name.endswith('.xlsx'):
            file_path = os.path.join(catchments_folder, file_name)
            process_atc_fit(file_path, output_folder)

    print("\n所有流域气温ATC拟合处理完成！")


# if __name__ == "__main__":
    # main()





# =============================================  纳入站点观测水温数据  ====================================================
import pandas as pd
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')  # 忽略Excel读写中的警告信息


def add_temp_to_catchment_files(
        temp_file_path: str,  # 包含temp的总表路径
        catchments_folder: str,  # 流域Excel文件夹路径
        output_folder: str = None  # 结果输出文件夹（默认覆盖原文件夹）
):
    """
    根据COMID和date，将temp列添加到每个流域Excel文件的最后一列

    参数：
    temp_file_path: 包含"COMID"、"date"、"temp"列的总表路径
    catchments_folder: 以COMID命名的流域Excel文件所在文件夹
    output_folder: 结果保存路径（None则覆盖原文件，建议先指定新路径测试）
    """
    # --------------------------
    # 1. 读取温度总表并预处理
    # --------------------------
    try:
        # 读取总表，确保关键列存在
        temp_df = pd.read_excel(temp_file_path)
        required_temp_cols = ["COMID", "date", "temp"]
        missing_temp_cols = [col for col in required_temp_cols if col not in temp_df.columns]
        if missing_temp_cols:
            raise ValueError(f"温度总表缺少必要列：{missing_temp_cols}")

        # 预处理日期格式（统一转为datetime，避免格式不匹配）
        temp_df["date"] = pd.to_datetime(temp_df["date"], errors="coerce")
        # 过滤无效数据（日期无效或temp为空的行）
        temp_df = temp_df.dropna(subset=["COMID", "date", "temp"])
        # 将COMID转为字符串（避免数字格式不一致，如123和123.0）
        temp_df["COMID"] = temp_df["COMID"].astype(str).str.strip()

        print(f"成功读取温度总表：共{len(temp_df)}条有效温度数据，覆盖{temp_df['COMID'].nunique()}个COMID")

    except Exception as e:
        print(f"读取温度总表失败：{str(e)}")
        return

    # --------------------------
    # 2. 配置输出文件夹
    # --------------------------
    if output_folder is None:
        output_folder = catchments_folder  # 默认覆盖原文件
    else:
        os.makedirs(output_folder, exist_ok=True)  # 确保输出文件夹存在
    print(f"结果将保存至：{output_folder}")

    # --------------------------
    # 3. 遍历所有流域文件并添加temp列
    # --------------------------
    # 获取文件夹中所有Excel文件（排除临时文件）
    catchment_files = [f for f in os.listdir(catchments_folder)
                       if f.endswith(".xlsx") and not f.startswith("~$")]

    if not catchment_files:
        print(f"流域文件夹 {catchments_folder} 中未找到Excel文件")
        return

    # 逐个处理流域文件
    for file_name in catchment_files:
        # 提取文件名中的COMID（假设文件名就是COMID，如"12345.xlsx"）
        comid = os.path.splitext(file_name)[0].strip()
        file_path = os.path.join(catchments_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        try:
            # --------------------------
            # 3.1 读取当前流域文件
            # --------------------------
            catchment_df = pd.read_excel(file_path)
            if "date" not in catchment_df.columns:
                print(f"跳过 {file_name}：文件中缺少'date'列")
                continue

            # 预处理流域文件的日期（与温度总表格式统一）
            catchment_df["date"] = pd.to_datetime(catchment_df["date"], errors="coerce")
            # 记录原始数据行数（用于后续验证）
            original_rows = len(catchment_df)

            # --------------------------
            # 3.2 筛选当前COMID的温度数据
            # --------------------------
            comid_temp_df = temp_df[temp_df["COMID"] == comid].copy()
            if len(comid_temp_df) == 0:
                print(f"警告 {file_name}：温度总表中无该COMID（{comid}）的温度数据，temp列设为NaN")
                catchment_df["temp"] = pd.NA
            else:
                # --------------------------
                # 3.3 按日期匹配并添加temp列
                # --------------------------
                # 将温度数据转为字典（date→temp），提高匹配效率
                temp_dict = dict(zip(comid_temp_df["date"], comid_temp_df["temp"]))
                # 按日期匹配temp（无匹配则为NaN）
                catchment_df["temp"] = catchment_df["date"].map(temp_dict)

                # 统计匹配结果
                matched_count = catchment_df["temp"].notna().sum()
                print(f"处理 {file_name}：共{original_rows}行数据，成功匹配{matched_count}条温度数据")

            # --------------------------
            # 3.4 保存结果
            # --------------------------
            catchment_df.to_excel(output_path, index=False)

        except Exception as e:
            print(f"处理 {file_name} 失败：{str(e)}")
            continue

    print(f"\n所有文件处理完成！共处理 {len(catchment_files)} 个流域文件")


# --------------------------
# 4. 主函数：配置路径并运行
# --------------------------
if __name__ == "__main__":
    # --------------------------
    # 请根据你的实际路径修改以下参数！！！
    # --------------------------
    TEMP_FILE_PATH = r"E:\\huai_river\\Station_RWT\\Extracted_matching_with_temp_LST.xlsx"  # 包含COMID、date、temp的总表路径
    CATCHMENTS_FOLDER = r"E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\EATC\\COMID_AT_ATC"  # 以COMID命名的流域Excel文件夹
    OUTPUT_FOLDER = r"E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\EATC\\COMID_AT_ATC_temp"  # 结果输出文件夹（建议新建，避免覆盖原文件）

    # 运行函数
    # add_temp_to_catchment_files(
    #     temp_file_path=TEMP_FILE_PATH,
    #     catchments_folder=CATCHMENTS_FOLDER,
    #     output_folder=OUTPUT_FOLDER
    # )








# ==============================================   EATC重建RWT =========================================================
# ==============================================     波动优化   =================================================
import pandas as pd
import numpy as np
from scipy.optimize import leastsq
import os


# --------------------------
# 1. 核心拟合函数
# --------------------------
def fitfunc_eatc(p, x, delta_Ta, gamma, year):
    T0, A, theta, lam = p
    # 闰年判断
    is_leap = np.where(((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0), True, False)
    N = np.where(is_leap, 366, 365)
    return T0 + A * np.sin(2 * np.pi * x / N + theta) + lam * delta_Ta * gamma


# 2. 残差函数
def error_eatc(p, x, delta_Ta, gamma, y_obs, year):
    model_residual = fitfunc_eatc(p, x, delta_Ta, gamma, year) - y_obs
    return model_residual


# --------------------------
# 3. 单个catchment处理函数
# --------------------------
def process_catchment(file_path, catchment_id, params_df, output_folder):
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"错误：读取{catchment_id}文件失败 - {str(e)}")
        return params_df

    required_cols = ['DOY2', 'AT_mean', 'LAI_mean', 'temp', 'Year', 'AT_ATC']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"警告：{catchment_id} 缺少必要列：{missing_cols}，跳过该流域！")
        params_df.loc[len(params_df)] = [catchment_id, np.nan, np.nan, np.nan, np.nan]
        return params_df

    LAI_data = df['LAI_mean'].dropna()
    if len(LAI_data) == 0 or (LAI_data.max() <= LAI_data.min() + 1e-6):
        print(f"警告：{catchment_id} 的LAI_mean数据无效，跳过该流域！")
        params_df.loc[len(params_df)] = [catchment_id, np.nan, np.nan, np.nan, np.nan]
        return params_df
    LAI_min, LAI_max = LAI_data.min(), LAI_data.max()

    df_fit = df.dropna(subset=['temp', 'DOY2', 'LAI_mean', 'AT_mean', 'AT_ATC', 'Year']).reset_index(drop=True)
    if len(df_fit) < 4:
        print(f"警告：{catchment_id} 有效拟合数据不足（{len(df_fit)}个），无法拟合参数！")
        params_df.loc[len(params_df)] = [catchment_id, np.nan, np.nan, np.nan, np.nan]
        return params_df

    x_fit = df_fit['DOY2'].astype(float).values
    y_obs = df_fit['temp'].astype(float).values
    AT_mean_fit = df_fit['AT_mean'].astype(float).values
    LAI_fit = df_fit['LAI_mean'].astype(float).values
    year_fit = df_fit['Year'].astype(int).values
    AT_ATC_fit = df_fit['AT_ATC'].astype(float).values

    delta_Ta_fit = AT_mean_fit - AT_ATC_fit
    gamma_fit = (LAI_max - LAI_min) / (LAI_fit - LAI_min + 1)

    T0_init = np.nanmean(y_obs)
    A_init = np.nanstd(y_obs) * 1.5
    p0 = [T0_init, A_init, 0, 0.5]

    try:
        p_fit, cov_x, infodict, mesg, ier = leastsq(
            func=error_eatc,
            x0=p0,
            args=(x_fit, delta_Ta_fit, gamma_fit, y_obs, year_fit),
            full_output=True,
            maxfev=2000,
            ftol=1e-8,
            xtol=1e-8
        )
        if ier not in [1, 2, 3, 4]:
            print(f"警告：{catchment_id} 拟合未收敛（状态码：{ier}），参数无效！")
            p_fit = [np.nan, np.nan, np.nan, np.nan]
        else:
            print(f"拟合成功：{catchment_id}，参数：T0={p_fit[0]:.2f}, A={p_fit[1]:.2f}, theta={p_fit[2]:.2f}, lambda={p_fit[3]:.4f}")
    except Exception as e:
        print(f"警告：{catchment_id} 拟合过程出错 - {str(e)}")
        p_fit = [np.nan, np.nan, np.nan, np.nan]

    delta_Ta_all = df['AT_mean'].astype(float) - df['AT_ATC'].astype(float)
    gamma_all = (LAI_max - LAI_min) / (df['LAI_mean'].astype(float) - LAI_min + 1)

    reconstructed_RWT = []
    year_all = df['Year'].astype(int).values
    for idx, row in df.iterrows():
        if np.isnan(p_fit[0]):
            reconstructed_RWT.append(np.nan)
        else:
            x_val = float(row['DOY2'])
            delta_Ta_val = delta_Ta_all.iloc[idx]
            gamma_val = gamma_all.iloc[idx]
            year_val = year_all[idx]
            rwt_val = fitfunc_eatc(p_fit, x_val, delta_Ta_val, gamma_val, year_val)
            reconstructed_RWT.append(rwt_val)

    df['EATC_RWT'] = reconstructed_RWT
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"{catchment_id}.xlsx")
    try:
        df.to_excel(output_file, index=False)
        print(f"成功处理：{catchment_id}，结果保存至：{output_file}\n")
    except Exception as e:
        print(f"错误：保存{catchment_id}结果失败 - {str(e)}\n")

    params_df.loc[len(params_df)] = [catchment_id, p_fit[0], p_fit[1], p_fit[2], p_fit[3]]
    return params_df


# --------------------------
# 4. 主函数（需根据实际情况修改输入输出路径）
# --------------------------
def main():
    input_folder = r'E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\EATC\\COMID_AT_ATC_temp'
    output_folder = r'E:\\huai_river\\Huairiver_GEE_data\\Daily_data\\EATC\\output_full_new'
    params_file = os.path.join(output_folder, 'All_Catchments_EATC_Params.xlsx')

    params_df = pd.DataFrame(columns=['catchment_id', 'T0', 'A', 'theta', 'lambda'])

    for filename in os.listdir(input_folder):
        if filename.endswith('.xlsx'):
            catchment_id = os.path.splitext(filename)[0]
            file_path = os.path.join(input_folder, filename)
            params_df = process_catchment(file_path, catchment_id, params_df, output_folder)

    params_df.to_excel(params_file, index=False)
    print(f"所有流域处理完毕，模型参数保存至：{params_file}")


if __name__ == "__main__":
    main()