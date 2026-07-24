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
    """Ensure test data contains all feature columns"""
    for col in features:
        if col not in X.columns:
            X[col] = np.nan
        if f'{col}_missing' not in X.columns:
            X[f'{col}_missing'] = X[col].isnull().astype(int)
    return X[features + [f'{c}_missing' for c in features]]


def create_param_grid():
    """Return parameters search grid"""
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
    """Calculate overall RMSE, MAE, R2, ME"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    me = np.mean(y_pred - y_true)
    return rmse, mae, r2, me


def calc_median_metrics(df, comid_col, target_col, pred_col):
    """
    By COMID Group calculation Median RMSE and Median AE
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
    Calculate each COMID(station) RMSE and MAE
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


# ========================== Station-level SHAP FeatureImportance ==========================

def calc_comid_shap_importance(model, df, comid_col, feature_cols,
                                sample_per_comid=None, random_state=42):
    """
    Calculate SHAP mean absolute value for each feature at each station (station-level feature importance)。

    Parameters:
        model: TrainingOK XGBoost Model
        df: ContainsFeatureand COMID Dataframe
        comid_col: Station column name（such as 'COMID'）
        feature_cols: ModelInputFeatureColumn name list
        sample_per_comid: Maximum per stationSamplingsample count，None means noSampling
        random_state: Randomseed

    Returns:
        DataFrame: Row indexis COMID，ColumnisEachFeaturemean absolutefor SHAP value
    """
    np.random.seed(random_state)
    X_full = df[feature_cols].copy()
    comids = df[comid_col].values

    # If data volume is large, sample separately for each station to accelerate SHAP calculation
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

    # Calculate SHAP value
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Construct DataFrame and aggregate by COMID (mean absolute SHAP value)
    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    shap_df[comid_col] = comids_sub

    importance = shap_df.groupby(comid_col)[feature_cols].apply(
        lambda x: np.abs(x).mean()
    )

    # importance = shap_df.groupby(comid_col)[feature_cols].mean()

    return importance


# ========================== Generaluseparameterssearch ==========================

def param_search(param_grid, X, y, cv_func):
    """
    Generaluseparameterssearch framework
    cv_func: Receive (param_dict, X, y) -> (avg_val_rmse, fold_results) function
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

        print(f"  [{idx}/{len(param_combinations)}] parameters: {param_dict} -> Average validation RMSE: {avg_val_rmse:.4f}")

    return best_params, best_val_rmse, all_results


# ========================== Space CV ==========================

def space_cv_func(param_dict, X, y, train_data, comid_splits):
    """Spatial cross-validation: 5-fold (based on COMID split)"""
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
            'Parameter combination': str(param_dict),
            'Fold number': val_fold + 1,
            'Validation set RMSE': val_rmse
        })

    avg_val_rmse = np.mean(val_rmse_list)
    fold_results.append({
        'Parameter combination': str(param_dict),
        'Fold number': 'Average',
        'Validation set RMSE': avg_val_rmse
    })

    return avg_val_rmse, fold_results


def run_space_cv(train_data, X, y, features, output_folder):
    """Execute spatial cross-validation full workflow"""
    print("\n" + "=" * 60)
    print("【Spatial cross-validation】")
    print("=" * 60)

    # SplitCOMID
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

    # parameterssearch
    param_grid = create_param_grid()
    best_params, best_val_rmse, all_results = param_search(
        param_grid, X, y,
        lambda p, X_, y_: space_cv_func(p, X_, y_, train_data, comid_splits)
    )

    print(f"\nOptimal parameters: {best_params}")
    print(f"OptimalAverage validation RMSE: {best_val_rmse:.4f}")

    # Rerun 5-fold CV with optimal parameters and save each fold prediction + station feature importance
    train_val_splits = np.array_split(train_val_comids, 5)
    fold_metrics_list = []
    all_folds_list = []
    all_fold_comid_importance = []  # Collect each fold stationFeatureImportance

    # All model input features (including missing indicator columns)
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

        # Save this fold validation set prediction
        val_original = train_data[val_idx].copy().reset_index(drop=True)
        val_original['Predicted_RWT'] = y_val_pred
        val_original['fold'] = val_fold + 1
        val_path = os.path.join(output_folder, f'Space_Optimal parameters_Fold{val_fold + 1}Fold validation set prediction.xlsx')
        val_original.to_excel(val_path, index=False)
        print(f"  ✓ SavedFold{val_fold + 1}Fold validation set prediction: {val_path}")

        # Calculate this fold overall accuracy
        rmse_f, mae_f, r2_f, me_f = calc_metrics(y_val_fold, y_val_pred)
        median_rmse_f, median_ae_f = calc_median_metrics(val_original, 'COMID', 'in-situ RWT', 'Predicted_RWT')

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

        # Calculate and save each station accuracy for this fold
        comid_metrics = calc_comid_metrics(val_original, 'COMID', 'in-situ RWT', 'Predicted_RWT')
        comid_metrics_path = os.path.join(output_folder, f'Space_Optimal parameters_Fold{val_fold + 1}Fold_each station accuracy.xlsx')
        comid_metrics.to_excel(comid_metrics_path, index=False)
        print(f"  ✓ SavedFold{val_fold + 1}foldEach station accuracy: {comid_metrics_path}")

        # ========== New：① Feature importance (SHAP) for each station in this fold validation set ==========
        print(f"  CalculateFold{val_fold + 1}foldValidationset stationFeatureImportance（SHAP）...")
        fold_comid_imp = calc_comid_shap_importance(
            model, val_original, 'COMID', all_feature_cols
        )
        fold_imp_path = os.path.join(output_folder, f'Space_Fold{val_fold + 1}Fold_validation set station feature importance.xlsx')
        fold_comid_imp.to_excel(fold_imp_path)
        print(f"  ✓ SavedFold{val_fold + 1}foldValidationset stationFeatureImportance: {fold_imp_path}")

        # Record this fold importance for subsequent aggregation
        fold_comid_imp_long = fold_comid_imp.reset_index().melt(
            id_vars=['COMID'], var_name='Feature', value_name='MeanAbsSHAP'
        )
        fold_comid_imp_long['fold'] = val_fold + 1
        all_fold_comid_importance.append(fold_comid_imp_long)

        all_folds_list.append(val_original)

    # Merge all fold validation sets
    all_folds_df = pd.concat(all_folds_list, ignore_index=True)
    merge_path = os.path.join(output_folder, 'Space_Optimal parameters_All foldsValidationsetMerge.xlsx')
    all_folds_df.to_excel(merge_path, index=False)
    print(f"\nAlreadyMerge all fold validation sets: {merge_path} ({len(all_folds_df)} records)")

    # Save each fold accuracy summary
    fold_metrics_df = pd.DataFrame(fold_metrics_list)
    fold_metrics_path = os.path.join(output_folder, 'Space_Optimal parameters_Each fold validation set accuracy summary.xlsx')
    fold_metrics_df.to_excel(fold_metrics_path, index=False)
    print(f"SavedEach foldAccuracySummary: {fold_metrics_path}")

    # Calculate 5-fold validation set overall accuracy (based on merged data)
    rmse_val, mae_val, r2_val, me_val = calc_metrics(
        all_folds_df['in-situ RWT'].values, all_folds_df['Predicted_RWT'].values
    )
    median_rmse_val, median_ae_val = calc_median_metrics(all_folds_df, 'COMID', 'in-situ RWT', 'Predicted_RWT')
    print(f"\n5foldValidationset overallAccuracy:")
    print(f"  RMSE={rmse_val:.4f}, MAE={mae_val:.4f}, R2={r2_val:.4f}, ME={me_val:.4f}")
    print(f"  Median_RMSE={median_rmse_val:.4f}, Median_AE={median_ae_val:.4f}")

    # Aggregate all folds station feature importance (long format)
    if all_fold_comid_importance:
        all_fold_imp_df = pd.concat(all_fold_comid_importance, ignore_index=True)
        all_fold_imp_path = os.path.join(output_folder, 'Space_All folds_validation set station feature importance summary.xlsx')
        all_fold_imp_df.to_excel(all_fold_imp_path, index=False)
        print(f"SavedAll foldsValidationset stationFeatureImportanceSummary（Long format）: {all_fold_imp_path}")

    # Train final model (all train_val data)
    final_model = XGBRegressor(**best_params)
    final_model.fit(X_train_val, y_train_val)

    # Independent test set evaluation
    y_test_pred = final_model.predict(X_test)
    rmse_test, mae_test, r2_test, me_test = calc_metrics(y_test, y_test_pred)

    test_df = train_data[test_idx].copy().reset_index(drop=True)
    test_df['Predicted_RWT'] = y_test_pred
    median_rmse_test, median_ae_test = calc_median_metrics(test_df, 'COMID', 'in-situ RWT', 'Predicted_RWT')

    # SaveIndependent test set (with predicted values)
    test_path = os.path.join(output_folder, 'Space_Independent test set samples.xlsx')
    test_df.to_excel(test_path, index=False)
    print(f"\nExportedIndependent test set(includingPredictionvalue): {test_path}")

    # CalculateandSaveEach station accuracy for independent test set
    test_comid_metrics = calc_comid_metrics(test_df, 'COMID', 'in-situ RWT', 'Predicted_RWT')
    test_comid_path = os.path.join(output_folder, 'Space_Independent test set_Each station accuracy.xlsx')
    test_comid_metrics.to_excel(test_comid_path, index=False)
    print(f"SavedIndependent test set each station accuracy: {test_comid_path}")

    # ========== New：② Final model - independent test set station feature importance ==========
    print("  Calculate finalModelIndependent test setstationFeatureImportance（SHAP）...")
    test_comid_imp = calc_comid_shap_importance(
        final_model, test_df, 'COMID', all_feature_cols
    )
    test_imp_path = os.path.join(output_folder, 'Space_Final model_test set station feature importance.xlsx')
    test_comid_imp.to_excel(test_imp_path)
    print(f"  ✓ SavedFinalModelTestset stationFeatureImportance: {test_imp_path}")

    # Trainingset back-casting
    y_train_pred = final_model.predict(X_train_val)
    rmse_train, mae_train, r2_train, me_train = calc_metrics(y_train_val, y_train_pred)

    train_df = train_data[train_val_idx].copy().reset_index(drop=True)
    train_df['Predicted_RWT'] = y_train_pred
    median_rmse_train, median_ae_train = calc_median_metrics(train_df, 'COMID', 'in-situ RWT', 'Predicted_RWT')

    # ========== New：② Final model - training set station feature importance ==========
    print("  Calculate finalModelTrainingset stationFeatureImportance（SHAP）...")
    train_comid_imp = calc_comid_shap_importance(
        final_model, train_df, 'COMID', all_feature_cols
    )
    train_imp_path = os.path.join(output_folder, 'Space_Final model_training set station feature importance.xlsx')
    train_comid_imp.to_excel(train_imp_path)
    print(f"  ✓ SavedFinalModelTrainingset stationFeatureImportance: {train_imp_path}")

    print(f"\nFinal model accuracy:")
    print(f"  Trainingset: RMSE={rmse_train:.4f}, MAE={mae_train:.4f}, R2={r2_train:.4f}, ME={me_train:.4f}, Median_RMSE={median_rmse_train:.4f}, Median_AE={median_ae_train:.4f}")
    print(f"  Validationset: RMSE={rmse_val:.4f}, MAE={mae_val:.4f}, R2={r2_val:.4f}, ME={me_val:.4f}, Median_RMSE={median_rmse_val:.4f}, Median_AE={median_ae_val:.4f}")
    print(f"  Testset: RMSE={rmse_test:.4f}, MAE={mae_test:.4f}, R2={r2_test:.4f}, ME={me_test:.4f}, Median_RMSE={median_rmse_test:.4f}, Median_AE={median_ae_test:.4f}")

    # SaveResult
    results_df = pd.DataFrame(all_results)
    final_results = pd.DataFrame({
        'Evaluation dataset': ['Training set (back-casting)', 'Validation set (5-fold CV)', 'Independent test set'],
        'RMSE': [rmse_train, rmse_val, rmse_test],
        'MAE': [mae_train, mae_val, mae_test],
        'R2': [r2_train, r2_val, r2_test],
        'ME': [me_train, me_val, me_test],
        'Median_RMSE': [median_rmse_train, median_rmse_val, median_rmse_test],
        'Median_AE': [median_ae_train, median_ae_val, median_ae_test]
    })

    excel_path = os.path.join(output_folder, 'Space_Model evaluation results.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='All parameter results', index=False)
        final_results.to_excel(writer, sheet_name='Optimal parameter final evaluation', index=False)
        pd.DataFrame([best_params]).to_excel(writer, sheet_name='Optimal parameters', index=False)
        fold_metrics_df.to_excel(writer, sheet_name='Each fold validation set accuracy', index=False)

    print(f"\nResultSaved: {excel_path}")

    return final_model, best_params, features


# ========================== Time CV ==========================

def time_cv_func(param_dict, X, y, train_data):
    """Temporal cross-validation: 3-fold (based on date split)"""
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
            'Parameter combination': str(param_dict),
            'Fold number': val_fold + 1,
            'Validation set RMSE': val_rmse
        })

    avg_val_rmse = np.mean(val_rmse_list)
    fold_results.append({
        'Parameter combination': str(param_dict),
        'Fold number': 'Average',
        'Validation set RMSE': avg_val_rmse
    })

    return avg_val_rmse, fold_results, test_idx


def run_time_cv(train_data, X, y, features, output_folder):
    """Execute temporal cross-validation full workflow"""
    print("\n" + "=" * 60)
    print("【Temporal cross-validation】")
    print("=" * 60)

    all_feature_cols = features + [f'{c}_missing' for c in features]

    # parameterssearch
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

        print(f"  [{idx}/{len(param_combinations)}] parameters: {param_dict} -> Average validation RMSE: {avg_val_rmse:.4f}")

    print(f"\nOptimal parameters: {best_params}")
    print(f"OptimalAverage validation RMSE: {best_val_rmse:.4f}")

    # TrainingFinalModel（before3Timeperiod）
    train_val_idx = (train_data['date'] >= '2019-01-01') & (train_data['date'] <= '2020-06-30')
    X_train_val, y_train_val = X[train_val_idx], y[train_val_idx]

    final_model = XGBRegressor(**best_params)
    final_model.fit(X_train_val, y_train_val)

    # Independent test set evaluation（2020second half）
    X_test, y_test = X[test_idx], y[test_idx]
    y_test_pred = final_model.predict(X_test)
    rmse_test, mae_test, r2_test, me_test = calc_metrics(y_test, y_test_pred)

    test_df = train_data[test_idx].copy().reset_index(drop=True)
    test_df['Predicted_RWT'] = y_test_pred
    test_path = os.path.join(output_folder, 'Time_Independent test set samples.xlsx')
    test_df.to_excel(test_path, index=False)
    print(f"ExportedIndependent test set(includingPredictionvalue): {test_path}")

    # ========== New：Final model - test set station feature importance ==========
    print("  Calculate finalModelTestset stationFeatureImportance（SHAP）...")
    test_comid_imp = calc_comid_shap_importance(
        final_model, test_df, 'COMID', all_feature_cols
    )
    test_imp_path = os.path.join(output_folder, 'Time_Final model_test set station feature importance.xlsx')
    test_comid_imp.to_excel(test_imp_path)
    print(f"  ✓ SavedFinalModelTestset stationFeatureImportance: {test_imp_path}")

    # Trainingset back-casting
    y_train_pred = final_model.predict(X_train_val)
    rmse_train, mae_train, r2_train, me_train = calc_metrics(y_train_val, y_train_pred)

    train_df = train_data[train_val_idx].copy().reset_index(drop=True)
    train_df['Predicted_RWT'] = y_train_pred

    # ========== New：Final model - training set station feature importance ==========
    print("  Calculate finalModelTrainingset stationFeatureImportance（SHAP）...")
    train_comid_imp = calc_comid_shap_importance(
        final_model, train_df, 'COMID', all_feature_cols
    )
    train_imp_path = os.path.join(output_folder, 'Time_Final model_training set station feature importance.xlsx')
    train_comid_imp.to_excel(train_imp_path)
    print(f"  ✓ SavedFinalModelTrainingset stationFeatureImportance: {train_imp_path}")

    print(f"FinalModel - TrainingsetRMSE: {rmse_train:.4f}")
    print(f"FinalModel - Validation set RMSE: {best_val_rmse:.4f}")
    print(f"FinalModel - Independent test setRMSE: {rmse_test:.4f}")

    # SaveResult
    results_df = pd.DataFrame(all_results)
    final_results = pd.DataFrame({
        'Evaluation dataset': ['Training set (back-casting)', 'Validation set (3-fold CV)', 'Independent test set'],
        'RMSE': [rmse_train, best_val_rmse, rmse_test],
        'MAE': [mae_train, np.nan, mae_test],
        'R2': [r2_train, np.nan, r2_test],
        'ME': [me_train, np.nan, me_test]
    })

    excel_path = os.path.join(output_folder, 'Time_Model evaluation results.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='All parameter results', index=False)
        final_results.to_excel(writer, sheet_name='Optimal parameter final evaluation', index=False)
        pd.DataFrame([best_params]).to_excel(writer, sheet_name='Optimal parameters', index=False)

    print(f"ResultSaved: {excel_path}")

    return final_model, best_params, features


# ========================== Random CV ==========================

def random_cv_func(param_dict, X, y):
    """Random cross-validation: 5-fold"""
    val_rmse_list = []
    fold_results = []

    for val_fold in range(5):
        # RandomSplit20%Validationset
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
            'Parameter combination': str(param_dict),
            'Fold number': val_fold + 1,
            'Validation set RMSE': val_rmse
        })

    avg_val_rmse = np.mean(val_rmse_list)
    fold_results.append({
        'Parameter combination': str(param_dict),
        'Fold number': 'Average',
        'Validation set RMSE': avg_val_rmse
    })

    return avg_val_rmse, fold_results


def run_random_cv(train_data, X, y, features, output_folder):
    """Execute random cross-validation full workflow"""
    print("\n" + "=" * 60)
    print("【Random cross-validation】")
    print("=" * 60)

    all_feature_cols = features + [f'{c}_missing' for c in features]

    # parameterssearch
    param_grid = create_param_grid()
    best_params, best_val_rmse, all_results = param_search(
        param_grid, X, y,
        lambda p, X_, y_: random_cv_func(p, X_, y_)
    )

    print(f"\nOptimal parameters: {best_params}")
    print(f"OptimalAverage validation RMSE: {best_val_rmse:.4f}")

    # RandomSplit20%Testset
    np.random.seed(42)
    test_idx = np.random.choice(len(X), int(0.2 * len(X)), replace=False)
    test_mask = np.isin(np.arange(len(X)), test_idx)

    X_test, y_test = X[test_mask], y[test_mask]
    train_val_mask = ~test_mask
    X_train_val, y_train_val = X[train_val_mask], y[train_val_mask]

    # TrainingFinalModel
    final_model = XGBRegressor(**best_params)
    final_model.fit(X_train_val, y_train_val)

    # TestsetEvaluation
    y_test_pred = final_model.predict(X_test)
    rmse_test, mae_test, r2_test, me_test = calc_metrics(y_test, y_test_pred)

    test_df = train_data.iloc[test_idx].copy().reset_index(drop=True)
    test_df['Predicted_RWT'] = y_test_pred
    test_path = os.path.join(output_folder, 'Random_Independent test set samples.xlsx')
    test_df.to_excel(test_path, index=False)
    print(f"ExportedIndependent test set(includingPredictionvalue): {test_path}")

    # ========== New：Final model - test set station feature importance ==========
    print("  Calculate finalModelTestset stationFeatureImportance（SHAP）...")
    test_comid_imp = calc_comid_shap_importance(
        final_model, test_df, 'COMID', all_feature_cols
    )
    test_imp_path = os.path.join(output_folder, 'Random_Final model_test set station feature importance.xlsx')
    test_comid_imp.to_excel(test_imp_path)
    print(f"  ✓ SavedFinalModelTestset stationFeatureImportance: {test_imp_path}")

    # Trainingset back-casting
    y_train_pred = final_model.predict(X_train_val)
    rmse_train, mae_train, r2_train, me_train = calc_metrics(y_train_val, y_train_pred)

    train_df = train_data.iloc[train_val_mask].copy().reset_index(drop=True)
    train_df['Predicted_RWT'] = y_train_pred

    # ========== New：Final model - training set station feature importance ==========
    print("  Calculate finalModelTrainingset stationFeatureImportance（SHAP）...")
    train_comid_imp = calc_comid_shap_importance(
        final_model, train_df, 'COMID', all_feature_cols
    )
    train_imp_path = os.path.join(output_folder, 'Random_Final model_training set station feature importance.xlsx')
    train_comid_imp.to_excel(train_imp_path)
    print(f"  ✓ SavedFinalModelTrainingset stationFeatureImportance: {train_imp_path}")

    print(f"FinalModel - TrainingsetRMSE: {rmse_train:.4f}")
    print(f"FinalModel - Validation set RMSE: {best_val_rmse:.4f}")
    print(f"FinalModel - TestsetRMSE: {rmse_test:.4f}")

    # SaveResult
    results_df = pd.DataFrame(all_results)
    final_results = pd.DataFrame({
        'Evaluation dataset': ['Trainingset', 'Validation set (5-fold CV)', 'Testset'],
        'RMSE': [rmse_train, best_val_rmse, rmse_test],
        'MAE': [mae_train, np.nan, mae_test],
        'R2': [r2_train, np.nan, r2_test],
        'ME': [me_train, np.nan, me_test]
    })

    excel_path = os.path.join(output_folder, 'Random_Model evaluation results.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='All parameter results', index=False)
        final_results.to_excel(writer, sheet_name='Optimal parameter final evaluation', index=False)
        pd.DataFrame([best_params]).to_excel(writer, sheet_name='Optimal parameters', index=False)

    print(f"ResultSaved: {excel_path}")

    return final_model, best_params, features


# ========================== Batch prediction ==========================

def batch_predict(test_folder_path, output_folder_path, models_dict, features):
    """Batch prediction based on selected models"""
    print("\n" + "=" * 60)
    print("【Batch prediction date folder】")
    print("=" * 60)

    test_files = [f for f in os.listdir(test_folder_path) if f.endswith('.xlsx')]
    if not test_files:
        print(f"Warning: Folder {test_folder_path} not found in .xlsx File")
        return

    for file in test_files:
        try:
            test_data = pd.read_excel(os.path.join(test_folder_path, file))

            # EnsureFeatureComplete
            X_test = ensure_features(test_data, features)

            # Only predict selected models
            for model_name, model_info in models_dict.items():
                if model_info is None:
                    continue
                model = model_info['model']
                pred_col = f'Predicted_RWT_{model_name}'
                test_data[pred_col] = model.predict(X_test)
                print(f"  {model_name} PredictionCompleted: {file}")

            output_path = os.path.join(output_folder_path, file)
            test_data.to_excel(output_path, index=False)

        except Exception as e:
            print(f"ProcessingFile {file} Error occurred: {str(e)}")


# ========================== SHAP Feature contribution（Global） ==========================

def calculate_feature_contributions(X, models_dict, output_folder):
    """Calculate SHAP feature contribution for selected models"""
    print("\n" + "=" * 60)
    print("【Feature contribution analysis】")
    print("=" * 60)

    # SamplingAcceleration
    sample_idx = np.random.choice(len(X), min(1000, len(X)), replace=False)
    X_sample = X.iloc[sample_idx] if isinstance(X, pd.DataFrame) else X[sample_idx]

    contributions_list = []
    valid_models = []

    for model_name, model_info in models_dict.items():
        if model_info is None:
            print(f"  {model_name} Not run，SkipSHAPCalculate")
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

            print(f"  {model_name} SHAPCalculation completed")

        except Exception as e:
            print(f"  {model_name} SHAPCalculation failed: {str(e)}")

    if not valid_models:
        print("No available models for SHAP analysis")
        return

    # ConstructResultTable
    feature_names = X.columns if isinstance(X, pd.DataFrame) else [f'Feature_{i}' for i in range(X.shape[1])]
    result_df = pd.DataFrame({'Feature': feature_names})

    for model_name, mean_abs, direction in contributions_list:
        result_df[f'{model_name}_Mean_Abs_SHAP'] = mean_abs
        result_df[f'{model_name}_Rank'] = np.argsort(np.argsort(-mean_abs)) + 1
        result_df[f'{model_name}_Direction'] = direction

    # CalculateComprehensive contribution
    mean_cols = [f'{m}_Mean_Abs_SHAP' for m in valid_models]
    result_df['Mean_Contribution'] = result_df[mean_cols].mean(axis=1)
    result_df['Overall_Rank'] = np.argsort(np.argsort(-result_df['Mean_Contribution'])) + 1
    total = result_df['Mean_Contribution'].sum()
    result_df['Contribution_Percentage'] = result_df['Mean_Contribution'] / total * 100

    # Save
    output_path = os.path.join(output_folder, 'Feature_Contributions.xlsx')
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        result_df.to_excel(writer, sheet_name='Contributions', index=False)

        # Beautify
        workbook = writer.book
        sheet = workbook['Contributions']
        header_font = Font(bold=True)
        for cell in sheet[1]:
            cell.font = header_font
        for col in sheet.columns:
            max_len = max(len(str(cell.value)) for cell in col) + 2
            sheet.column_dimensions[get_column_letter(col[0].column)].width = min(max_len, 30)

    print(f"Feature contributionSaved: {output_path}")

    # Plotting
    plot_folder = os.path.join(output_folder, 'Contribution_Plots')
    os.makedirs(plot_folder, exist_ok=True)

    top_n = min(20, len(result_df))
    top_features = result_df.sort_values('Mean_Contribution', ascending=False).head(top_n)

    # Comprehensive plot
    plt.figure(figsize=(10, 8))
    colors = np.where(top_features['Overall_Rank'] <= 10, '#3498db', '#95a5a6')
    plt.barh(top_features['Feature'], top_features['Mean_Contribution'], color=colors)
    plt.xlabel('AverageSHAPvalue absoluteforvalue（Contribution）')
    plt.ylabel('Feature')
    plt.title(f'Feature overall contribution ranking (Top {top_n}）')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, 'Overall_Contribution.png'), dpi=300)
    plt.close()

    # EachModelsubplot
    for model_name in valid_models:
        plt.figure(figsize=(10, 8))
        colors = np.where(top_features[f'{model_name}_Direction'] > 0, '#3498db', '#e74c3c')
        plt.barh(top_features['Feature'], top_features[f'{model_name}_Mean_Abs_SHAP'], color=colors)
        plt.xlabel('AverageSHAPvalue absoluteforvalue（Contribution）')
        plt.ylabel('Feature')
        plt.title(f'{model_name}Model feature contribution ranking (Top {top_n}）')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_folder, f'{model_name}_Contribution.png'), dpi=600)
        plt.close()

    print(f"Contribution plotsSaved: {plot_folder}")


# ========================== Main workflow ==========================

def train_and_predict(train_file_path, test_folder_path, output_folder_path, run_models=None):
    """
    Main workflow entry

    parameters:
        run_models: List, optional 'Space', 'Time', 'Random'
                   Default is None, means run all
    """
    if run_models is None:
        run_models = ['Space', 'Time', 'Random']

    # Unify case
    run_models = [m.capitalize() for m in run_models]
    valid_models = {'Space', 'Time', 'Random'}
    run_models = [m for m in run_models if m in valid_models]

    if not run_models:
        raise ValueError("run_models Must contain 'Space', 'Time', 'Random' at least one")

    print(f"\n{'#' * 60}")
    print(f"# Will run the following models: {', '.join(run_models)}")
    print(f"{'#' * 60}")

    os.makedirs(output_folder_path, exist_ok=True)

    # Read training data
    print(f"\nRead training data: {train_file_path}")
    train_data = pd.read_excel(train_file_path)
    train_data['date'] = pd.to_datetime(train_data['date'])

    # FeatureColumn
    features = [col for col in train_data.columns if col not in ['COMID', 'date', 'in-situ RWT']]

    # Create missing value indicator features
    for col in features:
        train_data[f'{col}_missing'] = train_data[col].isnull().astype(int)

    X = train_data[features + [f'{c}_missing' for c in features]]
    y = train_data['in-situ RWT']

    # Store each model result
    models_dict = {}

    # Run selected CV
    if 'Space' in run_models:
        space_model, space_params, _ = run_space_cv(train_data, X, y, features, output_folder_path)
        models_dict['Space'] = {'model': space_model, 'params': space_params}

    if 'Time' in run_models:
        time_model, time_params, _ = run_time_cv(train_data, X, y, features, output_folder_path)
        models_dict['Time'] = {'model': time_model, 'params': time_params}

    if 'Random' in run_models:
        random_model, random_params, _ = run_random_cv(train_data, X, y, features, output_folder_path)
        models_dict['Random'] = {'model': random_model, 'params': random_params}

    # Batch prediction（Only predict selected models）
    batch_predict(test_folder_path, output_folder_path, models_dict, features)

    # SHAPAnalysis（Only calculate selected models）
    calculate_feature_contributions(X, models_dict, output_folder_path)

    print(f"\n{'=' * 60}")
    print("All completed！")
    print(f"{'=' * 60}")


# ========================== User configuration area ==========================

if __name__ == "__main__":
    # Path configuration
    train_file_path = "D:\\Demo_for_train.xlsx"  #All training and validation samples
    test_folder_path = 'D:\\Demo_for_prediction'  #Input variables for RWT prediction
    output_folder_path = "D:\Demo_result"   # output the estimated RWT


    # Select models to run
    run_models = ['Space']  # Select Model: 'Space','Time','Random'
    train_and_predict(train_file_path, test_folder_path, output_folder_path, run_models=run_models)