import numpy as np
import torch
from sklearn.metrics import root_mean_squared_error, mean_squared_error
from model import Encoder  # Make sure this points to your actual model class
import gzip
import pickle

def load_data(file_path='test0val1normtanh_norm.p.gz'):
    with gzip.open(file_path, 'rb') as f:
        X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test, index_names, f_feature_origin, f_feature_group = pickle.load(f)
    return X_test, y_test, f_feature_group

def load_model(input_size, model_path='saved_models/test0val1normtanh_norm.p_model.pt'):
    model = Encoder(
        input_size=input_size,
        layers=[8192, 4096, 1],
        input_dropout=0.2,
        hidden_dropout=0.5
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def predict(model, X):
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        preds = model(X_tensor).numpy().flatten()
    return preds

def main():
    # Load data
    X_test, y_test, f_feature_group = load_data()
    input_size = X_test.shape[1]

    # Load full model
    model = load_model(input_size)

    # Predict with all features
    y_pred_full = predict(model, X_test)
    rmse_full = root_mean_squared_error(y_test, y_pred_full)
    mse_full = mean_squared_error(y_test, y_pred_full)
    print(f"RMSE (all features): {rmse_full:.4f}")
    print(f"MSE (all features): {mse_full:.4f}")

    # Get unique groups
    groups = np.unique(f_feature_group)
    print(f"Evaluating ablation for feature groups: {groups}")

    for group in groups:
        # Create a copy and zero-out current group features
        X_test_masked = X_test.copy()
        X_test_masked[:, f_feature_group == group] = 0.0

        # Predict with ablated input
        y_pred_masked = predict(model, X_test_masked)
        rmse_masked = root_mean_squared_error(y_test, y_pred_masked)
        mse_masked = mean_squared_error(y_test, y_pred_masked)
        delta = rmse_masked - rmse_full
        print(f"Ablation '{group}': RMSE = {rmse_masked:.4f} (Δ RMSE = {delta:+.4f}), MSE = {mse_masked:.4f}")

if __name__ == "__main__":
    main()


"""
RMSE (all features): 15.3063
MSE (all features): 234.2828

Evaluating ablation for feature groups: ['ECFP_6' 'genomic' 'phys-chem' 'toxicophore']
Ablation 'ECFP_6': RMSE = 17.7575 (Δ RMSE = +2.4512), MSE = 315.3304
Ablation 'genomic': RMSE = 19.6751 (Δ RMSE = +4.3688), MSE = 387.1095
Ablation 'phys-chem': RMSE = 15.2515 (Δ RMSE = -0.0548), MSE = 232.6078
Ablation 'toxicophore': RMSE = 16.2144 (Δ RMSE = +0.9081), MSE = 262.9062
"""