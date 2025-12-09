import numpy as np
import pandas as pd
import torch
import joblib
from train_model import Model

def load_model_and_scalers():
    """
    Load the trained neural network and the corresponding input/output scalers.

    Returns:
        model (torch.nn.Module): The trained regression model in eval mode.
        scaler_x (StandardScaler): Fitted scaler for input features.
        scaler_y (StandardScaler): Fitted scaler for output targets.
    """
    model = Model(input_dim=3, output_dim=3)
    model.load_state_dict(torch.load("models/multi_output_model.pth", map_location="cpu"))
    model.eval()

    scaler_x = joblib.load("models/scaler_x.pkl")
    scaler_y = joblib.load("models/scaler_y.pkl")

    return model, scaler_x, scaler_y


def predict_from_csv(input_csv, output_csv="predictions.csv"):
    """
    Run model inference on a CSV containing (Lat_Dec, Lon_Dec, Depth).

    Steps:
    - Load the trained model and scalers.
    - Read input coordinates from `input_csv`.
    - Standardize inputs using the training scaler.
    - Predict temperature, sigma-theta, and fluorescence.
    - Inverse-transform predictions back to physical units.
    - Save the augmented dataset to `output_csv`.

    Args:
        input_csv (str): Path to CSV containing the 3 input coordinate columns.
        output_csv (str): Output file containing predictions appended as columns.
    """
    df = pd.read_csv(input_csv)

    model, scaler_x, scaler_y = load_model_and_scalers()

    X = df[["Lat_Dec", "Lon_Dec", "Depth"]].to_numpy(dtype=np.float32)

    X_scaled = scaler_x.transform(X)

    with torch.no_grad():
        preds_scaled = model(torch.tensor(X_scaled).float()).numpy()

    preds = scaler_y.inverse_transform(preds_scaled)

    df["temp"] = preds[:, 0]
    df["sigmatheta"] = preds[:, 1]
    df["fluor"] = preds[:, 2]

    df.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")


if __name__ == "__main__":
    predict_from_csv("grid.csv", "predictions.csv")
