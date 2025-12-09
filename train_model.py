import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Neural Network Model
class Model(nn.Module):
    """
    Fully connected neural network for predicting temperature, sigma-theta, 
    and fluorescence from (lat, lon, depth). The architecture uses several 
    GELU-activated hidden layers and outputs three continuous values 
    corresponding to the physical oceanographic variables.
    """
    def __init__(self, input_dim=3, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# Main Training Routine
def main():
    """
    Train the multi-output regression model using CalCOFI CTD data.

    Steps:
    - Load the required CTD dataset columns.
    - Remove rows missing any target variables.
    - Extract features (lat, lon, depth) and targets (TempAve, SigThetaTS1, FluorV).
    - Compute and save spatial bounds for downstream grid generation.
    - Standardize inputs and outputs.
    - Split into train, validation, and test sets.
    - Train the model with Adam and ReduceLROnPlateau scheduler.
    - Save the best model checkpoint, the input/output scalers, and grid bounds.
    - Compute RMSE in physical units on the test set.
    """
    # Load dataset
    cols = ["Lat_Dec", "Lon_Dec", "Depth",
            "TempAve", "SigThetaTS1", "FluorV"]

    df = pd.read_csv("data/20-2507_CTD_001-055D.csv", usecols=cols)

    # Drop rows missing targets
    df = df.dropna(subset=["TempAve", "SigThetaTS1", "FluorV"])

    # Features (inputs) / Targets (outputs)
    X = df[["Lat_Dec", "Lon_Dec", "Depth"]].to_numpy(dtype=np.float32)
    Y = df[["TempAve", "SigThetaTS1", "FluorV"]].to_numpy(dtype=np.float32)

    # Save spatial bounds for grid generation
    bounds = {
        "lat_min": float(df["Lat_Dec"].min()),
        "lat_max": float(df["Lat_Dec"].max()),
        "lon_min": float(df["Lon_Dec"].min()),
        "lon_max": float(df["Lon_Dec"].max()),
        "depth_min": float(df["Depth"].min()),
        "depth_max": float(df["Depth"].max()),
    }

    os.makedirs("models", exist_ok=True)
    pd.DataFrame([bounds]).to_csv("models/grid_bounds.csv", index=False)

    # Scaling
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_x.fit_transform(X)
    Y_scaled = scaler_y.fit_transform(Y)
    
    # Train / Validation / Test Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, Y_scaled, train_size=0.8, random_state=42)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=0.5, random_state=42)

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=64, shuffle=True
    )
    
    # Setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(input_dim=3, output_dim=3).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=10
    )
    
    # Training Loop
    num_epochs = 300
    print(f"Training on device: {device}\n")

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            preds = model(xb)
            loss = criterion(preds, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        
        # Validation Loss
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val.to(device)).cpu()
            val_loss = criterion(val_preds, y_val).item()

        scheduler.step(val_loss)

        print(f"Epoch {epoch:03d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/multi_output_model.pth")

    
    # Final Test Evaluation
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_test.to(device)).cpu().numpy()

    # Unscale to physical units
    preds_phys = scaler_y.inverse_transform(preds_scaled)
    test_phys = scaler_y.inverse_transform(y_test.numpy())

    rmse = np.sqrt(((preds_phys - test_phys) ** 2).mean(axis=0))

    print("\nTest RMSE (physical units):")
    print(f"  Temperature (°C):       {rmse[0]:.3f}")
    print(f"  Sigma-Theta (kg/m³):    {rmse[1]:.3f}")
    print(f"  Fluorescence (a.u.):    {rmse[2]:.3f}")
    
    # Save scalers
    import joblib
    joblib.dump(scaler_x, "models/scaler_x.pkl")
    joblib.dump(scaler_y, "models/scaler_y.pkl")

    print("\nSaved model, scalers, and bounds to /models/")
    print("Training complete.")


if __name__ == "__main__":
    main()
