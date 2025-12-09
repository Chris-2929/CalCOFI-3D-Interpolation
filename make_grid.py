import pandas as pd
import numpy as np

def main():
    """
    Generate a 3-D prediction grid covering the spatial bounds of the training data.

    Loads bounding box information saved during model training, builds a regular 
    lat-lon-depth mesh grid at fixed resolution, flattens the grid into a table, 
    and writes it to `grid.csv` for downstream model inference.
    """
    # Load bounds saved during training
    bounds = pd.read_csv("models/grid_bounds.csv").iloc[0]

    lat_min, lat_max = bounds["lat_min"], bounds["lat_max"]
    lon_min, lon_max = bounds["lon_min"], bounds["lon_max"]
    depth_min, depth_max = bounds["depth_min"], bounds["depth_max"]

    # Resolution of grid
    N_LAT = 40
    N_LON = 40
    N_DEPTH = 60

    lat_vals = np.linspace(lat_min, lat_max, N_LAT)
    lon_vals = np.linspace(lon_min, lon_max, N_LON)
    depth_vals = np.linspace(depth_min, depth_max, N_DEPTH)

    LAT, LON, DEPTH = np.meshgrid(lat_vals, lon_vals, depth_vals, indexing="ij")

    df = pd.DataFrame({
        "Lat_Dec": LAT.ravel(),
        "Lon_Dec": LON.ravel(),
        "Depth": DEPTH.ravel()
    })

    df.to_csv("grid.csv", index=False)
    print(f"Generated grid.csv with {len(df)} points.")

if __name__ == "__main__":
    main()
