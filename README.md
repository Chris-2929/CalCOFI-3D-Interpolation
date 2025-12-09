# CalCOFI 3D Interpolation Model

This project trains a neural network to predict temperature, sigma-theta, and fluorescence across a 3-D ocean grid using CalCOFI CTD data. The workflow is:

1. Train a PyTorch model on CalCOFI CTD profiles
2. Save spatial bounds and scaling information
3. Generate a regular lat–lon–depth grid
4. Predict physical variables across the entire grid
5. Visualize the 3-D field on a tiled 4-GPU OpenGL display


## Project Structure
```
train_model.py          – Train neural network & save model + scalers
make_grid.py            – Build a 3D grid for inference
predict.py              – Run predictions over the grid
visualize.py            – MPI/OpenGL visualization

data/20-2507_CTD_001-055D.csv

models/
    multi_output_model.pth
    scaler_x.pkl
    scaler_y.pkl
    grid_bounds.csv

grid.csv
predictions.csv
```

## Workflow


1. Train the model:
    python3 train_model.py

   Produces:
   - models/multi_output_model.pth
   - models/scaler_x.pkl, models/scaler_y.pkl
   - models/grid_bounds.csv

2. Generate prediction grid:
    python3 make_grid.py

   Produces:
   - grid.csv

3. Run predictions:
    python3 predict.py

   Produces:
   - predictions.csv

4. Visualize (requires MPI + OpenGL):
    mpiexec --pernode -hostfile hostfilename python3 visualize.py --var temp

   Options for --var:
   - temp
   - sigma
   - fluor


## Data Attribution
This project uses CalCOFI (California Cooperative Oceanic Fisheries Investigations) oceanographic data.  
CalCOFI data are licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).  
CTD data available at: https://calcofi.org/data/oceanographic-data/ctd-cast-files/

### Citation
Data obtained from the California Cooperative Oceanic Fisheries Investigations (CalCOFI).  
Data available at https://calcofi.org/.

## Model Details
- Inputs: latitude, longitude, depth
- Outputs: temperature (°C), sigma-theta (kg/m³), fluorescence (a.u.)

- Architecture: fully connected MLP
- Activation: GELU
- Loss: MSE
- Optimizer: Adam
- Training: 80/10/10 split with LR scheduler

## Requirements
See requirements.txt for package dependencies.


## License
Code: MIT License (see LICENSE)

CalCOFI data remain licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).