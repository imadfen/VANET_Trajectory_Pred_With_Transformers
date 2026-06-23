# Unsupervised Transformer Model for Time Series Data (VANETs Vehicle Trajectory Adaptation)

**Notice:** This project has been adapted from its original form (predicting pedestrian trajectories) to focus on **Vehicle Trajectory Prediction using VANETs (Vehicular Ad-hoc Networks) data**. The original workflow and intentions are preserved below.

## VANETs Vehicle Trajectory Prediction
This adapted model processes individualized vehicle tracking data, taking into account complex kinematic states and network conditions. Features processed by the model include:
`Time, X, Y, Speed, Acceleration, Heading, AngularVelocity, LaneID, LaneDist`, nearest neighbors' network telemetry, `AvgDistToSender`, `AvgMsgDelay`, and `PacketLossRate`.

### Setup for VANETs Data
Place your individual car CSV files (e.g., `data_car_160_t18099.csv`) inside a dedicated folder, for example `resources/VANET_data/`.

```bash
mkdir -p resources/VANET_data
# Move your downloaded `.csv` files into this directory
```

### Run VANETs Training
Use the following command to train the model over your VANETs dataset. Note the `--pattern` argument helps locate only the car tracking datasets:
```bash
python main.py --output_dir ./experiments --comment "pretraining over VANETs" --name VANETDataset_pretrained --data_dir resources/VANET_data --data_class sind --pattern "data_car_" --pos_encoding learnable --harden
```

---

## Original Project: Unsupervised Transformer Model for Time Series Data

This code includes a transformer-based framework for unsupervised representation learning of multivariate time series, inspired by [Zerveas et al.](https://dl.acm.org/doi/10.1145/3447548.3467401).

The model is trained using the missing value imputation task to create embeddings that potentially extract complex features from pedestrian trajectories. 
These embeddings are subsequently used for clustering to reveal different behaviors. These behavior clusters are combined with data-driven reachability analysis, yielding an end-to-end data-driven approach to predicting the future motion of pedestrians.
