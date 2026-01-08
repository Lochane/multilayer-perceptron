import pandas as pd
import numpy as np
from standardScaler import StandardScaler
from denseLayer import DenseLayer
from loss import LossCategoricalCrossentropy
from activations import ActivationSoftmax, ActivationSigmoid 
from network import SequentialNetwork as MLP

INCLUDE_FEATURES = ["texture_mean", "perimeter_mean", "compactness_mean", "concavity_mean","concave_points_mean"]


#! check if data file exists
df = pd.read_csv("data_splits/train_data.csv")

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
split = int(len(df) * 0.2)
data_train = df.iloc[split:]
data_valid = df.iloc[:split]

scaler = StandardScaler(features=INCLUDE_FEATURES)
x_train = scaler.fit_transform(data_train)
x_valid = scaler.transform(data_valid)
y_train = np.array(data_train["diagnosis"].map({"M": 1, "B": 0}))
y_valid = np.array(data_valid["diagnosis"].map({"M": 1, "B": 0}))

data_train = (x_train, y_train)
data_valid = (x_valid, y_valid)

network = MLP([
	DenseLayer(len(INCLUDE_FEATURES), 40),
	ActivationSigmoid(),
	DenseLayer(40, 40),
	ActivationSigmoid(),
	DenseLayer(40, 20),
	ActivationSigmoid(),
	DenseLayer(20, 2),
	ActivationSoftmax()
])

network.fit(
	network,
	data_train,
	data_valid,
	loss=LossCategoricalCrossentropy(),
	learning_rate=0.01,
	batch_size=32,
	epochs=100
)