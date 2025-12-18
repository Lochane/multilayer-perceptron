import pandas as pd
import numpy as np
from utils import stats_tools as st

INCLUDE_FEATURES = ["texture_mean", "perimeter_mean", "compactness_mean", "concavity_mean","concave_points_mean",]

def main(argv=None):
	train_df = pd.read_csv("data_splits/train_data.csv")


	# Séparer les caractéristiques et les étiquettes
	x = train_df[INCLUDE_FEATURES]
	y = train_df["diagnosis"].map({"M": 1, "B": 0})

	# Preprocessing
	x_mean = []
	x_std = []
	n_features = len(INCLUDE_FEATURES)

	for feature in INCLUDE_FEATURES:
		feature_values = x[feature].tolist()
		x_mean.append(st.ft_mean(feature_values))
		x_std.append(st.ft_std_dev(feature_values))

	x_norm = []
	for patient in range(len(x)):
		normed_features = []
		for feature in range(n_features):
			val = x.iloc[patient, feature]
			if val is None or (isinstance(val, (int, float)) and not (val != val)):
				val = x_mean[feature]
			std = x_std[feature] if x_std[feature] != 0 else 1.0
			normed_features.append((val - x_mean[feature]) / std)
		x_norm.append(normed_features)

	x_norm = np.array(x_norm, dtype=float)
	y = np.array(y.tolist(), dtype=float)

if __name__ == "__main__":
	main()