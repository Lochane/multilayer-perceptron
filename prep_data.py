import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Colonnes attendues (dataset sans en-tête : id, diagnosis, puis 30 features)
columns = [
"id","diagnosis",
"radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
"compactness_mean","concavity_mean","concave_points_mean","symmetry_mean",
"fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se",
"smoothness_se","compactness_se","concavity_se","concave_points_se","symmetry_se",
"fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst",
"smoothness_worst","compactness_worst","concavity_worst","concave_points_worst",
"symmetry_worst","fractal_dimension_worst"
]

EXCLUDE_MEAN = {"fractal_dimension_mean", "symmetry_mean", "smoothness_mean", "radius_mean", "area_mean"}

# def prepvalues(df):
# 	"""Standardise les colonnes numériques (optionnel).

# 	Ne modifie pas la colonne `diagnosis`.
# 	"""
# 	numeric_cols = df.select_dtypes(include=["number"]).columns
# 	df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
# 	return df



def plot_histograms_grid(df, bins=30):
	features = [c for c in df.columns if c not in ("id", "diagnosis")]
	n = len(features)

	cols = 5
	rows = (n + cols - 1) // cols
	fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
	axes = axes.flatten()

	for i, feature in enumerate(features):
		for label in ("M", "B"):
			subset = df[df["diagnosis"] == label][feature].dropna()
			axes[i].hist(subset, bins=bins, alpha=0.5, label=label, density=True, edgecolor='black')
		axes[i].set_title(feature)

	for j in range(i + 1, len(axes)):
		axes[j].axis('off')

	axes[0].legend()	
	plt.tight_layout()
	outdir = os.path.join("plots")
	os.makedirs(outdir, exist_ok=True)
	plt.savefig(os.path.join(outdir, "all_features_histograms.png"))
	plt.close(fig)


def plot_scatter_matrix(df):
	feature_cols = [c for c in df.columns if c.endswith("_mean") and c not in EXCLUDE_MEAN]
	df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
	data = df[feature_cols + ["diagnosis"]].dropna()

	g = sns.pairplot(data, hue="diagnosis", diag_kind='kde', plot_kws={'alpha': 0.5, 's': 20})
	g.fig.tight_layout()
	outdir = os.path.join("plots")
	os.makedirs(outdir, exist_ok=True)
	g.fig.savefig(os.path.join(outdir, "scatter_matrix.png"))
	plt.close(g.fig)


def main(argv=None):
	if len(sys.argv) != 2:
		print("Usage: python prep_data.py <filename>")
		sys.exit(1)

	df = pd.read_csv(sys.argv[1], header=None, names=columns)

	# vérifier la colonne diagnosis ; si les valeurs sont en 2e colonne non nommée, c'est OK
	if df.shape[1] < 3:
		raise ValueError("Le dataset semble ne pas contenir assez de colonnes.")

	# Comportement unique : produire un seul fichier combiné
	plot_histograms_grid(df, bins=30)
	plot_scatter_matrix(df)
	df = df.sample(frac=1, random_state=42).reset_index(drop=True)
	test_ratio = 0.2
	test_size = int(len(df) * test_ratio)
	train_df = df.iloc[test_size:]
	test_df = df.iloc[:test_size]

	outdir = os.path.join("data_splits")
	os.makedirs(outdir, exist_ok=True)

	train_df.to_csv(os.path.join(outdir, "train_data.csv"), index=False)
	test_df.to_csv(os.path.join(outdir, "test_data.csv"), index=False)
	# Only compute correlations on numeric columns to avoid casting errors from labels
	corr_matrix = train_df.corr(numeric_only=True)
	print(corr_matrix)

if __name__ == "__main__":
	main()

