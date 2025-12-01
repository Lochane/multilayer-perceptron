import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt

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

def prepvalues(df):
	"""Standardise les colonnes numériques (optionnel).

	Ne modifie pas la colonne `diagnosis`.
	"""
	numeric_cols = df.select_dtypes(include=["number"]).columns
	df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
	return df



def plot_histograms_grid(df, outpath="plots/all_features_grid.png", normalize=False, bins=30, cols=5, figsize_per_plot=(3, 2.5)):
	"""Trace tous les histogrammes dans une seule figure en grille et sauvegarde l'image.

	- `cols` : nombre de colonnes dans la grille.
	- `figsize_per_plot` : taille (width, height) par sous-plot en pouces.
	"""
	# préparer les features
	feature_cols = [c for c in df.columns if c not in ("id", "diagnosis")]
	df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

	diagnoses = sorted(df["diagnosis"].dropna().unique())

	n = len(feature_cols)
	if n == 0:
		raise ValueError("Aucune feature trouvée pour tracer.")

	cols = max(1, int(cols))
	rows = (n + cols - 1) // cols

	figsize = (figsize_per_plot[0] * cols, figsize_per_plot[1] * rows)
	fig, axes = plt.subplots(rows, cols, figsize=figsize)
	axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

	for i, feat in enumerate(feature_cols):
		ax = axes[i]
		for d in diagnoses:
			subset = df[df["diagnosis"] == d][feat].dropna()
			if subset.empty:
				continue
			ax.hist(subset, bins=bins, alpha=0.5, label=str(d), edgecolor="black")
		ax.set_title(feat)
		ax.tick_params(axis='both', which='major', labelsize=8)

	# masquer les axes restants
	for j in range(n, len(axes)):
		axes[j].axis('off')

	# légende globale
	if len(diagnoses) > 0:
		# ajouter légende en dessous
		handles, labels = axes[0].get_legend_handles_labels()
		if handles:
			fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, 0.01))

	plt.tight_layout(rect=[0, 0.03, 1, 1])
	outdir = os.path.dirname(outpath) or '.'
	if outdir:
		os.makedirs(outdir, exist_ok=True)
	plt.savefig(outpath)
	plt.close()


def main(argv=None):
	parser = argparse.ArgumentParser(description="Tracer des histogrammes comparatifs M vs B pour chaque feature.")
	parser.add_argument("--input", default="data.csv", help="Chemin vers le fichier CSV (par défaut data.csv)")
	parser.add_argument("--output", default=os.path.join("plots", "all_features_grid.png"), help="Chemin du fichier unique (png/pdf) pour regrouper tous les histogrammes (par défaut plots/all_features_grid.png)")
	parser.add_argument("--normalize", action="store_true", help="Standardiser les features avant tracé")
	parser.add_argument("--bins", type=int, default=30, help="Nombre de bins pour les histogrammes")
	parser.add_argument("--cols", type=int, default=5, help="Nombre de colonnes dans la grille pour le fichier unique")
	args = parser.parse_args(argv)

	df = pd.read_csv(args.input, header=None, names=columns)

	# vérifier la colonne diagnosis ; si les valeurs sont en 2e colonne non nommée, c'est OK
	if df.shape[1] < 3:
		raise ValueError("Le dataset semble ne pas contenir assez de colonnes.")

	# Comportement unique : produire un seul fichier combiné
	plot_histograms_grid(df, outpath=args.output, normalize=args.normalize, bins=args.bins, cols=args.cols)


if __name__ == "__main__":
	main()

