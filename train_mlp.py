import pandas as pd

def main(argv=None):
	train_df = pd.read_csv("data_splits/train_data.csv")
	test_df = pd.read_csv("data_splits/test_data.csv")

	# Séparer les caractéristiques et les étiquettes
	X_train = train_df.drop(columns=["diagnosis", "id"])
	y_train = train_df["diagnosis"].map({"M": 1, "B": 0})
	X_test = test_df.drop(columns=["diagnosis", "id"])
	y_test = test_df["diagnosis"].map({"M": 1, "B": 0})
	

if __name__ == "__main__":
	main()