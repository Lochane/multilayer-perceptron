import sys
from utils import stats_tools
import padas as pd

def prepvalues(df):
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
    return df


if __name__ = "__main__":
	if len(sys.argv) != 1:
		sys.exit(1)

	dataset = pd.read_csv("data.csv")
