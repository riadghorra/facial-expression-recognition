import pandas as pd

fer_path = "/Users/scrab/Documents/centrale/2019-facial-emotions/fer2013.csv"
train_proportion = 0.7
val_proportion = 0.15

def split_fer():
    """
    Split FER dataset into train / val / test by adding a column "usage"
    """
    fer = pd.read_csv(fer_path)
    nrows = len(fer)
    ntrain = int(nrows * train_proportion)
    nval = int(nrows * val_proportion)
    ntest = nrows - ntrain - nval
    attribution = ["train"] * ntrain + ["val"] * nval + ["test"] * ntest
    fer["attribution"] = attribution
    fer.to_csv("attributed_fer.csv", index=False)

split_fer()
