import pandas as pd

fer_path = "/Users/scrab/Documents/centrale/2019-facial-emotions/fer_datasets/fer.csv"
fer_plus_path = "/Users/scrab/Documents/centrale/2019-facial-emotions/fer_datasets/fer_plus_labels.csv"

labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def merge_ferplus_labels(original_df_path, new_labels_path):
    fer = pd.read_csv(original_df_path)
    ferplus_labels = pd.read_csv(new_labels_path)

    columns_to_concatenate = ferplus_labels[
        ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral", 'contempt', 'unknown', 'NF']]

    ferplus = pd.concat([fer, columns_to_concatenate], axis=1)

    return ferplus


def drop_bad_rows(df):
    """
    We drop all rows where NF=10 (corresponds to bad images) and where the majority of voters labeled an image as
     'unknown' or 'contempt' (because we don't use the contempt emotion in our model)
    :param df: fer dataset with fer plus labels
    :return: cleaned fer plus dataset
    """
    df = df[df["NF"] != 10].reset_index(drop=True)
    df['Max'] = df[["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral", 'contempt', 'unknown']].idxmax(
        axis=1)
    df = df[~((df["Max"] == "unknown") | (df["Max"] == "contempt"))].reset_index(drop=True)
    df["emotion"] = df["Max"].map(labels.index)

    return df.drop(["Max", "Usage", "unknown", "contempt", "NF"], axis=1)


def build_emotion_tensor(df, emotions):
    def to_probability_vector(row):
        s = 0
        for x in emotions:
            s += row[x]
        out = [0.]*7
        for i, x in enumerate(emotions):
            out[i] += row[x] / s
        return out

    df["emotions_tensor"] = df.apply(to_probability_vector, axis=1)
    return df


ferplus = drop_bad_rows(merge_ferplus_labels(fer_path, fer_plus_path))
fer_plus_tensor = build_emotion_tensor(ferplus, labels)
fer_plus_tensor.to_csv("./fer_datasets/ferplustensor.csv", index=False)
