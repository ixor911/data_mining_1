import pandas as pd


def data_convert(data: dict, features: list):
    df = pd.DataFrame()
    for feature in features:
        df[feature] = data.get(feature)

    return df
