import itertools
import numpy as np

def filter_outliers(df):
    outliers_list  = []

    for feature in df.columns:
        Q1 = np.percentile(df.loc[:, feature], 25)
        Q3 = np.percentile(df.loc[:, feature], 75)
        step = 1.5 * (Q3 - Q1)
        outliers_found = df.loc[~((df[feature] >= Q1 - step) & (df[feature] <= Q3 + step)), :]
        outliers_list.append(list(outliers_found.index))

    outliers = list(itertools.chain.from_iterable(outliers_list))
    uniq_outliers = list(set(outliers))
    dup_outliers = list(set([x for x in outliers if outliers.count(x) > 1]))
    filtered_df = df.drop(df.index[dup_outliers]).reset_index(drop = True)
    return filtered_df