import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def minor_class_over_sample(train, n_over=2, is_plot=True):
    """サンプル数少ないクラスだけ(3以外)サンプル数倍々にする
    https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/203594"""
    for n in range(n_over):
        for i_cla in [0, 1, 2, 4]:
            train = pd.concat([train, train[train["label"] == i_cla]])

    if is_plot:
        sns.countplot(y=sorted(train["label"].map(name_mapping)), orient="v")
        plt.title("Target distribution")
        plt.show()

    return train


# 全クラスのサンプル数同じになるようにover/under_samplingする
def imblearn_over_sampling(X, y, random_state=42):
    sample = RandomOverSampler(random_state=random_state)
    X_resampled, y_resampled = sample.fit_sample(pd.DataFrame(X), y)

    sns.countplot(y=sorted(X_resampled["label"].map(name_mapping)), orient="v")
    plt.title("Target distribution")
    plt.show()

    return X_resampled


def imblearn_under_sampling(X, y, random_state=42):
    sample = RandomUnderSampler(random_state=random_state)
    X_resampled, y_resampled = sample.fit_sample(pd.DataFrame(X), y)

    sns.countplot(y=sorted(X_resampled["label"].map(name_mapping)), orient="v")
    plt.title("Target distribution")
    plt.show()

    return X_resampled
