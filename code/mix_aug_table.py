# https://qiita.com/Y_oHr_N/items/7d2a8ca320f3e658b200

import random as rn

import numpy as np
from sklearn.utils import check_random_state


def mixup(x, y=None, alpha=0.2, p=1.0, random_state=None):
    n, _ = x.shape

    if n is not None and rn.random() < p:
        random_state = check_random_state(random_state)
        l = random_state.beta(alpha, alpha)
        shuffle = random_state.choice(n, n, replace=False)

        x = l * x + (1.0 - l) * x[shuffle]

        if y is not None:
            y = l * y + (1.0 - l) * y[shuffle]

    return x, y


def cutmix(x, y=None, alpha=1.0, p=1.0, random_state=None):
    n, h, w, _ = x.shape

    if n is not None and rn.random() < p:
        random_state = check_random_state(random_state)
        l = np.random.beta(alpha, alpha)
        r_h = int(h * np.sqrt(1.0 - l))
        r_w = int(w * np.sqrt(1.0 - l))
        x1 = np.random.randint(h - r_h)
        y1 = np.random.randint(w - r_w)
        x2 = x1 + r_h
        y2 = y1 + r_w
        shuffle = random_state.choice(n, n, replace=False)

        x[:, x1:x2, y1:y2] = x[shuffle, x1:x2, y1:y2]

        if y is not None:
            y = l * y + (1.0 - l) * y[shuffle]

    return x, y


def cutmix_for_tabular(x, y=None, alpha=1.0, p=1.0, random_state=None):
    print(x.shape)
    n, d = x.shape

    # yはonehotじゃないとだめ
    if y.ndim == 1:
        y = np.array([np.identity(y.max() + 1)[_y] for _y in y])

    if n is not None and rn.random() < p:
        random_state = check_random_state(random_state)
        l = random_state.beta(alpha, alpha)
        mask = random_state.choice([False, True], size=d, p=[l, 1.0 - l])
        mask = np.where(mask)[0]  # シャッフルした行の値に変更する列のid
        shuffle = random_state.choice(n, n, replace=False)  # 行の順番シャッフル
        # print(mask)
        # musk列をシャッフルした値に置き換え
        x[:, mask] = x[shuffle][:, mask]

        if y is not None:
            y = l * y + (1.0 - l) * y[shuffle]

    return x, y
