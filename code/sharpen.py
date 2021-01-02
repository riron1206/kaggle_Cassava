# https://github.com/FelixAbrahamsson/mixmatch-pytorch/blob/master/mixmatch_pytorch/sharpen.py
# https://github.com/FelixAbrahamsson/mixmatch-pytorch/blob/master/mixmatch_pytorch/guess_targets.py
import torch


def sharpen(probabilities, T=0.5):

    if probabilities.ndim == 1:
        tempered = torch.pow(probabilities, 1 / T)
        tempered = tempered / (torch.pow((1 - probabilities), 1 / T) + tempered)

    else:
        tempered = torch.pow(probabilities, 1 / T)
        tempered = tempered / tempered.sum(dim=-1, keepdim=True)

    return tempered


def test_func():
    """
    テスト駆動開発での関数のテスト関数
    test用関数はpythonパッケージの nose で実行するのがおすすめ($ conda install -c conda-forge nose などでインストール必要)
    →noseは再帰的にディレクトリ探索して「Test」や「test」で始まるクラスとメソッドを実行する
    $ cd <このモジュールの場所>
    $ nosetests -v -s --nologcapture <本モジュール>.py  # 再帰的にディレクトリ探索して「Test」や「test」で始まるクラスとメソッドを実行
      -s付けるとprint()の内容出してくれる
      --nologcapture付けると不要なログ出さない
    """
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = np.array([[0.1, 0.6, 0.2, 0.1]])
    # x = np.array([[0.1, 0.6, 0.2, 0.1], [0.6, 0.3, 0.08, 0.02]])
    print("x:", x)

    x = torch.from_numpy(x.astype(np.float32)).clone()
    probabilities = sharpen(x, T=0.5).to(device)
    print("sharpen x:", probabilities)
