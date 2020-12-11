# kaggle キャッサバコンペのコード
- https://www.kaggle.com/c/cassava-leaf-disease-classification



## コンペ概要
- キャッサバの葉の病気の種類を分類
    - アフリカの農家の方が病気のキャッサバを素早く見分けるのに使える
- 特徴量は画像
    - train: 21,367枚
- マルチクラス分類問題
    - 5クラス（病気4種類、健康1種）
- ラベル不均衡あり
    - https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/202651
- ラベルノイズ多い
    - クラウドソーシングでラベル付けされたので、間違ったラベル多い模様
- 評価指標はaccuracy
- コードコンペ