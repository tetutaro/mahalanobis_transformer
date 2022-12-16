# mahalanobis_transformer

原点からのユークリッド距離（L2ノルム）を計算したらマハラノビス距離になるようにデータを変換する変換器。

The transformer that transforms data so to squared norm of transformed data becomes Mahalanobis' distance.

## マハラノビス変換

* ユークリッド距離はユークリッド空間における距離であり、ユークリッド空間上のベクトルの長さはノルム（L2ノルム：原点とのユークリッド距離）と呼ばれる
    * 厳密に言えば「ユークリッド距離を距離関数とすれば、ユークリッド空間は距離空間となる」なので、この表現は因果が逆だが、こっちの方が分かりやすいかと・・・
* 実は、ユークリッド空間の各軸（＝特徴量）は直行している（相関が無い）ことが前提である
* しかし一般的なデータの場合、特徴量間には少なからず相関があることがほとんどである
* 特徴量間に相関がある場合の距離として、[マハラノビス距離](https://ja.wikipedia.org/wiki/%E3%83%9E%E3%83%8F%E3%83%A9%E3%83%8E%E3%83%93%E3%82%B9%E8%B7%9D%E9%9B%A2) が良く使われる
	* $D_{M}(X)=\sqrt{(X-\mu)S^{-1}(X-\mu)^{T}}$
        * 上記の $X$ は pandas, scikit-learn における DataFrame を想定し (n\_sample, n\_feature) 行列を考えているので、Wikipedia の表記とは行と列が逆（転地している）
        * $S$ はデータの [共分散行列](https://ja.wikipedia.org/wiki/%E5%88%86%E6%95%A3%E5%85%B1%E5%88%86%E6%95%A3%E8%A1%8C%E5%88%97) （ $(X-\mu)^{T}(X-\mu)$ である (n\_feature, n\_feature) 行列（Wikipedia の表記とは行と列が逆））
        * この場合の原点は、各特徴量毎の平均ベクトル $\mu$
        * 行列からベクトルを引き算しちゃっているけど、numpy でも暗黙的に計算してくれるし、良い感じに脳内補間してください
* マハラノビス距離は便利
    * 軸間に相関が無い（ホワイトノイズ化）
    * ｎ次元空間におけるマハラノビス距離の２乗の分布は、自由度ｎのカイ２乗分布に従う
        * 各変数が正規分布に従う場合（ [多変量正規分布](https://ja.wikipedia.org/wiki/%E5%A4%9A%E5%A4%89%E9%87%8F%E6%AD%A3%E8%A6%8F%E5%88%86%E5%B8%83) ）
    * 実際に使う場合、この「各変数が正規分布に従う」ということをあまり気にしないで使っても、意外と良い結果が得られたりする
    * そこで、カイ２乗分布（の [信頼区間](https://ja.wikipedia.org/wiki/%E4%BF%A1%E9%A0%BC%E5%8C%BA%E9%96%93) ）から外れるものを異常値であるとすれば、「異常検知」（Anomaly Detection）が実現でき、これは意外と良く使われたりする
    * 特徴量の中に、実数の変数だけではなく、バイナリ変数が混じっていても、マハラノビス距離を使うことが出来ちゃったりする
        * バイナリ変数：値として 0 と 1 だけが存在する変数
        * そのまま使っちゃっても意外と大丈夫
	* なので、カテゴリ変数がある場合でも、マハラノビス距離が使える
        * カテゴリ変数をダミー変数化する
        * もちろんマルチコ防止のためにダミー変数を１個削除するんやで
* 特徴量の中にカテゴリ変数（バイナリ変数）が混じっていても距離が計算できるというのは非常に強力なので、是非使いたい
    * 例えば k-Means とか k-Means とか
    * しかし、このようなモデルのライブラリ（scikit-learn 等）は、距離の計算は決め打ちでユークリッド距離を使うことがほとんどで、マハラノビス距離を使うには自前で実装しなければならない
    * そして多くの機械学習モデルはユークリッド距離を使う！
* そこで、ユークリッド距離（L2ノルム）を計算したらマハラノビス距離になるように、データの方を変換してしまえば、色々なライブラリをそのまま使っても距離としてマハラノビス距離を利用することが出来る
* この「ユークリッド距離を計算したらそれは実はマハラノビス距離になるようにデータを変換する」変換を「マハラノビス変換」と呼ぶことにする

## マハラノビス変換の方法

* $X$ を正規化する
    * $\mu=0$ になる
* $S^{-1}$ を [コレスキー分解](https://ja.wikipedia.org/wiki/%E3%82%B3%E3%83%AC%E3%82%B9%E3%82%AD%E3%83%BC%E5%88%86%E8%A7%A3) する
    * $S$ は共分散行列なので、明らかに [対称行列](https://ja.wikipedia.org/wiki/%E5%AF%BE%E7%A7%B0%E8%A1%8C%E5%88%97) である
    * 対称行列の逆行列は対称行列である
        * $A$ を対称行列とする（ $A^{T}=A$ ）
        * 逆行列の定義より $AA^{-1}=I$
        * 両辺の転地を取る $(A^{-1})^{T}A^{T}=I^{T}$
        * 対称行列なので $(A^{-1})^{T}A=I$
        * $A$ を右辺にもっていくと $(A^{-1})^{T}=A^{-1}$
        * $A^{-1}$ は対称行列である
    * よって $S^{-1}=LL^{T}$ のように、コレスキー分解の共役転置は単なる転地になる
	* $L$ は下三角行列とする
* $D_{M}(X)^2=XLL^{T}X^{T}=(XL)(XL)^{T}$
* よって $XL$ を２乗する（内積を取る）とそれがマハラノビス距離の２乗になる
* すなわち $XL$ がマハラノビス変換である

### 頭の整理

* $XS^{-1}X^{T}=X(X^{T}X)^{-1}X^{T}=XX^{-1}(X^{T})^{-1}X^{T}=(XX^{-1})(XX^{-1})^{T}$ じゃないかよ、と思ってしまうことがある
* これは間違い
    * $(AB)^{-1}=B^{-1}A^{-1}$ は、$A$, $B$ ともに [正則行列](https://ja.wikipedia.org/wiki/%E6%AD%A3%E5%89%87%E8%A1%8C%E5%88%97) の場合のみ
    * $(A^{T})^{-1}=(A^{-1})^{T}$ も、$A$ が正則行列の場合のみ
    * $X$ は一般的に正則行列ではない
        * n\_sample >> n\_feature

## Examples

The library `mahalanobis_transfomer` has the `MahalanobisTransformer` class.
`MahalanobisTransformer` has scikit-learn like API.

```
>>> import numpy as np
>>> from mahalanobis_transformer import MahalanobisTransformer
>>> m = np.array([1, 2])
>>> K = np.array([[2, 1], [1, 2]])
>>> np.random.seed(seed=12)
>>> X = np.random.multivariate_normal(mean=m, cov=K, size=3)
>>> X
array([[ 0.903,  0.939]
       [ 1.906,  0.500]
       [ 1.163, -0.008]])
>>> transformer = MahalamobisTransformer().fit(X)
>>> Z = transformer.transform(X)
>>> Z
array([[-0.619,  0.975]
       [ 1.154,  0.049]
       [-0.534, -1.024]])
>>> transformer.inverse_transform(Z)
array([[ 0.903,  0.939]
       [ 1.906,  0.500]
       [ 1.163, -0.008]])
>>> from sklearn.preprocessing import StandardScaler
>>> from scipy.spatial.distance import cdist
>>> ss = StandardScaler().fit(X)
>>> X_normed = ss.transform(X)
>>> vi = np.linalg.inv(np.cov(X_normed, rowvar=False))
>>> cdist(X_normed, np.zeros((1, 2)), metric='mahalanobis', VI=vi).ravel()
array([1.155, 1.155, 1.155])
>>> cdist(Z, metric='euclidean').ravel()
array([1.155, 1.155, 1.155])
```

## Methods

| Method | Description |
| :--- | :--- |
| fit(X[, y]) | Fit transformer by checking X. |
| fit_transform(X[, y]) | Fit to data, then transform it. |
| inverse_transform(X) | Inverse Mahalanobis transform X. |
| transform(X) | Mahalanobis transform X. |
