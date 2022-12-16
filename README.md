# mahalanobis_transformer

原点からのユークリッド距離（２乗ノルム）を計算したらマハラノビス距離になるようにデータを変換する変換器。

A transformer that transforms data so to the squared norm of the transformed data becomes Mahalanobis' distance.

## マハラノビス変換

* ユークリッド距離はユークリッド空間における距離であり、ユークリッド空間上のベクトルの長さはノルム（２乗ノルム：原点とのユークリッド距離）と呼ばれる
* ユークリッド空間の各軸は直行していることが前提である
* しかし一般的なデータの場合、特徴量毎に少なからず相関があることがほとんどである
* 軸間に相関がある場合の距離として、[マハラノビス距離](https://ja.wikipedia.org/wiki/%E3%83%9E%E3%83%8F%E3%83%A9%E3%83%8E%E3%83%93%E3%82%B9%E8%B7%9D%E9%9B%A2) が良く使われる
	* $d_{M}(\mathbf{x}_i, \mathbf{x}_{i'})^2=(\mathbf{x}_i-\mu)^{T}S^{-1}(\mathbf{x}_{i'}-\mu)$
        * この場合の原点は、各特徴量毎の平均（$\mu$）
* マハラノビス距離は便利
    * 軸間に相関が無い（ホワイトノイズ化）
    * ｎ次元空間におけるマハラノビス距離の２乗の分布は、自由度ｎのカイ２乗分布に従う
        * 各変数が正規分布に従う場合
    * 実際に使う場合、この「各変数が正規分布に従う」ということをあまり気にしないで使っても、意外と良い結果が得られたりする
    * そこで、カイ２乗分布から外れるものが異常値であるとする「異常検知」（Anomaly Detection）によく使われる
    * 特徴量として、実数の変数だけではなく、バイナリ変数が混じっていても、マハラノビス距離を使うことが出来ちゃったりする
        * そのまま使っちゃっても意外と大丈夫
	* なので、カテゴリ変数がある場合でも、マハラノビス距離が使える
        * カテゴリ変数をダミー変数化する
        * もちろんマルチコ防止のためにダミー変数１個削除するんやで
* 特徴量の中にカテゴリ変数（バイナリ変数）が混じっても距離が計算できるというのは非常に強力なので、是非使いたい
    * 例えば k-Means とか
    * しかし、このようなモデルのライブラリは、距離の計算は決め打ちでユークリッド距離を使うことがほとんどで、マハラノビス距離を使うには自前で実装しなければならない
* そこで、ユークリッド距離（２乗ノルム）を計算したらマハラノビス距離を計算した結果になるようにデータの方を変換すれば、色々なライブラリをそのまま使っても距離としてマハラノビス距離を利用することが出来る
* この「ユークリッド距離を計算したらそれは実はマハラノビス距離になるようにデータを変換する」変換を「マハラノビス変換」と呼ぶことにする

## マハラノビス変換の方法

* $X$を正規化する（$\mu=\mathbf{0}$）
* $S^{-1}$を [コレスキー分解](https://ja.wikipedia.org/wiki/%E3%82%B3%E3%83%AC%E3%82%B9%E3%82%AD%E3%83%BC%E5%88%86%E8%A7%A3) する（$S^{-1}=LL^{T}$）
	* $L$は下三角行列
* $d_{M}(\mathbf{x}_i, \mathbf{x}_{i'})^2=\mathbf{x}_i^TLL^{T}\mathbf{x}_{i'}$であるが、pythonプログラム上は特徴量ベクトルになるよう転置してる
* よって全データ間のマハラノビス距離行列は$D_{M}(X, X) =XLL^{T}X^T=(XL)(XL)^T \in \mathbb{R}^{N \times N}$
	* $(XL)^{T}$が各データをマハラノビス変換した\newline 各データベクトルの行列
	* なので$D_{M}(X,X)$は変換後データベクトルの内積＝変換後のユークリッド距離行列
* $XL$が求める変換後のデータ（特徴量ベクトルの行列）
