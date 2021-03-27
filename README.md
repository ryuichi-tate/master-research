# maseter research



## tsModel.py
人工データを作成する関数。
- <code>ARIMA()</code>
- <code>SARIMA()</code>
- <code>NeuralNet()</code>：4層Neural Networkを用いた非線形自己回帰時系列モデル

の三つある。特に<code>NeuralNet()</code>の引数について、
- model_random_seed：人工データ作成に用いるニューラルネットのパラメータを決定する乱数のシード
- p：次数pの自己回帰モデル
- q：次数qの移動平均モデル
- n_unit：4層ニューラルネットの中間層のユニット数（list形式）
- mu, sigma：innovationの平均と分散
- N；作成する人工データの長さ
- random_seed：innovation系列生成の乱数シード
- burn_in：burn-in期間をintで与える（defaultでok）
- randomness：innovationの分布を決定する
  - "normal"：正規分布
  - "uniform"：一様分布
  - "gamma"：ガンマ分布
  - "normal&uniform"：系列の前半2/Nは正規分布、後半は一様分布
  - 他
- return_net：Trueならネットワークのインスタンスが返ってくる
- return_innovation：Trueならinnovation系列が返ってくる

## networkModel.py
時系列モデルの構築に用いるニューラルネットのクラスが定義されている。GeneratorとDiscriminatorとPredictorの三つの役割分担がある。

## Wasserstein.py
一次元の正規分布と一次元の経験分布とのWasserstein距離を計算している。Discriminatorではなく直接Wasserstein距離を損失として用いる場合に使う。

![\begin{align*}
W_p=\left\{\sum_{n=1}^N\int_{\Phi^{-1}(\frac{n-1}{N})}^{\Phi^{-1}(\frac{n}{N})}|x_n-x|^p\frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}dx\right\}^{\frac{1}{p}}
\end{align*}
](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0AW_p%3D%5Cleft%5C%7B%5Csum_%7Bn%3D1%7D%5EN%5Cint_%7B%5CPhi%5E%7B-1%7D%28%5Cfrac%7Bn-1%7D%7BN%7D%29%7D%5E%7B%5CPhi%5E%7B-1%7D%28%5Cfrac%7Bn%7D%7BN%7D%29%7D%7Cx_n-x%7C%5Ep%5Cfrac%7B1%7D%7B%5Csqrt%7B2%5Cpi%7D%7De%5E%7B-%5Cfrac%7Bx%5E2%7D%7B2%7D%7Ddx%5Cright%5C%7D%5E%7B%5Cfrac%7B1%7D%7Bp%7D%7D%0A%5Cend%7Balign%2A%7D%0A)<br>
を計算する。<code>p=1</code>の場合のみ計算できる。

