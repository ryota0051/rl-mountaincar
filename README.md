## これはなに

mountain car を強化学習で play する

## 環境構築方法

1. `docker build -t rl-mountain-car .`で Docker image をビルド

2. `docker run -it --rm -v ${PWD}:/work rl-mountain-car` で jupyter lab を起動(windows の場合、${PWD} が上手く行かない可能性があるので、該当部分を本ディレクトリへの絶対パスに置き換えること。)

3. 以下の notebook のうち、好きなものを実行

- play_with_DQN.ipynb => DQN で mountain car を解いたもの
