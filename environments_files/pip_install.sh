conda activate iguane

# 先给 pip 换国内源，加速明显
mkdir -p ~/.pip
echo '[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn' > ~/.pip/pip.conf

# 一次装容易再卡，所以按“大坨”分组装，每装完一组落盘
# （组已帮你调好顺序，TensorFlow / torch 放最前，避免后面编译）
cat requirements.txt | grep -Ei 'tensorflow|tensorboard|nvidia-cuda|keras' > tf.txt
cat requirements.txt | grep -Ei 'torch|jax' > torch.txt
cat requirements.txt | grep -Ei 'scipy|numpy|scikit-learn|pandas|matplotlib|seaborn|statsmodels' > sci.txt
cat requirements.txt | grep -Ei 'jupyter|notebook|ipy|lab' > jupyter.txt
# 剩余
grep -v -f tf.txt -f torch.txt -f sci.txt -f jupyter.txt requirements.txt > rest.txt

# 逐组安装，哪组报错立刻知道是谁
for grp in tf.txt torch.txt sci.txt jupyter.txt rest.txt; do
  echo "==== Installing $grp ===="
  pip install -r $grp
done