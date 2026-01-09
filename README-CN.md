```bash
source ~/.profile
conda activate iguane
export CUDA_VISIBLE_DEVICES=6
python ./all_in_one.py --in-mri ~/datasets/hcp/100206/MNINonLinear/T1w.nii.gz --out-mri ./my_test/hcp/100206/MNINonLinear/T1w_iguane.nii.gz
```

RuntimeError: jaxlib version 0.4.30 is newer than and incompatible with jax version 0.4.8. Please update your jax and/or jaxlib packages.
(iguane) lengjingcheng@expm9:~/codes/iguane_harmonization$ pip uninstall -y jax jaxlib
Found existing installation: jax 0.4.8
Uninstalling jax-0.4.8:
  Successfully uninstalled jax-0.4.8
Found existing installation: jaxlib 0.4.30
Uninstalling jaxlib-0.4.30:
  Successfully uninstalled jaxlib-0.4.30

# Github REDME
> https://yiyibooks.cn/arxiv/2402.03227v4/index.html
> 
> https://github.com/RocaVincent/iguane_harmonization

`# IGUANe模型的图像协调（IGUANe harmonization） 本代码仓库提供了使用IGUANe模型进行磁共振（MR）图像协调的代码。完整的方法及验证实验已在一篇同行评审论文中详细阐述。该模型经训练后，可用于T1加权脑图像的协调处理。`

本仓库中的脚本适用于扩展名为.nii.gz的Nifti格式文件。

## 目录

- 安装说明
    
    - Anaconda环境
        
    - 预处理工具
        
- 一站式处理（All-in-one）
    
- 预处理（Preprocessing）
    
- 协调推理（Harmonization inference）
    
- 协调训练（Harmonization training）
    
- 预测模型（Prediction models）
    
- 元数据（Metadata）
    

## 安装说明

IGUANe模型可在含GPU和不含GPU的环境下使用。但需注意，GPU环境下运行速度更快，尤其是在执行HD-BET脑提取步骤时。

### Anaconda环境

如需使用IGUANe进行图像协调，可通过文件`./iguane.yml`创建名为iguane的Anaconda环境，命令如下：  
`conda env create -f ./iguane.yml`

若要使用GPU，需先设置环境变量（每次激活环境后使用前均需执行此操作）：

```
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib
```

在我们的实验环境中，曾遇到libdevice相关问题，需创建符号链接解决，命令如下：

```
mkdir -p $CONDA_PREFIX/lib/nvvm/libdevice
ln -s $CONDA_PREFIX/lib/libdevice.10.bc $CONDA_PREFIX/lib/nvvm/libdevice
```

### 预处理工具

预处理步骤需额外安装以下工具：

- FSL（功能性磁共振成像软件库）
    
- ANTs（高级归一化工具包）
    
- HD-BET（脑提取工具）：建议在之前创建的iguane环境中安装，步骤如下：
    
    1. 若使用GPU（如有GPU建议优先使用），需先按官方指引安装PyTorch，例如执行命令：`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
        
    2. 安装HD-BET：`git clone https://github.com/MIC-DKFZ/HD-BET.git &amp;&amp; cd HD-BET &amp;&amp; git checkout ae16068 &amp;&amp; pip install -e .`
        

## 一站式处理（All-in-one）

若需执行完整的推理流程（包括预处理和使用已训练模型进行协调），可直接运行脚本`./all_in_one.py`，操作方式有两种：

1. 处理单张MR图像（需指定输入和输出文件路径）：  
    `python all_in_one.py --in-mri &lt;输入MR图像路径&gt; --out_mri &lt;输出MR图像路径&gt; [可选参数]`
    
2. 处理多张MR图像（输入为CSV文件，该文件至少需包含两列：in_paths（输入文件路径）和out_paths（输出文件路径））：  
    `python all_in_one.py --in-csv &lt;输入CSV文件路径&gt; [可选参数]`
    

其中，第二种方式处理多张图像时速度更快。此外，还可指定以下补充参数：

- `--hd-bet-cpu`：若没有GPU或不想使用GPU进行HD-BET脑提取，必须指定此参数。需注意，CPU版本速度更慢，且功能比GPU版本更简单（详情可参考官方说明）。
    
- `--n-procs &lt;数量&gt;`：若通过`--in-csv`参数处理多个输入图像，可通过此参数指定用于并行执行预处理（HD-BET除外）的CPU核心数，默认值为1。
    

若环境中无可用GPU，IGUANe推理将自动在CPU上执行。

需注意，预处理步骤包含图像裁剪操作：将图像尺寸从(182, 218, 182)裁剪为(160, 192, 160)。具体做法为：先计算图像体数据6个侧面的背景切片数量，再执行裁剪以确保不删除任何脑体素。若某一轴向上的背景切片数量不足，则取更小的中心区域进行裁剪（第1、2、3轴对应的裁剪后尺寸分别为176、208、176）。

## 预处理（Preprocessing）

也可自行按以下步骤对MR图像进行预处理：

1. 使用`fslreorient2std`工具将MR图像调整为标准MNI152坐标系 orientation。
    
2. 使用HD-BET（commit版本：ae16068）执行脑提取（去颅骨）。
    
3. 使用N4偏置场校正（N4BiasFieldCorrection）工具进行偏置校正，校正过程需使用步骤2中计算得到的脑掩码。
    
4. 使用FSL-FLIRT工具将图像线性配准到MNI152模板（路径：`./preprocessing/MNI152_T1_1mm.nii.gz`），配准过程采用三线性插值（trilinear interpolation），并设置为6个自由度（six degrees of freedom）。
    
5. 将脑区强度的中位数归一化至500。可使用脚本`./preprocessing/median_norm.py`实现此操作，该操作需用到每张MR图像对应的脑掩码。获取脑掩码的方法为：将步骤4中计算得到的配准变换应用于步骤2中得到的脑掩码（采用最近邻插值（nearestneighbour interpolation））。
    
6. 将图像从(182, 218, 182)尺寸裁剪为(160, 192, 160)。可使用脚本`./preprocessing/crop_mris.py`实现此操作。需注意，该脚本不处理某一轴向上背景切片数量不足的图像。
    

## 协调推理（Harmonization inference）

若需将IGUANe协调与预处理步骤分开执行，可使用脚本`./harmonization/inference.py`，需定义以下三个变量：

- `mri_paths`：预处理后MR图像的文件路径列表。
    
- `dest_paths`：协调后MR图像的输出文件路径列表。
    
- `weights_path`：协调模型权重文件（.h5格式）的路径。可直接使用`./iguane_weights.h5`（即我们研究中训练的模型权重），也可使用自定义模型的权重。
    

该脚本在GPU环境下运行速度更快，但也支持仅使用CPU运行。

使用该脚本时，需先进入`./harmonization`目录。

## 协调训练（Harmonization training）

若需训练自定义的协调模型，可使用脚本`./harmonization/training/main.py`，需定义以下变量：

- `dataset_pairs`：对应每个源域的无限迭代器列表。每个迭代器会生成一个批次的数据，包含来自参考域和源域的图像。实现该迭代器时，可使用`./harmonization/training/input_pipeline/tf_dataset.py`中定义的两个函数之一。这两个函数均基于TFRecord格式文件（文档参考链接见原文）工作，且TFRecord文件中的每个条目需从包含“mri”键的字典编码生成，其中“mri”键对应的值为MR图像矩阵。**重要提示**：需对图像强度进行缩放/偏移处理，使脑区强度的中位数为0、背景强度为-1（预处理后执行`mri = mri/500 - 1`即可实现）。
    
    - `datasets_from_tfrecords`：创建无偏采样的数据集对。
        
    - `datasets_from_tfrecords_biasSampling`：创建含偏采样的数据集对，采样策略与我们论文中描述的一致。我们实验中用于计算采样概率的函数位于`./harmonization/training/input_pipeline/bias_sampling_age.py`。
        
- `DEST_DIR_PATH`：用于保存模型权重和训练统计数据的目录路径。
    
- `N_EPOCHS`（训练轮数）、`STEPS_PER_EPOCH`（每轮训练的步数）
    
- `eval_model`：验证函数，无输入参数，返回需最大化的评分。该函数每`EVAL_FREQ`轮执行一次，最优模型的权重会保存至`&lt;DEST_DIR_PATH&gt;/best_genUniv.h5`。也可修改`EVAL_FREQ`（验证频率）的值。
    

可在`./harmonization/training/input_pipeline/constants.py`中修改图像尺寸。运行该脚本需有可用GPU。

使用该脚本时，需先进入`./harmonization/training`目录。

## 预测模型（Prediction models）

本仓库提供了我们用于年龄预测模型和二分类器的代码：

### 训练脚本

`./prediction/training/main.py`：用于模型训练，需定义以下变量：

- `dataset`：TensorFlow数据集，生成包含MR图像批次及对应目标值的数据。可使用`./prediction/training/input_pipeline/tf_dataset.py`中定义的函数创建该数据集。
    
- `loss`：损失函数（例如，回归任务使用“mae”（平均绝对误差），分类任务使用`tensorflow.keras.losses.BinaryCrossentropy(from_logits=True)`（二元交叉熵））。
    
- `DEST_DIR_PATH`：用于保存模型权重和训练统计数据的目录路径。
    
- `N_EPOCHS`（训练轮数）、`STEPS_PER_EPOCH`（每轮训练的步数）
    

### 推理脚本

`./prediction/inference.py`：用于模型推理，需定义以下变量：

- `mri_paths`：MR图像文件路径列表。
    
- `ids_dict`：字典，包含用于在输出CSV文件中标识图像的字段（例如，`{'sub_id':[1,2], 'session':['sesA','sesB']}`，其中sub_id为被试ID，session为扫描场次）。
    
- `intensity_norm`：推理前应用于每张图像的强度归一化函数（例如，`def intensity_norm(mri): return mri/500 - 1`）。
    
- `activation`：模型的最后一层激活函数（例如，二分类器使用sigmoid函数，回归模型使用恒等函数（identity））。
    
- `model_weights`：预测模型权重文件（.h5格式）的路径。
    
- `csv_dest`：输出CSV文件的路径。
    
- `IMAGE_SHAPE`：图像尺寸，需与`mri_paths`中图像的尺寸（含通道维度）一致。
    

训练相关说明：可在`./prediction/training/input_pipeline/constants.py`中修改图像尺寸。运行训练脚本需先进入`./prediction/training`目录，且需有可用GPU。

推理相关说明：运行推理脚本需先进入`./prediction`目录。

## 元数据（Metadata）

我们在研究中使用的各数据集的元数据均保存在`./metadata/`目录下。

aws s3 cp s3://hcp-openaccess/HCP_1200/100206/MNINonLinear/T1w.nii.gz ./T1w/100206_T1w.nii.gz


aws s3 ls  --no-sign-request s3://fcp-indi/data/Projects/INDI/SALD/RawData_tar/  
2018-03-29 23:11:28 2366575388 sub-031274_sub-031323.tar.gz
2018-03-29 23:11:28 2352637152 sub-031324_sub-031373.tar.gz
2018-03-29 23:11:28 2338653180 sub-031374_sub-031423.tar.gz
2018-03-29 23:11:28 2376733911 sub-031424_sub-031473.tar.gz
2018-03-29 23:11:28 2363159772 sub-031474-sub-031523.tar.gz
2018-03-29 23:29:54 2364442636 sub-031524_sub-031573.tar.gz
2018-03-29 23:30:06 2388069645 sub-031574_sub-031623.tar.gz
2018-03-29 23:30:11 2408585032 sub-031624_sub-031673.tar.gz
2018-03-29 23:30:18 2390685632 sub-031674_sub-031723.tar.gz
2018-03-29 23:30:21 2125151183 sub-031724_sub-031767.tar.gz


aws s3 cp --no-sign-request  s3://fcp-indi/data/Projects/INDI/SALD/RawData_tar/sub-031324_sub-031373.tar.gz  ./