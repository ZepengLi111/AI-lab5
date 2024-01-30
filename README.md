## Setup

项目依赖如下

- torch==2.1.0+cu121

- torchvision==0.16.0+cu121

- tqdm==4.64.1

- transformers==4.34.0

可以使用以下指令安装

```
pip install -r requirements.txt
```

## Repository structure

```
|-- data				# 数据
	|-- data
		|-- 1.txt
		|-- 1.jpg
		...
	|-- test_without_label.txt
	|-- train.txt
|-- model 				# 模型
	|-- encoder.py		# 图像编码器和文本编码器
	|-- VL_model.py 	# 模态融合模型
|-- utils 
    |-- dataloader.py	# 构建数据集
    |-- predict.py		# 预测标签
    |-- train.py		# 训练与评估
|-- main.py				# 主程序入口
```

## How to run

在运行之前，请确保在`./src`目录下放置了data文件夹，命名方式见上面的仓库架构。

然后请在`./src`目录下运行下面的指令

```
python main.py --lr 1e-4 --batch_size 64 --model '1' --ablation 'img' --epoch 10
```

其中各个参数的可选值及解释如下：

- model
  - '1'：表示resnet + bert + transformer模型
  - '2'：表示vit + bert + transformer模型
  - '3'：表示resnet + bert + cross attention模型
  - '4'：表示vit + bert + cross attention模型

- ablation
  - 'img'：表示只保留图像的消融实验
  - 'text'：表示只保留文本的消融实验
  - None（直接删去这个参数即可）：表示都保留

## Hardware requirements

如果使用GPU运行，请保证显存达16G以上。推荐使用kaggle运行。