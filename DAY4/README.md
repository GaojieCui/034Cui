每个图像样本对应一行文本标签，格式如下：

复制
编辑
img001.jpg 0
img002.jpg 1
img003.jpg 2
图像文件应放在指定文件夹中，路径在 .txt 文件中只需写文件名。

🚀 训练方式
运行以下命令启动训练：

bash
复制
编辑
python vit_train.py
默认参数：

输入图像大小：(3, 1, 256)，在训练中压缩为 (3, 256)

Patch Size: 16

Hidden Dim: 1024

Transformer Depth: 6

Head 数量: 8

训练轮次: 10

Batch Size: 64

Optimizer: Adam

Learning Rate: 1e-4

训练结束后模型将保存在 model_save/ 目录中。

📈 可视化日志（TensorBoard）
训练中日志会写入 logs_vit_rewrite/ 目录，可使用 TensorBoard 查看：

bash
复制
编辑
tensorboard --logdir=logs_vit_rewrite
在浏览器中打开 http://localhost:6006 查看训练 loss 和 accuracy 曲线。

🧠 模型结构概览
基于原始 Vision Transformer（ViT）结构

支持任意一维序列图像（本例中为压缩图像通道为 1）

分类层采用 cls_token + Transformer + MLP Head

📌 注意事项
图像 resize 到 (1, 256) 后会被展平为一维向量序列输入 ViT

需确保 train.txt 和 val.txt 标签与图像路径匹配

支持自动检测分类数（最大标签值 + 1）

