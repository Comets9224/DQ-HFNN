#neural network for different task
#image, time series and text
import torch
import torch.nn as nn

"""
作用: 定义了一个非常简单的全连接网络 classical_part_layer。
结构: Input -> Linear(128) -> Sigmoid -> Linear(output) -> Sigmoid。
意图:
通用特征提取器: 这是一个非常通用的、与特定数据（如图像）无关的特征提取器。它可以处理任何扁平化的向量输入。
早期原型/组件: 在 QAHFNN.py 的早期原型 MyModel 中，self.classical_layer = classical_part_layer(...) 被注释掉了。这表明作者最初可能想用这样一个简单的全连接层作为经典分支，但后来为每个数据集都换上了更强大的、定制化的CNN或DenseNet模型。
是否还在使用: 基本可以确定已被弃用，因为所有最终的模型都使用了更复杂的、针对特定任务的经典层。

"""
#简单的全连接网络提取特征
#input= [batch, input_shape]
#output = [batch, output_shape]
class classical_part_layer(nn.Module):
    """ 全连接层"""
    def __init__(self, input_feature_num, output_feature_num):
        super(classical_part_layer, self).__init__()
        self.dense_layer_1 = nn.Linear(input_feature_num, 128)
        self.dense_layer_2 = nn.Linear(128, output_feature_num)

    def forward(self, x):

        output1 =  torch.sigmoid(self.dense_layer_1(x))
        output2 = torch.sigmoid(self.dense_layer_2(output1))

        return output2

