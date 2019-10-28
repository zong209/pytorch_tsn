#! .pyenv/bin/python
# -*- encoding:utf-8 -*-
'''
@File     : test.py
@Time     : 2019/09/30 17:18:37
@Author   : Gaozong/260243
@Contact  : 260243@gree.com.cn/zong209@163.com
@Describe : test
'''
import numpy as np
import torch
from model import TSN
from image_util import GroupScale, GroupCenterCrop, Stack, ToTorchFormatTensor, GroupNormalize
from PIL import Image
from torchvision import transforms

settings = {
    "PT": 'models/1571224414_zlfy_30_32_0.70857.pt',
    "segements_num": 3,
    "class_num": 4
}

print("* Loading model...")
net_pt = settings["PT"]

global net
net = TSN(settings["class_num"], settings["segements_num"])
net.eval()
checkpoint = torch.load(net_pt)
net.load_state_dict(checkpoint['model_state_dict'])

print("* Model loaded")


def model_eval(images):

    composed = transforms.Compose([
        GroupScale(256),
        GroupCenterCrop(224),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(mean=[.485, .456, .406], std=[.229, .224, .225])
    ])
    images = composed(images)

    preds = net.forward(images)

    print(preds)


import cv2
import os

idxs = [10, 35, 60]
root_dir = "data/clips_1516"
images0 = []
images1 = []
for ind in idxs:
    image_ind = "{:0>6d}.jpg".format(ind)
    # print(image_ind)
    img0 = cv2.imread(os.path.join(root_dir, image_ind))
    images0.append(Image.fromarray(img0))
    # img1 = Image.open(os.path.join(root_dir, image_ind))
    # images1.append(img1)

model_eval(images0)
# model_eval(images1)

# ###########################################
# #　测试部分

# img0, label0 = self.valid_transformed_dataset[0]
# img1, label1 = self.valid_transformed_dataset[1]
# inputs = torch.cat([img0, img1], 0).reshape([2, 9, 224, 224])
# label = torch.Tensor([label0, label1]).reshape([2, 1])
# outputs = net.forward(inputs)
# # outputs = F.softmax(outputs)
# # lable_vector = torch.zeros(1, self.output_nums).scatter_(
# #     1, label.long(), 1)

# # 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# outputs = outputs.reshape([-1, self.output_nums])
# loss = criterion(outputs, label.long().flatten())
# print(loss)
# loss.backward()
# optimizer.step()
