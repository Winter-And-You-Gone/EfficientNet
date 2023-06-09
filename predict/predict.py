# -*- coding: utf-8 -*-
import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model.model import efficientnet_b0 as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 定义预测时的预处理方法,如下所示:------------------------->>>
    # 此部分需要参赛队伍添加，和测试的预处理方法保持一致。
    img_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}
    num_model = "B0"

    data_transform = transforms.Compose(
        [transforms.Resize(img_size[num_model]),
         transforms.CenterCrop(img_size[num_model]),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path = r"./test_img/Apple_scab/1.JPG"  # 测试图片路径
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(
        img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = '../train/class_indices.json'  # 对应图像标签(json格式)路径
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(
        json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = create_model(num_classes=10).to(device)  # 创建模型
    # load model weights
    model_weight_path = "../train/save_weight/model-b0-134.pth"  # 训练保存的权重路径
    # model.load_state_dict(torch.load(model_weight_path, map_location=device))

    weights_dict = torch.load(model_weight_path, map_location=device)
    model_dict = model.state_dict()
    model.load_state_dict(
        {k: v for k, v in weights_dict.items() if k in model_dict})
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(
        class_indict[str(predict_cla)],
        predict[predict_cla].numpy())
    plt.title(print_res)
    print(class_indict[str(predict_cla)], predict[predict_cla].item())

    data = open("result_img/predict.txt",
                mode="a")  # 保存预测结果文件路径（txt文件），需保留9位小数
    data.write("image: {}   class: {}   prob: {:.9}".format(
        img_path,
        class_indict[
            str(predict_cla)],
        predict[
            predict_cla].numpy()) + "\n")
    data.close()

    plt.savefig("test_img", dpi=300)  # 预测图片保存路径
    plt.show()


if __name__ == '__main__':
    main()
