import os
import sys
import json
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

from model import vgg

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))


    '''------------------数据准备工作--------------'''

    '''---------数据增强-------'''
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224), # 随机裁剪
            transforms.RandomHorizontalFlip(), # 水平随机反转
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))   # mean 和 std 
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }



    '''----------数据路径---------'''
    data_root = os.path.abspath(os.path.join(os.getcwd(),
                                 "../../pilibala/deep-learning-for-image-processing")) 
    image_path = os.path.join(data_root, "data_set", "flower_data")
    '''
    Python assert（断言）用于判断一个表达式，
    在表达式条件为 false 的时候触发异常。
    断言可以在条件不满足程序运行的情况下直接返回错误，
    而不必等待程序运行后出现崩溃的情况
    '''
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)


    '''--------定义dataset和dataloader-------'''
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                        transform=data_transform['train'])
    train_num = len(train_dataset)
    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str) #写入到json文件中
    
    batch_size = 8  # 减少模型复杂度，但是不改变训练的batch
    # nw = min([os.cpu_count(), batch_size if batch_size >1 else 0, 8]) # number of workers
    nw = 0 # windows中只能为0
    print("using {} dataloader workers every process".format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw)



    '''-----------定义验证集的dataset和dataloader-----------'''
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))


    '''---------定义模型，损失函数，优化器----------'''
    model_name = "vgg16"
    net = vgg(model_name=model_name, num_classes=5, init_weights=True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)


    '''----------开始训练------------'''
    epochs = 10
    best_acc = 0.0
    save_path = './{}Net.pth'.format(model_name)
    train_steps = len(train_loader)
    acc_list = []
    loss_list = []
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        # 进度条
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad() # 每个batch梯度都要清零
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # 在训练集上训练，记录loss
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        loss_list.append(loss)


        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()    

        # 在验证集上，记录accuracy
        val_accurate = acc / val_num
        acc_list.append(acc)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

            

    print("Fnished Training")
    plt.figure(1)
    plt.title("loss")
    plt.plot(loss_list)
    plt.show()

    plt.figure(2)
    plt.title("accuracy")
    plt.plot(acc_list)
    plt.show()

if __name__ == '__main__':
    main()        