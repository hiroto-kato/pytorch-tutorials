"""
Deep Learning with PyTorch A 60 Minute Blitz
Tutorial
"""

import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import ssl
import matplotlib.pyplot as plt
import numpy as np

from model import net, conv_net


def tensor():
    """tensorの使い方"""
    data = [[1, 2], [3, 4]]

    # 直でTensorに
    x_data = torch.tensor(data)
    print("Directly from data:\n {}\n".format(x_data))

    # numpyからTensorに変換
    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)
    print("Numpy to tensor: {}, {}\n".format(np_array, x_np))

    # 変換元のTensorの(shape, datatype)をそのままに、他のTensorに変換
    x_ones = torch.ones_like(x_data)
    print("Ones Tensor: \n {}\n".format(x_ones))
    x_rand = torch.rand_like(x_data, dtype=torch.float)
    print("Rnadom Tensor: \n {}\n".format(x_rand))

    # shapeを決めて、ランダムにTensorを生成する
    shape = (
        2,
        3,
    )
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)
    print("Rondom Tensor: \n {}\n".format(rand_tensor))
    print("Ones Tensor: \n {}\n".format(ones_tensor))
    print("Zeros Tensor: \n {}\n".format(zeros_tensor))

    # Tensorの属性
    tensor = torch.rand(3, 4)
    print("Tensor: \n {}".format(tensor))
    print("Shepe of tensor: {}".format(tensor.shape))
    print("Datatype of tensor: {}".format(tensor.dtype))
    print("Device tensor is stored on: {}\n".format(tensor.device))

    # Tensorの操作(いっぱいあるので公式サイト参照)
    # GPUにTensorを移動
    if torch.cuda.is_available():
        tensor = tensor.to("cuda")
        print(tensor)
        print("Device tensor is stored on: {}".format(tensor.device))

    # slicing, indexing
    tensor = torch.ones(4, 4)
    tensor[:, 1] = 0
    print("Slicing and indexing:\n {}".format(tensor))

    # tensorの結合
    t1 = torch.cat([tensor, tensor, tensor], dim=1)
    print(t1)

    # tensorの掛け算
    print("tensor.mul(tensor):\n {}\n".format(tensor.mul(tensor)))
    print("tensor * tensor:\n {}\n".format(tensor * tensor))

    # tensroの行列掛け算
    print("tensor.matmul(tensor):\n {}\n".format(tensor.matmul(tensor)))
    print("tensor @ tensor:\n {}\n".format(tensor @ tensor))

    # in-place
    print(tensor, "\n")
    tensor.add_(5)
    print(tensor)

    # TensorをNumpyに
    t = torch.ones(5)
    print("t: {}".format(t))
    n = t.numpy()
    print("n: {}".format(n))

    # メモリは共有されてるのでtensorを変えるとt,n両方変わる
    t.add_(1)
    print("t: {}".format(t))
    print("n: {}".format(n))

    # NumpyをTensorに
    n = np.ones(5)
    t = torch.from_numpy(n)
    print("n: {}".format(n))
    print("t: {}".format(t))

    # Numpyのほうを変えても変わる
    np.add(n, 1, out=n)
    print("n: {}".format(n))
    print("t: {}".format(t))


def autograd():
    """自動微分の使い方"""
    # PyTorchでの使い方
    model = torchvision.models.resnet18(pretrained=True)
    data = torch.rand(1, 3, 64, 64)
    labels = torch.rand(1, 1000)

    prediction = model(data)  # forward pass
    loss = (prediction - labels).sum()
    loss.backward()  # backward pass

    # 最適化アルゴリズムにSGDを使って、学習率0.01、momentum0.09
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer.step()  # 勾配を計算

    # 以下自動微分で勾配を計算する方法
    a = torch.tensor([2.0, 3.0], requires_grad=True)
    b = torch.tensor([6.0, 4.0], requires_grad=True)
    print("a = {}, b = {}".format(a, b))

    # Q = 3a^3 - b^2
    Q = 3 * a ** 3 - b ** 2
    print("Q = 3a^3 - b^2: {}".format(Q))

    external_grad = torch.tensor([1.0, 1.0])
    Q.backward(gradient=external_grad)

    print("dQ/da = {}".format(a.grad))
    print("dQ/db = {}".format(b.grad))

    # DAGの自動微分の対象外にする
    x = torch.rand(5, 5)
    y = torch.rand(5, 5)
    z = torch.rand((5, 5), requires_grad=True)
    a = x + y
    print("Does `a` require gradients? : {}".format(a.requires_grad))
    b = x + z
    print("Does `b` require gradients? : {}".format(b.requires_grad))

    # 自動微分をしないときは大体finetuningをするとき。finetuningの例を以下でやる
    # すべてのパラメーターをfreezeさせる
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(512, 10)
    # classifierのみ最適化する
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    print(optimizer)


def neural_networks():
    """ニューラルネットワークの使い方"""
    model = net.Net()
    print("Neural Network Model: \n{}".format(model))
    params = list(model.parameters())
    print(len(params))
    print(params[0].size())  # conv1の重み

    input = torch.randn(1, 1, 32, 32)
    out = model(input)
    print(out)
    # model.zero_grad()
    # out.backward(torch.randn(1, 10))

    # 損失関数
    target = torch.randn(10)  # 適当な正解データ
    target = target.view(1, -1)  # 正解データをoutputに合わせる
    criterion = nn.MSELoss()
    loss = criterion(out, target)
    print("loss: {}".format(loss))
    print(loss.grad_fn)  # MSELoss
    print(loss.grad_fn.next_functions[0][0])  # Linear
    print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

    # 誤差逆伝搬
    model.zero_grad()  # 勾配を初期化
    print("conv1.bias.grad before backward")
    print(model.conv1.bias.grad)
    loss.backward()
    print("conv1.bias.grad after backward")
    print(model.conv1.bias.grad)

    # 重みの更新
    learning_rate = 0.01
    for f in model.parameters():
        f.data.sub_(f.grad.data * learning_rate)

    # optimizerを使ったやり方
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # optimizerを生成
    # trainingのループの中で以下の処理を行う
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()


def classifier():
    """クラス分類モデルの学習"""
    # 証明書の期限切れでダウンロードできないため
    ssl._create_default_https_context = ssl._create_unverified_context

    # CIFAR10からデータセットのダウンロード
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    batch_size = 4
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = trainloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    # クラス名
    classes = [
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    # 適当に画像データを取得
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # 画像の表示
    # imshow(torchvision.utils.make_grid(images))
    # labels
    print(" ".join("%5s" % classes[labels[j]] for j in range(batch_size)))

    # 畳み込みニューラルネットワーク
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = conv_net.ConvNet()
    model.to(device)
    # 損失関数とoptimizerの定義
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # モデルの学習
    print("Size: {}".format(len(trainloader)))
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # inputs,labelを取得する
            inputs, labels = data[0].to(device), data[1].to(device)
            # 勾配の初期化
            optimizer.zero_grad()

            # 順伝搬
            outputs = model(inputs)
            # 損失計算
            loss = criterion(outputs, labels)
            # 逆伝搬
            loss.backward()
            optimizer.step()

            # 表示
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(
                    "[{:d}, {:5d}] loss: {:.3f}".format(
                        epoch + 1, i + 1, running_loss / 2000
                    )
                )
                running_loss = 0.0
    print("Finished Training")
    # 学習したモデルを保存
    path = "./cifar_net.pth"
    torch.save(model.state_dict(), path)

    # テストデータで検証
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    print(
        "GroundTruth: ", " ".join("%5s" % classes[labels[j]] for j in range(batch_size))
    )
    model = conv_net.ConvNet()
    model.load_state_dict(torch.load(path))
    outputs = model(images)
    print(outputs)
    _, predicted = torch.max(outputs, 1)
    print(predicted)
    print("Predicted: ", " ".join("%5s" % classes[predicted[j]] for j in range(4)))

    # 学習したモデルの性能を評価
    correct = 0
    total = 0
    # 学習しないので、勾配は計算しない
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # 正解と予測があっている数を足していく
            correct += (predicted == labels).sum().item()
    print(
        "Accuracy of the network on the 10000 test images: {:f} %".format(
            100 * correct / total
        )
    )

    # カウントするクラスを準備
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # それぞれのクラスの予測率を計算
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))


def imshow(img):
    """画像の表示"""
    img = img / 2 + 0.5  # unnormalize
    upimg = img.numpy()
    plt.imshow(np.transpose(upimg, (1, 2, 0)))
    plt.show()


def main():
    # tensor()
    # autograd()
    # neural_networks()
    classifier()


if __name__ == "__main__":
    main()
