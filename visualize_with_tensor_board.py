import matplotlib.pyplot as plt
import numpy as np
import ssl
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import tensorboard as tb

from model.fashion_net import FashionNet

# パラメーター
batch_size = 4
# クラス名
classes = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot",
)


# ヘルパー関数
def select_n_random(data, labels, n=100):
    """ランダムなデータとそれと一致するラベルをn個選択"""
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]


def matplotlib_imshow(img, one_channel=False):
    """画像の表示"""
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def images_to_probs(model, images):
    """学習済みのネットワークと画像の一致度と予測を生成する"""
    output = model(images)
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(model, images, labels):
    """学習済みのネットワークを使って画像をmatplotlibで生成"""
    preds, probs = images_to_probs(model, images)
    # 予測値と正解値をつけてバッチごとに画像を描画
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title(
            "{0}, {1:.1f}%\n(label: {2})".format(
                classes[preds[idx]],
                probs[idx] * 100.0,
                classes[labels[idx]],
                color=("green" if preds[idx] == labels[idx].item() else "red"),
            )
        )

    return fig


def add_pr_curve_tensorboard(
    class_index, test_probs, test_label, writer, global_step=0
):
    """precision-recall curveを描画"""
    tensorboard_truth = test_label == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(
        classes[class_index],
        tensorboard_truth,
        tensorboard_probs,
        global_step=global_step,
    )


def main():
    # 証明書の期限切れでダウンロードできないため
    ssl._create_default_https_context = ssl._create_unverified_context
    # tensorboardでエラーが出るため
    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

    # transforms
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # データセット
    trainset = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    testset = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # データローダー
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testloader = trainloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    model = FashionNet()

    # 損失関数と最適化アルゴリズムを定義
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # TensorBoardのsetup
    writer = SummaryWriter("runs/fashion_mnist_experiment_1")

    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    img_grid = torchvision.utils.make_grid(images)
    matplotlib_imshow(img_grid, one_channel=True)
    # tensorboardに書き込む
    writer.add_image("four_fashion_mnist_images", img_grid)
    writer.add_graph(model, images)

    # 画像をランダムに選択
    images, labels = select_n_random(trainset.data, trainset.targets)
    # クラスラベルを取得
    class_labels = [classes[lab] for lab in labels]
    # image mebeddings
    features = images.view(-1, 28 * 28)
    writer.add_embedding(features, metadata=class_labels, label_img=images.unsqueeze(1))

    # 学習
    running_loss = 0.0
    for epoch in range(1, 11):
        x = 0
        for i, data in enumerate(trainloader, 0):
            # inputsとラベルをそれぞれ取得
            inputs, labels = data

            # 勾配を初期化
            optimizer.zero_grad()

            # 順伝搬、逆伝搬、最適化
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            x += 1

        # 損失をログ
        writer.add_scalar("training loss", running_loss / len(trainloader), epoch)

        # モデルの予測を表示する図をログに追加
        writer.add_figure(
            "predictions vs. actuals",
            plot_classes_preds(model, inputs, labels),
            global_step=epoch,
        )
        running_loss = 0.0

    print("Finished Training")

    # 学習済みモデルの評価
    class_probs = []
    class_label = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            output = model(images)
            class_probs_batch = [F.softmax(el, dim=0) for el in output]

            class_probs.append(class_probs_batch)
            class_label.append(labels)

    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_label = torch.cat(class_label)

    # pr curvesの描画
    for i in range(len(classes)):
        add_pr_curve_tensorboard(i, test_probs, test_label, writer)

    writer.close()


if __name__ == "__main__":
    main()
