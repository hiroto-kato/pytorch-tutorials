"""
Introduction to PyTorch
Quickstart
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt


# パラメーター
batch_size = 64  # バッチサイズ
epochs = 5  # エポック数
alpha = 0.001  # 学習率


class NeuralNetwork(nn.Module):
    """modelの定義"""

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        """順伝搬"""
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x


def train(dataloader, device, model, loss_fn, optimizer):
    """学習"""
    size = len(dataloader.dataset)

    # ミニバッチ学習
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 損失誤差を計算
        pred = model(X)
        loss = loss_fn(pred, y)

        # 勾配を計算
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 学習状況を100回毎に表示
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print("loss: {:>7f}  [{:>5d}/{:>5d}]".format(loss, current, size))


def test(dataloader, model, device, loss_fn):
    """モデルの性能をテスト"""
    size = len(dataloader.dataset)
    # 評価モード
    model.eval()
    test_loss, correct = 0, 0

    # 平均ロスと正確さを計算
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(
        "Test Error: \n Accuracy: {:>0.1f}%, Avg loss: {:>8f} \n".format(
            100 * correct, test_loss
        )
    )


def main():
    # FashionMNISTは(画像, クラスid)のリストでつまってる。tmp[0][0]で画像, tmp[0][1]でクラスid
    # 訓練データをdatasetsからダウンロード
    training_data = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )
    # テストデータをdatasetsからダウンロード
    test_data = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=ToTensor()
    )

    # データローダーの作成
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # GPUが使用可能ならGPUを設定、なければCPUを使用する
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # model作成
    model = NeuralNetwork().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=alpha)

    # エポック分学習を行う
    for t in range(epochs):
        print("Epoch {}\n-------------------------------".format(t + 1))
        train(train_dataloader, device, model, loss_fn, optimizer)
        test(test_dataloader, model, device, loss_fn)
    print("Done.")

    # モデルの保存
    torch.save(model.state_dict(), "model.pth")
    print("Saved Pytorch Model State to model.pth")

    # 以下で学習したモデルを使ってテストする
    # モデルを読み込む
    model = NeuralNetwork()
    model.load_state_dict(torch.load("model.pth"))

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    # 予測する
    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print("Predicted: {}, Actual: {}".format(predicted, actual))


if __name__ == "__main__":
    main()
