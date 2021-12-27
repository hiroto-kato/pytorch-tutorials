import numpy as np
import math

import torch
import matplotlib.pyplot as plt


def numpy_pred_version():
    """numpyでsin(x)の近似関数を予測して3次多項式で表す"""
    print("Prediction by numpy.")
    # input(x),output(y)を作成
    x = np.linspace(-math.pi, math.pi, 2000)
    y = np.sin(x)

    # 重みをランダムに初期化
    a = np.random.randn()
    b = np.random.randn()
    c = np.random.randn()
    d = np.random.randn()

    # 計算とyを予測
    learning_rate = 1e-6
    for t in range(2001):
        # y = a + bx + cx^2 + dx^3
        y_pred = a + b * x + c * x ** 2 + d * x ** 3
        loss = np.square(y_pred - y).sum()
        if t % 500 == 0:
            print(t, loss)

        # 逆伝搬と勾配の計算
        grad_y_pred = 2.0 * (y_pred - y)
        grad_a = grad_y_pred.sum()
        grad_b = (grad_y_pred * x).sum()
        grad_c = (grad_y_pred * x ** 2).sum()
        grad_d = (grad_y_pred * x ** 3).sum()

        # 重みを更新
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
        c -= learning_rate * grad_c
        d -= learning_rate * grad_d

    print("Result: y = {} + {}x + {}x^2 + {}x^3".format(a, b, c, d))


def tensor_pred_version():
    """tensorでsin(x)の近似関数を予測して3次多項式で表す"""
    print("Prediction by tensor.")
    dtype = torch.float
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # input(x),output(y)を作成
    x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
    y = torch.sin(x)

    # 重みをランダムに初期化
    a = torch.randn((), device=device, dtype=dtype)
    b = torch.randn((), device=device, dtype=dtype)
    c = torch.randn((), device=device, dtype=dtype)
    d = torch.randn((), device=device, dtype=dtype)

    learning_rate = 1e-6
    for t in range(2001):
        # y = a + bx + cx^2 + dx^3
        y_pred = a + b * x + c * x ** 2 + d * x ** 3
        loss = (y_pred - y).pow(2).sum().item()
        if t % 500 == 0:
            print(t, loss)

        # 逆伝搬と勾配の計算
        grad_y_pred = 2.0 * (y_pred - y)
        grad_a = grad_y_pred.sum()
        grad_b = (grad_y_pred * x).sum()
        grad_c = (grad_y_pred * x ** 2).sum()
        grad_d = (grad_y_pred * x ** 3).sum()

        # 重みを更新
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
        c -= learning_rate * grad_c
        d -= learning_rate * grad_d

    print(
        "Result: y = {} + {}x + {}x^2 + {}x^3".format(
            a.item(), b.item(), c.item(), d.item()
        )
    )

    # plt.plot(range(2001), y_pred.to("cpu").detach().numpy())
    # plt.plot(range(2001), y.to("cpu").detach().numpy())


def torch_autograd_version():
    """tensorでsin(x)の近似関数を予測して3次多項式で表す(自動微分を仕様)"""
    print("Prediction by autograd.")
    dtype = torch.float
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # input(x),output(y)を作成
    # 逆伝搬しないのでrequired_grad=False
    x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
    y = torch.sin(x)

    # 重みをランダムに初期化
    #
    a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

    learning_rate = 1e-6
    for t in range(2001):
        # y = a + bx + cx^2 + dx^3
        y_pred = a + b * x + c * x ** 2 + d * x ** 3
        loss = (y_pred - y).pow(2).sum()
        if t % 500 == 0:
            print(t, loss.item())

        # 逆伝搬と勾配の計算
        loss.backward()

        # 重みを確率的勾配法で更新
        # autogradの対象外
        with torch.no_grad():
            a -= learning_rate * a.grad
            b -= learning_rate * b.grad
            c -= learning_rate * c.grad
            d -= learning_rate * d.grad

            # 勾配を初期化
            a.grad = None
            b.grad = None
            c.grad = None
            d.grad = None

    print(
        "Result: y = {} + {}x + {}x^2 + {}x^3".format(
            a.item(), b.item(), c.item(), d.item()
        )
    )


def main():
    numpy_pred_version()
    tensor_pred_version()
    torch_autograd_version()


if __name__ == "__main__":
    main()
