import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


# ====================== 像样的小UNet ======================
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin_t1 = nn.Linear(1, 28 * 28)
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, t, x):
        B = x.shape[0]
        t_emb = self.lin_t1(t).view(B, 1, 28, 28)
        h = F.silu(self.conv1(x + t_emb))
        h = F.silu(self.conv2(h))
        h = F.silu(self.conv3(h))
        return self.conv4(h)


# ====================== 数据 ======================
ds = MNIST("./mnist", train=True, download=True, transform=ToTensor())
dl = DataLoader(ds, batch_size=128, shuffle=True)


# ====================== 训练 ======================
def train():
    model = Net()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(12):
        total = 0
        for img, _ in dl:
            B = img.shape[0]
            t = torch.rand(B, 1)
            eps = torch.randn_like(img)
            x = t.view(B, 1, 1, 1) * img + (1 - t).view(B, 1, 1, 1) * eps

            v_pred = model(t, x)
            v_target = img - eps
            loss = F.mse_loss(v_pred, v_target)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * B

        print(f"Epoch {epoch + 1:2d} | Loss {total / len(ds):.4f}")
    return model


# ====================== 生成 ======================
@torch.no_grad()
def generate(model, n=64):
    x = torch.randn(n, 1, 28, 28)
    steps = 40
    dt = 1 / steps
    for i in range(steps):
        t = torch.ones(n, 1) * i * dt
        x = x + model(t, x) * dt
    return x


# ====================== 画图 ======================
def show(imgs):
    imgs = imgs.clamp(0, 1).numpy()
    plt.figure(figsize=(9, 9))
    for i in range(64):
        plt.subplot(8, 8, i + 1)
        plt.imshow(imgs[i, 0], cmap="gray")
        plt.axis("off")
    plt.show()


# ====================== 运行 ======================
if __name__ == "__main__":
    model = train()
    imgs = generate(model)
    show(imgs)
