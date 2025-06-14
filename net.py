import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchinfo import summary


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, relu: bool = True
    ):
        super().__init__()
        self._kernel_size = kernel_size
        # we only support the kernel sizes of 1 and 3
        assert kernel_size in (1, 3)

        self.conv: nn.Conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            bias=False,
        )
        self.bn: nn.BatchNorm2d = nn.BatchNorm2d(out_channels, affine=False)
        self.beta = nn.Parameter(torch.zeros(out_channels))  # type: ignore
        self.relu = relu

        # initializations
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        if self._kernel_size == 3:
            x = F.pad(x, (1, 1, 1, 1), "constant", -1)
        x = self.conv(x)
        x = self.bn(x)
        x += self.beta.view(1, self.bn.num_features, 1, 1).expand_as(x)
        return F.relu(x, inplace=True) if self.relu else x


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1: ConvBlock = ConvBlock(in_channels, out_channels, 3)
        self.conv2: ConvBlock = ConvBlock(out_channels, out_channels, 3, relu=False)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        out += identity
        return F.relu(out, inplace=True)


class Network(nn.Module):
    def __init__(
        self,
        board_size: int = 9,
        in_channels: int = 6,
        residual_channels: int = 192,
        residual_layers: int = 15,
    ):
        super().__init__()
        self.conv_input = ConvBlock(in_channels, residual_channels, 3)
        # self.residual_tower = nn.Sequential(
        #     *[
        #         ResBlock(residual_channels, residual_channels)
        #         for _ in range(residual_layers)
        #     ]
        # )
        self.residual_tower: nn.ModuleList = nn.ModuleList(
            [
                ResBlock(residual_channels, residual_channels)
                for _ in range(residual_layers)
            ]
        )
        self.policy_conv = ConvBlock(residual_channels, 2, 1)
        self.policy_fc = nn.Linear(
            2 * board_size * board_size, board_size * board_size * 6
        )
        torch.nn.init.constant_(self.policy_fc.weight.data,0.3)

        self.value_conv = ConvBlock(residual_channels, 1, 1)
        self.value_fc_1 = nn.Linear(board_size * board_size, 256)
        self.value_fc_2 = nn.Linear(256, 1)

    def forward(self, planes):
        # first conv layer
        x = self.conv_input(planes)

        # residual tower
        for block in self.residual_tower:
            x = block(x)

        # x = self.residual_tower(x)

        # policy head
        pol = self.policy_conv(x)
        pol = self.policy_fc(torch.flatten(pol, start_dim=1))
        # print(pol)
        pol = F.log_softmax(pol, dim=1)
        # print(pol)
        # print('----------')

        # value head
        val = self.value_conv(x)
        val = F.relu(self.value_fc_1(torch.flatten(val, start_dim=1)), inplace=True)
        val = torch.tanh(self.value_fc_2(val))

        return pol, val


class PolicyValueNet:

    def __init__(self, model_file=None, use_gpu=True, device="cuda"):
        self.use_gpu = use_gpu
        self.l2_const = 2e-3  # l2 正则化
        self.device = device
        self.policy_value_net = Network().to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.policy_value_net.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.l2_const,
        )
        if model_file:
            self.policy_value_net.load_state_dict(
                torch.load(model_file)
            )  # 加载模型参数

    # 输入一个批次的状态，输出一个批次的动作概率和状态价值
    # def policy_value(self, state_batch):
    #     self.policy_value_net.eval()
    #     state_batch = torch.tensor(state_batch).to(self.device)
    #     log_act_probs, value = self.policy_value_net(state_batch)
    #     log_act_probs, value = log_act_probs.cpu(), value.cpu()
    #     act_probs = np.exp(log_act_probs.detach().numpy())
    #     return act_probs, value.detach().numpy()

    # 输入棋盘，返回每个合法动作的（动作，概率）元组列表，以及棋盘状态的分数
    def policy_value_fn(self, board) -> tuple[np.ndarray, float]:
        self.policy_value_net.eval()
        # 获取合法动作列表
        legal_positions = board.availables()
        # print(legal_positions)
        current_state = torch.as_tensor(board.current_state()).to(self.device)
        # 使用神经网络进行预测
        # try:
        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(current_state)

        log_act_probs, value = log_act_probs.cpu(), value.cpu()
        act_probs = np.exp(log_act_probs.detach().numpy().flatten())
        # print(act_probs)
        # print('++++++++')

        act_probs = act_probs * legal_positions
        # print(act_probs)
        # except:
        #     print (log_act_probs)
        #        act_probs = np.multiply(legal_positions, act_probs)
        # print(act_probs)
        return act_probs, value

    # 保存模型
    def save_model(self, model_file):
        torch.save(self.policy_value_net.state_dict(), model_file)

    # 执行一步训练
    def train_step(self, state_batch, mcts_probs, winner_batch, lr=0.002):
        self.policy_value_net.train()
        # 包装变量
        state_batch = torch.tensor(state_batch).to(self.device)
        mcts_probs = torch.tensor(mcts_probs).to(self.device)
        winner_batch = torch.tensor(winner_batch).to(self.device)
        # 清零梯度
        self.optimizer.zero_grad()
        # 设置学习率
        for params in self.optimizer.param_groups:
            # 遍历Optimizer中的每一组参数，将该组参数的学习率 * 0.9
            params["lr"] = lr
        # 前向运算
        log_act_probs, value = self.policy_value_net(state_batch)
        value = torch.reshape(value, shape=[-1])
        # 价值损失
        value_loss = F.mse_loss(input=value, target=winner_batch)
        # 策略损失
        policy_loss = -torch.mean(
            torch.sum(mcts_probs * log_act_probs, dim=1)
        )  # 希望两个向量方向越一致越好
        # 总的损失，注意l2惩罚已经包含在优化器内部
        loss = value_loss + policy_loss
        # 反向传播及优化
        loss.backward()
        self.optimizer.step()
        # 计算策略的熵，仅用于评估模型
        with torch.no_grad():
            entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, dim=1)
            )
        return loss.detach().cpu().numpy(), entropy.detach().cpu().numpy()


def weight_input(i, layer, lines):
    weight_flat = [float(x) for x in lines[i].split()]
    weight_tensor = torch.tensor(weight_flat, dtype=torch.float32)
    weight_tensor = weight_tensor.view(layer.shape)
    # print(weight_tensor.shape)
    with torch.no_grad():
        layer.copy_(weight_tensor)


if __name__ == "__main__":
    # 15x192
    f = open("leelaz.txt", "r")
    lines = f.read().splitlines()
    in_channels = 6
    model = Network(
        board_size=9, in_channels=in_channels, residual_channels=192, residual_layers=15
    )

    weight_flat = [float(x) for x in lines[1].split()]
    weight_tensor = torch.tensor(weight_flat, dtype=torch.float32)
    weight_size = list(model.conv_input.conv.weight.shape)
    weight_size[1] = 18
    weight_tensor = weight_tensor.view(weight_size)
    # print(weight_tensor.shape)
    new_tensor = weight_tensor[:, [0, 1, 2, 3, 16, 17], :, :]
    print(weight_tensor.shape)
    with torch.no_grad():
        model.conv_input.conv.weight.copy_(new_tensor)

    weight_input(3 - 1, model.conv_input.beta.data, lines)
    weight_input(4 - 1, model.conv_input.bn.running_mean, lines)
    weight_input(5 - 1, model.conv_input.bn.running_var, lines)
    print(model.conv_input.beta.shape)
    print(len(lines[4].split()))
    for i in range(0, 8):
        # First conv block in residual block
        weight_input(
            5 + 1 + 8 * i - 1, model.residual_tower[i].conv1.conv.weight, lines
        )
        weight_input(5 + 2 + 8 * i - 1, model.residual_tower[i].conv1.beta.data, lines)
        weight_input(
            5 + 3 + 8 * i - 1, model.residual_tower[i].conv1.bn.running_mean, lines
        )
        weight_input(
            5 + 4 + 8 * i - 1, model.residual_tower[i].conv1.bn.running_var, lines
        )

        # Second conv block in residual block
        weight_input(
            5 + 5 + 8 * i - 1, model.residual_tower[i].conv2.conv.weight, lines
        )
        weight_input(5 + 6 + 8 * i - 1, model.residual_tower[i].conv2.beta.data, lines)
        print(type(model.residual_tower))

        weight_input(
            5 + 7 + 8 * i - 1, model.residual_tower[i].conv2.bn.running_mean, lines
        )
        weight_input(
            5 + 8 + 8 * i - 1, model.residual_tower[i].conv2.bn.running_var, lines
        )

    # weight_input(132 - 1, model.value_conv.conv.weight, lines)
    # weight_input(133 - 1, model.value_conv.beta.data, lines)

    # weight_input(134 - 1, model.value_conv.bn.running_mean, lines)
    # weight_input(135 - 1, model.value_conv.bn.running_var, lines)

    # weight_input(136 - 1, model.value_fc_1.weight, lines)
    # weight_input(137 - 1, model.value_fc_1.bias, lines)
    print(model.value_fc_1.weight.shape)

    # weight_input(138 - 1, model.value_fc_2.weight, lines)
    # weight_input(139 - 1, model.value_fc_2.bias, lines)

    print(model.value_fc_2.weight.shape)

    torch.save(model.state_dict(), "model.pth")
    summary(model, (1, in_channels, 9, 9))
