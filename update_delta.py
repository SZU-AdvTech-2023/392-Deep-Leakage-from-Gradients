import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, channel=1, hidden=588, num_classes=10):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on %s" % device)

def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)

net = LeNet(channel=3, hidden=768, num_classes=100).to(device)
net.apply(weights_init)

#ADMM settings
model_prev = copy.deepcopy(net.state_dict())
theta = copy.deepcopy(net.state_dict())
weights = copy.deepcopy(net.state_dict())
alpha = {}
for key in weights.keys():
    alpha[key] = torch.zeros_like(weights[key])

#hyperparameters
rho = 0.1
alpha_prev = copy.deepcopy(alpha)
eta = 0.1
E = 1

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

criterion = cross_entropy_for_onehot

def update(data, onehot_label, retain):
    delta = {}
    optimizer_sgd = torch.optim.SGD(net.parameters(), lr=0.00001)
    for i in range(E):
        net.zero_grad()
        pre = net(data)
        #gt_label = torch.argmax(gt_onehot_label, dim=-1)
        y = criterion(pre, onehot_label)
        y.backward(retain_graph=retain)
        net_weights_pre = copy.deepcopy(net.state_dict())
        for name, param in net.named_parameters():
            if param.requires_grad:
                param.grad = param.grad + (alpha[name] + rho * (net_weights_pre[name] - theta[name]))
        optimizer_sgd.step()

    weights = net.state_dict()
    for key in alpha.keys():
        alpha[key] = alpha[key] + rho * (weights[key] - theta[key])
    # step 7
    for key in alpha.keys():
        delta[key] = (weights[key] - model_prev[key]) + (1 / rho) * (alpha[key] - alpha_prev[key])

    for key in theta.keys():
        theta[key] = delta[key] * 0.001 + theta[key]
    # delta
    origin_delta = list((_.detach().clone() for _ in delta.values()))
    # approximately gradient = delta / -eta
    od_gradient = [grad / -eta for grad in origin_delta]
    return od_gradient



