from typing import Callable
import torch

from ..networks import MPS, MPS_Node, random_mps


class _LocalNeuralNet(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(dim, dim),
            torch.nn.LeakyReLU(),
        )

        def init_normal(m):
            if type(m) == torch.nn.Linear:
                torch.nn.init.kaiming_uniform_(m.weight, a=1.0, mode="fan_in")

        #         #torch.nn.init.xavier_uniform_(m.weight,gain=math.sqrt(2.1))

        self.linear_relu_stack.apply(init_normal)

    def forward(self, x):
        x = torch.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class MPS_Model:
    def __init__(self, nofq, bond_dim, mps: MPS = None, use_previt=True):
        if mps is None:
            self.mps = random_mps(nofq, bond_dim, 2)
        else:
            self.mps = mps.copy()
        self.nets = []
        self.params = []
        self.use_previt = use_previt
        for i in self.mps:
            dim = torch.flatten(i.tensor).shape[0]
            model = _LocalNeuralNet(dim)
            self.nets.append(model)
            self.params = self.params + list(model.parameters())

    def forward(self):
        mps = []
        for j in range(len(self.nets)):
            if self.use_previt:
                x = self.mps[j].tensor
            else:
                x = torch.ones_like(self.mps[j].tensor)
            mps.append(
                MPS_Node(self.nets[j].forward(x).reshape(tuple(self.mps[j].shape)))
            )
        return MPS(mps)

    def parameters(self):
        return self.params

    def train(self, lr_list, epoch_list, stopping_loss, loss_func: Callable = None):
        if loss_func is None:
            def loss_func(mps):
                return mps.norm

        for lr, epochs in zip(lr_list, epoch_list):
            print("LR: " + str(lr))
            optimizer = torch.optim.NAdam(
                self.parameters(), lr=lr, momentum_decay=0, betas=(0.9, 0.99)
            )
            for i in range(epochs):
                optimizer.zero_grad()
                mps = self.forward()
                loss = loss_func(mps)
                loss.backward()
                optimizer.step()

                if i % 1 == 0:
                    print("Epoch: " + str(i))
                    print("Train Loss: " + str(loss.item()))
                if loss.item() <= stopping_loss:
                    break

            if loss.item() <= stopping_loss:
                print("Early Stopping: Desired Loss reached")
                break
