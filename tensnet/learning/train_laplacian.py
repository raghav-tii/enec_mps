import torch

from tensnet.linalg.optimization import ground_state_solver
from tensnet.networks.base_mpo import mps_mpo_mul
from ..networks import MPO, MPO_Node, random_mpo
from ..linalg import identity_mpo


class NeuralNet(torch.nn.Module):
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


class L_Model:
    def __init__(self, bits, bond_dim):
        self.mpo1 = random_mpo(bits, bond_dim, 2)
        self.nets = []
        self.params = []
        for i in self.mpo1.nodes:
            dim = torch.flatten(i.tensor).shape[0]
            model = NeuralNet(dim)
            self.nets.append(model)
            self.params = self.params + list(model.parameters())

    def forward(self):
        mpo = []
        for j in range(len(self.nets)):
            x = torch.ones_like(self.mpo1.nodes[j].tensor)
            mpo.append(
                MPO_Node(
                    self.nets[j].forward(x).reshape(tuple(self.mpo1.nodes[j].shape))
                )
            )
        return MPO(mpo)

    def parameters(self):
        return self.params


def loss_fnc(mpo, L):
    N = len(L)
    mpo_I = identity_mpo(N)
    pred = L @ mpo
    t = pred + mpo_I * -1.0
    t1 = t.norm
    t2 = ground_state_solver(mps_mpo_mul(t.mpo_to_mps(), identity_mpo(N, pd=4)))[0]
    # t3 = ground_state_solver(mps_mpo_mul(t.mpo_to_mps(), identity_mpo(N, pd=4)) * -1.0)[0]
    # t2 = t.norm_diag
    # rand_ind = torch.randint(0, 2**7, (200,))
    # t3 = 0
    # val = 0
    # for i in rand_ind:
    #     j = int(i.item() + 2 * torch.randn(size=(1,)).item())
    #     if j < 0 or j >= 2**7 or val == 0:
    #         j = i.item()
    #     t3 = t3 + abs(t.sample(i.item() + 2**N - 2**7, j + 2**N - 2**7))
    #     if j == i.item():
    #         val = val + 1
    # ans = (t1 * 0.65 + t2 * 0.15)/(2**N) + (0.20 * t3/(val * 2**(N/2)))
    return (t1 + t2) / 2**N


def train_model_on_L(model, lr_list, epoch_list, stopping_loss, L):
    for lr, epochs in zip(lr_list, epoch_list):
        print("LR: " + str(lr))
        optimizer = torch.optim.NAdam(
            model.parameters(), lr=lr, momentum_decay=0, betas=(0.9, 0.99)
        )
        for i in range(epochs):
            optimizer.zero_grad()
            mpo = model.forward()
            loss = loss_fnc(mpo, L)
            loss.backward()
            # for j in model.parameters():
            #     print(j.grad)
            optimizer.step()

            if i % 1 == 0:
                print("Epoch: " + str(i))
                print("Train Loss: " + str(loss.item()))
            if loss.item() <= stopping_loss:
                break

        if loss.item() <= stopping_loss:
            print("Early Stopping: Desired Loss reached")
            break


# def loss_fnc2(mpo, x , y):
#     pred = mpo @ x
#     return torch.linalg.norm(pred.contract - y.contract)


# def train_model_on_data(model,lr,epochs,stopping_loss, x, y, L):


#     optimizer = torch.optim.NAdam(model.parameters() , lr=lr)

#     for i in range(epochs):
#         for j in range(len(x)):
#             optimizer.zero_grad()
#             mpo = model.forward()
#             loss = loss_fnc2(mpo, x[j], y[j], L)
#             loss.backward()
#             optimizer.step()


#         if i%2 == 0:
#             print("Epoch: " + str(i))
#             print("Train Loss: " + str(loss.item()))
#         if loss.item() <= stopping_loss:
#             print("Early Stopping: Desired Loss reached")
#             break
