import torch
import torch.nn as nn
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

class SparseRC(nn.Module):
    def __init__(self, X, lambda1, lambda2, constraint='notears', omega=0.3):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.X = X.clone().detach()
        self.d = self.X.shape[1]
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.constraint = constraint
        self.omega = omega
        self.fc = torch.nn.Linear(self.d, self.d, bias=False) # input x : output (A, ) A^Tx + b

    def postprocess_A(self):
        A = self.fc.weight.T
        A_est = torch.where(torch.abs(A) > self.omega, A, torch.tensor(0, dtype=torch.float32))
        return A_est.detach().cpu().numpy()
    
    def l1_reg(self):
        A = self.fc.weight
        return torch.sum(torch.abs(A)) # 
    
    def acyclicity(self):
        A = self.fc.weight
        if self.constraint == 'notears':
            return torch.trace(torch.matrix_exp(A * A)) - self.d
        elif self.constraint == 'dag-gnn':
            M = torch.eye(self.d) + A * A / self.d  # (Yu et al. 2019)
            return  torch.trace(torch.linalg.matrix_power(M, self.d)) - self.d
        elif self.constraint == 'frobenius':
            return torch.sum((A * A.T) ** 2)
        
    def forward(self, X):
        return self.fc(X) # output is XA
        


def sparserc_solver(X, lambda1, lambda2, epochs=3000, constraint="notears", omega=0.3):
    '''
        sparserc solver
        params:
        X: data (np.array) of size n x d
        lambda1: coefficient (double) for l1 regularization λ||Α||_1
        lambda2: coefficient (double) for the graph constraint 
        epochs: upper bound for the number of iterations of the optimization solver.
    '''
    X = torch.tensor(X, device=device, dtype=dtype)    
    N = X.shape[0]

    model = SparseRC(X, lambda1=lambda1, lambda2=lambda2, constraint=constraint, omega=omega)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    early_stop = 40
    best_loss = 100000000000000000

    for i in range(epochs):

        def closure():
            # nonlocal so that we can alter their value
            nonlocal best_loss, early_stop

            # zero gradients and compute output = XA
            optimizer.zero_grad()
            output = model(X)

            # compute total optimization loss and back-propagate
            loss = (1 / (2 * N)) * torch.norm((X - output), p=1)   # (1/2n) * |X-XA|_1
            loss = loss + lambda1 * model.l1_reg() + lambda2 * model.acyclicity() # (1/2n) * |X-XA|_1  + λ1 * |A| + λ2 *  h(A)
            loss.backward()

            # overview of performance
            if i % 10 == 0:
                print("Epoch: {}. Loss = {:.3f}".format(i, loss.item()))
            
            # early stopping 
            if loss.item() >= best_loss:
                early_stop -= 1
            else:
                early_stop = 40
                best_loss = loss.item()
                torch.save(model.state_dict(), 'results/best_model.pl')

            return loss
    
        optimizer.step(closure)

        if early_stop == 0:
            break

    # threshold values with absolute values < omega
    model = SparseRC(X, lambda1=lambda1, lambda2=lambda2, constraint=constraint, omega=omega)
    model.load_state_dict(torch.load('results/best_model.pl'))
    A = model.postprocess_A()
    
    return A


class SparseRC_weightfinder(nn.Module):
    def __init__(self, X, A):
        super().__init__()
        self.X = X.clone().detach()
        self.d = self.X.shape[1]
        self.mask = A.clone().detach()
        self.fc = torch.nn.Linear(self.d, self.d, bias=False) # input x : output (A, ) A^Tx + b

    def postprocess_A(self):
        A = self.fc.weight.T @ self.mask
        return A.detach().cpu().numpy()
        
    def forward(self, X):
        return X @ (self.fc.weight.T * self.mask) # output is X (A o M)
        

def sparserc_solver_weight_finder(X, A, epochs=3000):
    '''
        sparserc solver
        params:
        X: data (np.array) of size n x d
        lambda1: coefficient (double) for l1 regularization λ||Α||_1
        lambda2: coefficient (double) for the graph constraint 
        epochs: upper bound for the number of iterations of the optimization solver.
    '''
    X = torch.tensor(X, device=device, dtype=torch.float32)
    A = torch.tensor(A, device=device, dtype=torch.float32)
    N = X.shape[0]

    model = SparseRC_weightfinder(X, A)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    early_stop = 40
    best_loss = 10000000000000000

    for i in range(epochs):

        def closure():
            # nonlocal so that we can alter their value
            nonlocal best_loss, early_stop

            # zero gradients and compute output = XA
            optimizer.zero_grad()
            output = model(X)

            # compute total optimization loss and back-propagate
            loss = ( 1 / (2 * N) ) * torch.norm((X - output), p=1)   # (1/2n) * |X-XA|_1
            loss.backward()

            # overview of performance
            # if i % 10 == 0:
            #     print("Epoch: {}. Loss = {:.3f}".format(i, loss.item()))
            
            # early stopping 
            if loss.item() >= best_loss:
                early_stop -= 1
            else:
                early_stop = 40
                best_loss = loss.item()
                torch.save(model.state_dict(), 'results/best_model.pl')

            return loss
    
        optimizer.step(closure)

        if early_stop == 0:
            break

    # threshold values with absolute values < omega
    model = SparseRC_weightfinder(X, A)
    model.load_state_dict(torch.load('results/best_model.pl'))
    A = model.postprocess_A()
    
    return A
