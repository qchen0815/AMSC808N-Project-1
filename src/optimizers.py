import numpy as np
from abc import ABC, abstractmethod


class Optimizer(ABC):
    """
    Abstract base class for optimizers.
    """

    def __init__(self, data, batch_size, step_size, epochs, loss, grad) -> None:
        self.data = data
        self.batch_size = batch_size
        self.step_size = step_size
        self.epochs = epochs
        self.loss = loss
        self.grad = grad
        self.loss_hist = []
        self.grad_norms = []

    def _gen_batches(self) -> list:
        """
        Randomly splits data into batches of size batch_size.
        """
        remainder = len(self.data) % self.batch_size
        permutation = np.random.permutation(len(self.data))
        batches = np.array_split(permutation[:-remainder], len(self.data) // self.batch_size) \
            + ([permutation[-remainder]] if remainder else [])
        data_batches = [self.data[batch].reshape((-1, self.data.shape[1])) for batch in batches]
        return data_batches

    def get_histories(self) -> tuple:
        """
        Returns loss and gradient norm histories.
        """
        return self.loss_hist, self.grad_norms

    @abstractmethod
    def minimize(self) -> np.ndarray:
        """
        Runs the minimization algorithm using the parameters passed to the optimizer.
        """
        ...


class SGD(Optimizer):

    def __init__(self, data, batch_size, step_size, epochs, loss, grad):
        super().__init__(data, batch_size, step_size, epochs, loss, grad)

    def minimize(self):
        self.loss_hist = []
        self.grad_norms = []

        w = np.random.rand(self.data.shape[1])

        for _ in range(self.epochs):
            for data_batch in self._gen_batches():
                gradient = self.grad(w, data_batch)
                w = w - self.step_size * gradient

                self.grad_norms.append(np.linalg.norm(gradient))

            self.loss_hist.append(self.loss(w))

        return w


class Nesterov(Optimizer):

    def __init__(self, data, batch_size, step_size, epochs, loss, grad):
        super().__init__(data, batch_size, step_size, epochs, loss, grad)

    def minimize(self):
        self.loss_hist = []
        self.grad_norms = []

        y = np.random.rand(self.data.shape[1])
        x = np.zeros(self.data.shape[1])
        k = 0

        for _ in range(self.epochs):
            for data_batch in self._gen_batches():
                gradient = self.grad(y, data_batch)
                x_step = y - self.step_size * gradient
                mu = 1 - 3 / (5 + k)
                y = (1 + mu) * x_step - mu * x
                x = x_step
                k += 1

                self.grad_norms.append(np.linalg.norm(gradient))

            self.loss_hist.append(self.loss(y))

        return y


class Adam(Optimizer):

    def __init__(self, data, batch_size, step_size, epochs, loss, grad,
                 beta1=0.9, beta2=0.999, e=1e-7):
        super().__init__(data, batch_size, step_size, epochs, loss, grad)
        self.beta1 = beta1
        self.beta2 = beta2
        self.e = e

    def minimize(self):
        self.loss_hist = []
        self.grad_norms = []

        w = np.random.rand(self.data.shape[1])
        m = np.zeros(self.data.shape[1])
        v = np.zeros(self.data.shape[1])
        t = 0

        for _ in range(self.epochs):
            for data_batch in self._gen_batches():
                t += 1
                gradient = self.grad(w, data_batch)
                m = self.beta1 * m + (1 - self.beta1) * gradient
                v = self.beta2 * v + (1 - self.beta2) * gradient ** 2
                mb = m / (1 - self.beta1 ** t)
                vb = v / (1 - self.beta2 ** t)
                w = w - self.step_size * mb / (np.sqrt(vb) + self.e)

                self.grad_norms.append(np.linalg.norm(gradient))

            self.loss_hist.append(self.loss(w))

        return w


class LBFGS(Optimizer):

    def __init__(self, data, batch_size, step_size, epochs, loss, grad,
                 m=5, update_freq=10):
        super().__init__(data, batch_size, step_size, epochs, loss, grad)
        self.m = m
        self.update_freq = update_freq

    def _linesearch(self, w, p, g, eta=0.5, gam=0.9) -> float:
        """
        L-BFGS line search algorithm.
        """
        jmax = int(np.ceil(np.log(1e-14) / np.log(gam)))
        a = 1
        f0 = self.loss(w)
        aux = eta * np.inner(p, g)
        for _ in range(jmax):
            wtry = w + a * p
            f1 = self.loss(wtry)
            if f1 < f0 + a * aux:
                break
            else:
                a *= gam
        return a

    def _finddirection(self, g, s, y) -> np.ndarray:
        """
        Function to compute step direction in stochastic L-BFGS.
        """
        m = len(s)
        rho = 1 / np.sum(s * y, axis=1)
        a = np.zeros(m)
        for i in range(m):
            a[i] = rho[i] * np.inner(s[i], g)
            g -= a[i] * y[i]
        gam = np.inner(s[0], y[0]) / np.inner(y[0], y[0])
        g *= gam
        for i in range(m - 1, -1, -1):
            aux = rho[i] * np.inner(y[i], g)
            g += (a[i] - aux) * s[i]
        p = -g
        return p

    def minimize(self):
        self.loss_hist = []
        self.grad_norms = []

        w = np.random.rand(self.data.shape[1])
        s = np.zeros((self.m, self.data.shape[1]))
        y = np.zeros((self.m, self.data.shape[1]))
        step_count = 0
        updates = 1

        # first initial step
        batches = np.random.choice(len(self.data), 2 * self.batch_size, replace=False)
        gradient = self.grad(w, self.data[batches[:self.batch_size]])
        a = self._linesearch(w, -gradient, gradient)
        wnew = w - a * gradient
        gnew = self.grad(wnew, self.data[batches[self.batch_size:]])
        s[0] = wnew - w
        y[0] = gnew - gradient
        w = wnew
        gradient = gnew

        for _ in range(self.epochs):
            for data_batch in self._gen_batches():
                gnew = self.grad(w, data_batch)
                p = self._finddirection(gnew, s[:min(updates, self.m)], y[:min(updates, self.m)])
                w -= self.step_size * p
                step_count += 1
                if step_count % self.update_freq == 0:
                    s = np.roll(s, 1, axis=0)
                    y = np.roll(y, 1, axis=0)
                    s[0] = self.step_size * p
                    y[0] = gnew - gradient
                    updates += 1
                gradient = gnew

                self.grad_norms.append(np.linalg.norm(gradient))

            self.loss_hist.append(self.loss(w))

        return w
