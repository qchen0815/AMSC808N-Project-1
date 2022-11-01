import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm


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

    @abstractmethod
    def __str__(self) -> str:
        ...


class SGD(Optimizer):
    """
    The step_size argument can either be a `float` or a schedule
    function that takes a single parameter for iteration k and
    returns the step size at that iteration.
    """

    def __init__(self, data, batch_size, step_size, epochs, loss, grad):
        super().__init__(data, batch_size, step_size, epochs, loss, grad)
        if type(step_size) == float:
            self.step_size = lambda k: step_size

    def minimize(self):
        self.loss_hist = []
        self.grad_norms = []

        w = np.random.rand(self.data.shape[1])
        k = 0

        for _ in tqdm(range(self.epochs), desc='SGD'):
            for data_batch in self._gen_batches():
                gradient = self.grad(w, data_batch)
                w = w - self.step_size(k) * gradient
                k += 1

                self.grad_norms.append(np.linalg.norm(gradient))

            self.loss_hist.append(self.loss(w, self.data))

        return w

    def __str__(self):
        return 'SGD'


class Nesterov(Optimizer):

    def __init__(self, data, batch_size, step_size, epochs, loss, grad):
        super().__init__(data, batch_size, step_size, epochs, loss, grad)

    def minimize(self):
        self.loss_hist = []
        self.grad_norms = []

        y = np.random.rand(self.data.shape[1])
        x = np.zeros(self.data.shape[1])
        k = 0

        for _ in tqdm(range(self.epochs), desc='Nesterov'):
            for data_batch in self._gen_batches():
                gradient = self.grad(y, data_batch)
                x_step = y - self.step_size * gradient
                mu = 1 - 3 / (5 + k)
                y = (1 + mu) * x_step - mu * x
                x = x_step
                k += 1

                self.grad_norms.append(np.linalg.norm(gradient))

            self.loss_hist.append(self.loss(y, self.data))

        return y

    def __str__(self):
        return 'Nesterov'


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

        for _ in tqdm(range(self.epochs), desc='Adam'):
            for data_batch in self._gen_batches():
                t += 1
                gradient = self.grad(w, data_batch)
                m = self.beta1 * m + (1 - self.beta1) * gradient
                v = self.beta2 * v + (1 - self.beta2) * gradient ** 2
                mb = m / (1 - self.beta1 ** t)
                vb = v / (1 - self.beta2 ** t)
                w = w - self.step_size * mb / (np.sqrt(vb) + self.e)

                self.grad_norms.append(np.linalg.norm(gradient))

            self.loss_hist.append(self.loss(w, self.data))

        return w
    
    def __str__(self):
        return 'Adam'


class LBFGS(Optimizer):
    """
    The step_size argument can either be a `float` or `None`. `None` will
    use the L-BFGS line search function to compute step size at each iteration,
    a `float` value will use a fixed step size.
    """

    def __init__(self, data, batch_size, step_size, epochs, loss, grad,
                 m=5, update_freq=10, eta=0.5, gam=0.9):
        super().__init__(data, batch_size, step_size, epochs, loss, grad)
        self.m = m
        self.update_freq = update_freq
        self.eta = eta
        self.gam = gam
        self.jmax = int(np.ceil(np.log(1e-2) / np.log(gam)))

    def _linesearch(self, x, p, g) -> tuple:
        """
        L-BFGS line search algorithm from lecture notes.
        """
        a = 1
        f0 = self.loss(x, self.data)
        aux = self.eta * np.inner(p, g)
        for j in range(self.jmax):
            xtry = x + a * p
            f1 = self.loss(xtry, self.data)
            if f1 < f0 + a * aux:
                break
            else:
                a *= self.gam
        return a, j

    def _finddirection(self, g, s, y, rho) -> np.ndarray:
        """
        Function to compute step direction in stochastic L-BFGS
        taken from lecture notes.
        """
        m = len(s)
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

        x = np.random.rand(self.data.shape[1])
        s = np.zeros((self.m, self.data.shape[1]))
        y = np.zeros((self.m, self.data.shape[1]))
        rho = np.zeros(self.m)
        updates = 1

        # first initial step
        init_batches = np.random.choice(len(self.data), 2 * self.batch_size, replace=False)
        g = self.grad(x, self.data[init_batches[:self.batch_size]])
        a = self.step_size if self.step_size else self._linesearch(x, -g, g)[0]
        xnew = x - a * g
        gnew = self.grad(xnew, self.data[init_batches[self.batch_size:]])
        s[0] = xnew - x
        y[0] = gnew - g
        rho[0] = 1 / np.inner(s[0], y[0])
        x = xnew
        g = gnew
        iter = 1

        for _ in tqdm(range(self.epochs), desc='L-BFGS'):
            for data_batch in self._gen_batches():
                if updates < self.m:
                    p = self._finddirection(g, s[:updates], y[:updates], rho[:updates])
                else:
                    p = self._finddirection(g, s, y, rho)
                if self.step_size:
                    a = self.step_size
                else:
                    a, j = self._linesearch(x, p, g)
                    if j == self.jmax:
                        p = -g
                        a, j = self._linesearch(x, p, g)
                step = a * p
                xnew = x + step
                gnew = self.grad(xnew, data_batch)
                if iter % self.update_freq == 0:
                    s = np.roll(s, 1, axis=0)
                    y = np.roll(y, 1, axis=0)
                    rho = np.roll(rho, 1)
                    s[0] = step
                    y[0] = gnew - g
                    rho[0] = 1 / np.inner(step, y[0])
                    updates += 1
                x = xnew
                g = gnew
                iter += 1

                self.grad_norms.append(np.linalg.norm(g))

            self.loss_hist.append(self.loss(x, self.data))

        return x

    def __str__(self):
        return 'LBFGS'
