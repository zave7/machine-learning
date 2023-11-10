import numpy as np

class BinaryCrossEntropyLossFunction:

    def __call__(self, y, y_hat):
        a = y * np.log(y_hat)
        b = (1 - y) * np.log(1 - y_hat)
        return - (a + b)
        
bce = BinaryCrossEntropyLossFunction()
print(bce(y=1, y_hat=0.99))

preds = np.arange(start=0.1, stop=1, step=0.1)

print(bce(y=0, y_hat=preds))
print(bce(y=1, y_hat=preds))

print(1 * False)
print(1 * True)