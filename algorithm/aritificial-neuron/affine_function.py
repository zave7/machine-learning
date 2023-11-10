import numpy as np

class Affine:
    
    def __init__(self, w, b):
        self.w = np.array(w)
        self.b = np.array(b)
        
    def __call__(self, x) -> float:
        return np.dot(x, self.w) + self.b

if __name__ == '__main__':
    affine1 = Affine(w=[1, 1], b=-1.5)

    print(f"affine1 0 0 : {affine1([0, 0])}")
    print(f"affine1 0 1 : {affine1([0, 1])}")
    print(f"affine1 1 0 : {affine1([1, 0])}")
    print(f"affine1 1 1 : {affine1([1, 1])}")

    affine2 = Affine(w=[-1, -1], b=0.5)

    print(f"affine2 0 0 : {affine1([0, 0])}")
    print(f"affine2 0 1 : {affine1([0, 1])}")
    print(f"affine2 1 0 : {affine1([1, 0])}")
    print(f"affine2 1 1 : {affine1([1, 1])}")