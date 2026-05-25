import numpy as np
import pandas as pd
from pysr import PySSRegressor

model = PySRRegressor(niterations=200, maxsize=30, populations=20,
                      binary_operators=["+", "-", "*", "/", "^"],
                      unary_operators=["exp", "log", "sigmoid(x)=1/(1+exp(-x))"],
                      constraints = {"^": (-1,3),},
                      elementwise_loss = "loss(prediction,target) = (prediction-target)^2",
                      verbosity = 1,
                      random_state = 42,
                      deterministic=True)

model.fit(X, y)

print(model.equations_)

print(model.get_best())
model.sympy()


