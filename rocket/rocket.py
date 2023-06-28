from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeClassifier


class Rocket:
    # стандартный инит, задаем параметры модели
    def __init__(self, k: int = 10, weight_type: str = "normal") -> None:
        self.k = k
        self.weight_type = weight_type
        self.all_kernels: List[nn.Conv1d] = []
        self.classifier: RidgeClassifier = None

    # внутренняя функция для генерации ядер, все аргументы внутри берет из класса, передать нужно только длину последовательности
    def create_kernels(self, l_values: int) -> List[nn.Conv1d]:
        all_kernels = []

        for i in range(self.k):
            # choose kernel
            kernel_size = np.random.choice([7, 9, 11])

            # choose dilation
            A = np.log2((l_values - 1) / (kernel_size - 1))
            x = np.random.uniform(low=0, high=A)
            d = int(np.floor(2**x))

            # choose padding
            padding = np.random.choice([0, 1])
            if padding:
                padding = ((kernel_size - 1) * d) // 2

            conv = self._init_convolution(kernel_size, padding, d, self.weight_type)

            all_kernels.append(conv)

        return all_kernels

    # внутренний метод чтобы правильно генерить ядра и не забивать спаггети-кодом основные части
    # статик метод потому что возможно захотим вызывать его не только здесь, он не зависит от класса особо
    @staticmethod
    def _init_convolution(
        kernel_size: int, padding: int, dilation: int, weight_type: str
    ) -> nn.Conv1d:
        # в качестве конволюции будем использовать торчевскую, кажется так проще всего, параметры надо только передать
        # 1 канал это по факту наш временной ряд
        conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
        )

        # разные типы ядер, на нормальном даже получилось built-in методом сделать
        if weight_type == "normal":
            nn.init.normal_(conv.weight, mean=0.0, std=1.0)
            conv.weight - conv.weight.mean()
        elif weight_type == "binary":
            new_weight = torch.randint_like(conv.weight, low=0, high=2)
            new_weight[new_weight == 0] = -1
            conv.weight.data = new_weight
        elif weight_type == "ternary":
            new_weight = torch.randint_like(conv.weight, low=-1, high=2)
            conv.weight.data = new_weight

        nn.init.uniform_(conv.bias, -1, 1)
        # не забываем отключить градиенты
        conv.weight.requires_grad_(False)
        conv.bias.requires_grad_(False)

        return conv

    # вызываем эту функцию тогда когда у нас нет ядер и мы фитимся, здесь произойдет генерация ядер и последующий трансформ
    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        self.all_kernels = self.create_kernels(values.shape[1])

        return self.transform(values)

    # применение ядер просто
    def transform(self, values: np.ndarray) -> np.ndarray:
        total_kernels = self.k
        n, l = values.shape

        values_tensor = torch.tensor(values, dtype=torch.float)
        features = np.zeros((n, 2 * total_kernels))
        # мы заранее знаем сколько у нас будет признаков, так что будем заполнять матрицу по индексам
        for i in range(0, 2 * total_kernels, 2):
            out = self.all_kernels[i // 2](values_tensor.unsqueeze(1)).squeeze(1)
            maxes = torch.max(out, dim=1)[0]
            ppv = torch.mean((out > 0).float(), dim=1)

            features[:, i] = maxes.numpy()
            features[:, i + 1] = ppv.numpy()

        return features

    # метод фит, в котором заведем ядра, обучим классификатор
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        classifier_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        X_train_transformed = self.fit_transform(X.values)

        if not classifier_args:
            classifier_args = {"solver": "saga"}

        self.classifier = RidgeClassifier(**classifier_args)

        self.classifier.fit(X_train_transformed, y)

    # стандартный предикт
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_test_transformed = self.transform(X.values)

        return self.classifier.predict(X_test_transformed)
