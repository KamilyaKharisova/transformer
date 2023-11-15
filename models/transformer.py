import math
from math import sqrt

import numpy as np
from torch import nn
import torch

class SDPA(nn.Module):
    def __init__(self, cfg):
        super(SDPA, self).__init__()
        self.cfg = cfg
        self.dk = cfg.dk

        # TODO: инициализация Pytorch softmax
        self.softmax = ...

    def forward(self, Q, K, V):
        """
            Вычисляет SDPA.
            Формула: SDPA(Q, K, V) = softmax((QK^T) / sqrt(dk))V
            QK^T - матричное умножение query и key, K^T - транспонирование key.
            Масштабирующий множитель sqrt(dk) нормализует скалярные произведения.

        Args:
            Q (torch.Tensor): Тензор queries. Размерность  [batch_size, l, dk],
                              где seq_len - длина последовательности queries, dk - размерность векторов запросов.
            K (torch.Tensor): Тензор keys. Размерность  [batch_size, n, dk].
            V (torch.Tensor): Тензор values. Размерность  [batch_size, n, dv],
                              где dv - размерность векторов values.

        Returns:
            torch.Tensor: Тензор, представляющий взвешенное суммирование values, взвешенное
                          коэффициентами внимания, полученными после применения механизма SDPA к Q, K и V.
                          Размерность выходного тензора обычно [batch_size, l, dv].

        """

        # 1. Расчет скалярных произведений query (q) и key (k),
        #    деление каждого на sqrt(dk) для масштабирования.
        #    dk - размерность векторов key и query.
        #    Получаем необработанные оценки внимания.
        # TODO: написать код для получения необработанных оценок внимания

        # 2. Применение функции softmax к необработанным оценкам внимания для получения коэффициентов внимания.
        #    Шаг softmax гарантирует, что коэффициенты положительны и в сумме дают 1.
        # TODO: написать код с применением softmax к необработанным оценкам внимания

        # 3. Умножение коэффициентов внимания на матрицу values (V) и суммирование для получения итогового результата.
        #    Оператор @ здесь представляет собой пакетное матричное умножение коэффициентов внимания
        #    на тензор значений.
        #  TODO: написать код перемножения коэффициентов внимания на матрицу values
        ...

if __name__ == "__main__":
    from config.transformer_cfg import cfg
    q = torch.randn((1,5,cfg.dk))
    k = torch.randn((1,10,cfg.dk))
    v = torch.randn((1,10,20))

    sdpa = SDPA(cfg)
    output = sdpa(q,k,v)

