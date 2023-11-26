import math
from math import sqrt

import numpy as np
from torch import nn
import torch

class SDPA(nn.Module):
    def __init__(self, cfg):
        super(SDPA, self).__init__()
        self.cfg = cfg
        self.dk = cfg.dmodel // cfg.h

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

class SHA(nn.Module):
    def __init__(self, cfg):
        super(SHA, self).__init__()
        self.cfg = cfg
        self.dk = cfg.dmodel // cfg.h

        # TODO: Инициализация линейных преобразований для Q, K, V
        self.weights_q = ...
        self.weights_k = ...
        self.weights_v = ...

        # Инициализация механизма SDPA
        self.spda = SDPA(self.cfg)

    def forward(self, Q, K, V):
        """
            Вычисляет SHA.
            Формула: SHA(Q, K, V) = SDPA(Q', K', V')
            Q', K', V' - линейно преобразованные тензоры Q, K, V.

        Args:
            Q (torch.Tensor): Тензор queries.
            K (torch.Tensor): Тензор keys.
            V (torch.Tensor): Тензор values.

        Returns:
            torch.Tensor: Взвешенное суммирование values, взвешенное коэффициентами внимания.

        """

        # TODO: Линейные преобразования Q, K, V

        # TODO: Вызов SDPA с преобразованными Q, K, V

class MHA(nn.Module):
    def __init__(self, cfg):
        super(MHA, self).__init__()
        self.cfg = cfg

        # Инициализация списка SHA модулей
        self.sha_list = nn.ModuleList([SHA(cfg) for _ in range(cfg.h)])

        # TODO: Инициализация линейного преобразования для объединения выходов из всех головок внимания
        self.weights_o = ...

    def forward(self, Q, K, V):
        """
            Вычисляет MHA.
            Формула: MHA(q, k, v) = Concat(SHA1, SHA2, ..., SHAh)W^O
            где SHAi - выход i-го Single Head Attention, W^O - линейное преобразование.

        Args:
            Q (torch.Tensor): Тензор queries.
            K (torch.Tensor): Тензор keys.
            V (torch.Tensor): Тензор values.

        Returns:
            torch.Tensor: Результат Multi-Head Attention.

        """

        # TODO: Вычисление выходов для каждого SHA

        # TODO: Конкатенация выходов и применение линейного преобразования

        ...




if __name__ == "__main__":
    from config.transformer_cfg import cfg
    q = torch.randn((1,5,cfg.dmodel // cfg.h))
    k = torch.randn((1,10,cfg.dmodel // cfg.h))
    v = torch.randn((1,10,20))

    sdpa = SDPA(cfg)
    output = sdpa(q,k,v)

    q = torch.randn((1, 5, cfg.dmodel))

    mha = MHA(cfg)
    output = mha(q, q, q)

