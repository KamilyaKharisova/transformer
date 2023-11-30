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

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super(FeedForward, self).__init__()
        self.cfg = cfg

        # Первый линейный слой увеличивает размерность данных с dmodel до 4*dmodel.
        self.w1 = ...
        # Второй линейный слой уменьшает размерность обратно с 4*dmodel до dmodel.
        self.w2 = ...

        # Функция активации ReLU используется между двумя линейными слоями.
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Формула: FF(x) = ReLU(xW1 + b1)W2 + b2
        где:
        - W1, b1 - веса и смещение первого линейного слоя,
        - W2, b2 - веса и смещение второго линейного слоя,

        Args:
            x (torch.Tensor): Входной тензор с размерностью [batch_size, seq_len, dmodel].

        Returns:
            torch.Tensor: Выходной тензор с той же размерностью, что и входной.
        """
        ...

class PositionEncoder(nn.Module):
    def __init__(self, cfg):
        super(PositionEncoder, self).__init__()
        self.cfg = cfg

        # Создание матрицы позиционного кодирования
        # Размер матрицы: [cfg.max_sentence_len, cfg.dmodel]
        self.pe_matrix = torch.empty((cfg.max_sentence_len, cfg.dmodel))

        # Формула для позиционного кодирования:
        # PE(pos, 2i) = sin(pos / (10000 ^ (2i / dmodel)))
        # PE(pos, 2i+1) = cos(pos / (10000 ^ (2i / dmodel)))
        # где pos - позиция в предложении, i - индекс в векторе
        # ...
        # Полезно знать. Пусть a - numpy array. Тогда a[0::2] выдает элементы на четных позициях, а a[1::2] на нечетных.

        ...

    def forward(self, x):
        """
       Прямой проход PositionEncoder. Добавляет positional encoding к входному тензору.

       Positional encoding вектор вычисляется как:
       PE(pos, 2i) = sin(pos / (10000 ^ (2i / dmodel)))
       PE(pos, 2i+1) = cos(pos / (10000 ^ (2i / dmodel)))
       где pos - позиция в предложении, i - индекс в векторе.

       Args:
           x (torch.Tensor): Входной тензор с размерностью [batch_size, seq_len, dmodel].

       Returns:
           torch.Tensor: Тензор с добавленным позиционным кодированием.
       """
        # Вычисление размера предложения из входного тензора
        # ...

        # Добавление позиционного кодирования к входному тензору
        # ...
        ...

class EncoderSingleLayer(nn.Module):
    def __init__(self, cfg):
        super(EncoderSingleLayer, self).__init__()
        # Инициализация Multi-Head Attention (MHA)
        self.mha = MHA(cfg)
        # Инициализация нормализации
        self.ln1 = nn.LayerNorm(cfg.dmodel)
        self.ln2 = nn.LayerNorm(cfg.dmodel)
        # Инициализация полносвязного Feed Forward слоя
        self.ff = FeedForward(cfg)

    def forward(self, x):
        """
        Прямой проход одного слоя энкодера.

        Этапы:
        1. Применение Multi-Head Attention.
        2. Добавление исходного входа к результату (Residual Connection).
        3. Применение Layer Normalization.
        4. Применение Feed Forward слоя.
        5. Добавление результата после MHA к результату FF (Residual Connection).
        6. Применение Layer Normalization.

        Args:
            x (torch.Tensor): Входной тензор с размерностью [batch_size, seq_len, dmodel].

        Returns:
            torch.Tensor: Тензор после одного слоя энкодера.
        """
        # Применение MHA, добавление Residual Connection и Layer Normalization
        # ...

        # Применение Feed Forward, добавление Residual Connection и Layer Normalization
        # ...

class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        # Создание N слоев энкодера cfg.N
        # ...
        self.seq = ...
        self.cfg = cfg

    def forward(self, x):
        """
        Прямой проход через энкодер.

        Последовательно применяет N слоев энкодера к входным данным.

        Args:
            x (torch.Tensor): Входной тензор с размерностью [batch_size, seq_len, dmodel].

        Returns:
            torch.Tensor: Тензор после прохождения через N слоев энкодера.
        """
        # Применение каждого слоя энкодера
        # ...


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

