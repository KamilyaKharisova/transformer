
# Трансформер
## Задача 1. Реализация Модуля SDPA (Scaled Dot-Product Attention)


<p align="center">
  <img src="../../Downloads/equation-4.png" alt="Equation 4" style="vertical-align: middle;"/>
  <img src="../../../../var/folders/kh/w4d9sx415577q2rxs1jsrv700000gn/T/TemporaryItems/NSIRD_screencaptureui_ltlDZH/Снимок экрана 2023-11-16 в 01.25.51.png" alt="Screenshot" width="300" style="vertical-align: middle;"/>
<br>
  <em> "Attention Is All You Need" by Ashish Vaswani et al., published in 2017</em>
</p>

#### Задача:
Необходимо реализовать SDPA в Pytorch, используя формулу, представленную выше. Эта задача включает в себя следующие пункты:
1.  Инициализация Softmax:
     * Реализовать инициализацию softmax в конструкторе класса SDPA (model/transformer.py).
     * Используйте `nn.Softmax` модуль с правильным измерением для применения softmax.

2. Расчет Attention Scores:
     * Написать код для расчета скалярного произведения `Q` и `K^T` (транспонированного `K`).
     * Масштабировать результат с помощью <img src="../../Downloads/equation-5.png"/>
     * Применить softmax к полученным значениям для получения коэффициентов внимания.
3. Расчет Выходного Тензора:
     * Реализовать умножение коэффициентов внимания на матрицу `V`.
