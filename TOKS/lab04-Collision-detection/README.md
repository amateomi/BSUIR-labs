# Тема: Случайные методы доступа к моноканалу

**Задание:**

Написать программу пакетной передачи данных упрощенным алгоритмом CSMA/CD в
соответствии с требованиями.

Требования к наполнению программы:
1. На стороне передатчика, реализовать три ключевых шага алгоритма:
   прослушивание канала, обнаружение коллизии и розыгрыш случайной задержки
   (в соответствующей последовательности).
2. Предусмотреть возможность эмуляции коллизии. Вероятность коллизии должна
   составлять 50 %.
3. Для расчета случайной задержки использовать стандартную формулу.
4. Из дополнения к алгоритму, реализовать поддержку jam-сигнала (дополнительно и
   правильно; как на стороне передатчика, так и на стороне приемника).
