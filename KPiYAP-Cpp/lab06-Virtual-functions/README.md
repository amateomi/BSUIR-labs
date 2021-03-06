# Тема: Виртуальные функции и абстрактные классы

**Цель работы:** Создание консольной программы для реализации абстрактных классов и чисто виртуальных функций, создания
и наследования виртуальных функций, изучения статического и динамического полиморфизма на основе виртуальных функций,
включая виртуальные деструкторы.

**Общие требования к выполнению работы:**

1. По полученному базовому классу создать классы наследников по двум разным ветвям наследования (B←P1←P11 и B←P2):
   - во всех классах должны быть свои данные (характеристики объектов), расположенные в private или protected секциях;
   - во всех классах создать конструкторы инициализации объектов для всех классов (не забыть про передачу параметров в
     конструкции базовых классов);
   - во всех классах создать деструкторы;
   - остальные методы создавать по необходимости.
2. Создать в базовом классе чисто виртуальные функции расчета (например, расчет площади фигуры и т.п.) и вывода объекта
   на экран (всех его параметров). Выполнить реализацию этих виртуальных функций в классах наследниках.
3. Задать в базовом классе деструктор как виртуальный.
4. В головной функции динамически создать массив указателей на базовый класс. Заполнить массив указателями
   на динамически создаваемые объекты производных классов (P1, P11, P2). Для каждого элемента массива проверить работу
   виртуальных функций. Удалить из памяти динамически выделенные объекты.

**Контрольные вопросы:**

1. Виртуальные функции.
2. Абстрактные классы.
3. Наследование виртуальных функций.
4. Виртуальный деструктор.
5. Статический и динамический полиморфизм.

**Задание по варианту:**

Оружие *(преподаватель разрешил выбрать базовый класс на своё усмотрение)*

**Замечание:**

Большую часть полей создать в базовом классе, использовать все меньше и меньше полей в наследниках.

**Дополнительное задание:**

Ввести данные дальнобойного оружия с клавиатуры, спросить у пользователя хочет ли он заменить что-либо в только что 
введенном, если да, то заменить. Ввод проверять на корректность.
