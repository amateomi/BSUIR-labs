# Тема: POSIX-совместимая файловая система

Структура ФС, содержимое inode, команды оболочки <br>
Знакомство с POSIX-совместимой файловой системой ‒ opendir(3), readdir(3),
closedir(3), fstat(2), readlink(2), symlink(2), link(2), unlink(2), ...

**Задание:**

Разработать программу dirwalk, сканирующую файловую систему и выводящую в stdout
информацию в соответствии с опциями программы. Формат вывода аналогичен формату
вывода утилиты find.

dirwalk [dir] [options] <br>
dir ‒ начальный каталог. Если опущен, текущий (./). <br>
options ‒ опции. <br>
-l — только символические ссылки <br>
-d — только каталоги <br>
-f — только файлы <br>
-s — сортировать вывод

Если опции ldf опущены, выводятся каталоги, файлы, ссылки, как у find без
параметров.
