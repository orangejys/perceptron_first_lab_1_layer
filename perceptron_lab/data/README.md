# Локальная копия Iris

iris.csv содержит 150 исходных измерений и названия трёх видов.
Копия экспортирована из sklearn.datasets.load_iris() в scikit-learn 1.8.0.
CSV включается в пакет: сеть и scikit-learn при запуске проекта не нужны.

- Первоисточник: Fisher, R. A. (1936), The use of multiple measurements
  in taxonomic problems.
- Набор Iris: https://doi.org/10.24432/C56C76
- UCI: https://archive.ics.uci.edu/dataset/53/iris
- Загрузчик scikit-learn:
  https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
- Исходная копия scikit-learn:
  https://github.com/scikit-learn/scikit-learn/blob/1.8.0/sklearn/datasets/data/iris.csv

Версия scikit-learn содержит исправления двух измерений относительно старой
копии UCI. Происхождение указано как scikit-learn, а не как байтовая копия
старого файла UCI.

Четыре признака в сантиметрах: длина/ширина чашелистика, длина/ширина лепестка.
Программа выбирает 100 объектов setosa/versicolor и два признака лепестка.
Метки: setosa=0, versicolor=1. Virginica не участвует, так как задача бинарная.
Оценка делается на отложенной части этих 100 объектов.

Набор Iris в UCI распространяется по CC BY 4.0:
https://creativecommons.org/licenses/by/4.0/
