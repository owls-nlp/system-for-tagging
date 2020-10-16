# Тестирование PDFPlumber на нашем корпусе данных

Для загрузки PDF-файла используется метод *open(path_to_file)*.

`pdf = pdfplumber.open('./tables/one-2.pdf')`

Для обращения к страницам PDF-файла используется метод *pages*. Он возвращает список, содержащий один экземпляр *pdfplumber.Page* для каждой загруженной страницы.

`p0 = pdf.pages[0]`

Метод *to_image()* возвращает экземпляр класса *PageImage*. Затем метод *debug_tablefinder(table_settings={})* можно использовать для отображения таблицы на изображении, т. к. этот метод возвращает экземпляр класса *TableFinder* с доступом к свойствам *edges*, *intersections*, *cells* и *tables*. 

`im = p0.to_image()`

`im.debug_tablefinder()`

![Пример удачной работы PDFplumber (Пример 1)](https://github.com/owls-nlp/system-for-tagging/blob/master/info/img/PDFPlumber_test_1.png)

**Рис. 1.** Пример удачной работы PDFplumber (Пример 1)

![Пример неудачной работы PDFplumber, когда таблица не находится](https://github.com/owls-nlp/system-for-tagging/blob/master/info/img/PDFPlumber_test_2.png)

**Рис. 2.** Пример неудачной работы PDFplumber, когда таблица не находится

![Пример неудачной работы PDFplumber, когда таблица выделяется некорректно](https://github.com/owls-nlp/system-for-tagging/blob/master/info/img/PDFPlumber_test_3.png)

**Рис. 3.** Пример неудачной работы PDFplumber, когда таблица выделяется некорректно

![Пример неудачной работы PDFplumber, когда таблица находится там, где нет таблицы](https://github.com/owls-nlp/system-for-tagging/blob/master/info/img/PDFPlumber_test_4.jpg)

**Рис. 4.** Пример неудачной работы PDFplumber, когда таблица находится там, где нет таблицы

Для извлечения таблиц используется метод *extract_tables(table_settings={})*. Возвращает текст, извлеченный из всех таблиц, найденных на странице, в виде списка списков списков со структурой таблица, строка, ячейка.

`table = p0.extract_tables()`

`df = pd.DataFrame(table[0][1:], columns=table[0][0])`

![Первая таблица, извлеченная из примера 1, в виде Pandas DataFrame](https://github.com/owls-nlp/system-for-tagging/blob/master/info/img/PDFPlumber_test_5.png)

**Рис. 5.** Первая таблица, извлеченная из примера 1, в виде Pandas DataFrame

![Вторая таблица, извлеченная из примера 1, в виде Pandas DataFrame](https://github.com/owls-nlp/system-for-tagging/blob/master/info/img/PDFPlumber_test_6.png)

**Рис. 6.** Вторая таблица, извлеченная из примера 1, в виде Pandas DataFrame

Таким образом, **PDFplumber** справляется с таблицами, у которых есть границы (выделены границы самой таблицы, строки и столбцы), не находит таблицы без границ и плохо справляется с теми, у которых нестандартный стиль. Соответственно, логичнее всего находить таблицы среди тех информационных блоков, которые помечены классом «таблица», и, если не получилось извлечь из блока таблицу, сохранить ее как изображение. 
