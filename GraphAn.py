import graphviz as gviz

from borb.pdf import Document
from borb.pdf import Page
from borb.pdf import SingleColumnLayout
from borb.pdf import FixedColumnWidthTable
from borb.pdf import Paragraph
from borb.pdf import Image
from borb.pdf import PDF
from decimal import Decimal
from pathlib import Path
from PIL import Image as ImgSize

from math import ceil
import time


'''
(Eng) Class that contains all the information about a given graph    |
---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---|
(Укр) Клас, який містить всю інформацію щодо заданого графа          |
'''


class Graph:
  def __init__(self, graph, gSymbol, vSymbol, eSymbol):
    self.graph: dict = graph
    self.size: int = len(graph)
    self.name: str = gSymbol
    self.vertex: str = vSymbol
    self.edge: str = eSymbol
    self.fullName: str = f"{gSymbol} = ({vSymbol}, {eSymbol})"
    self.hd: dict = self.countHalfDegrees()

  '''
  (Eng) A method that calculates the outdegrees| 
  and indegrees of a given graph               |
  ---   ---   ---   ---   ---   ---   ---   ---|
  (Укр) Метод, який розраховує напівстпені     |
  виходу та входу заданого графа               |
  '''

  def countHalfDegrees(self):
    hd = {}
    for vertex in self.graph:
      hdOut = len(self.graph[vertex])
      hdIn = 0
      for adjList in self.graph.values():
        hdIn += adjList.count(vertex)
      hd[vertex] = [hdOut, hdIn]
    return hd

  '''
  (Eng) A method that formats a string to display information correctly      |
  ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---|
  (Укр) Метод, який форматувє строку для коректного зображення інформації    |
  '''

  @staticmethod
  def stringFormat(line, r1, r2):
    return line.replace('[', r1).replace(']', r2).replace("'", "")

  '''
  (Eng) The method that prints the calculated table of outdegrees      |
  and indegrees using adjacency lists representing the given graph     |
  ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---|
  (Укр) Метод, який друкує обраховану таблицю напівступенів            |
  із списками суміжності, якими заданий граф                           |
  '''

  def printHalfDegreesTable(self, output, todoPDF, printDetails):
    if printDetails:
      print(f"Граф {self.fullName} | Розмірність: {self.size}")
    if todoPDF:
      output.append(f"Graph {self.fullName} | Size: {self.size}")

    for vertex in self.graph:
      if printDetails:
        print(f"{vertex}" +
              self.stringFormat(f"\t| {self.hd[vertex]} |\t", '(', ')') +
              self.stringFormat(f"{self.graph[vertex]}", '{', '}'))
      if todoPDF:
        output.append(f"{vertex} | " +
                      self.stringFormat(f"{self.hd[vertex]} | ", '(', ')') +
                      self.stringFormat(f"{self.graph[vertex]}", '{', '}'))

  '''
  (Eng) A method that generates a graphical representation of a given graph  |
  ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---|
  (Укр) Метод, який формує графічне представлення заданого графа             |
  '''

  def buildGraphImage(self):
    g = gviz.Digraph('graph', engine='neato',
                     graph_attr={'splines': 'true', 'overlap': 'false',
                                 'sep': str(1.5), 'normalize': 'true',
                                 'label': f"{self.fullName}",
                                 'fontsize': str(20)},
                     node_attr={'shape': 'circle', 'fontsize': str(20)})
    for vertex in self.graph:
      g.node(vertex, vertex)
      for nextvertex in self.graph[vertex]:
        g.edge(vertex, nextvertex)
    try:
      g.render(filename=f"{self.fullName}", format="png", directory="pngs")
      return True
    except:
      return False


'''
(Eng) An analysis class that implements all the steps of |
the isomorphic embedding analysis algorithm              |
---   ---   ---   ---   ---   ---   ---   ---   ---   ---|
(Укр) Клас аналізу, в якому імплетовано всі кроки        |
алгоритму аналізу ізомофрного вкладення                  |
'''


class Analysis:
  def __init__(self, graph1, graph2,
               fileName=None, todoPDF=False, printDetails=False):
    self.graph1: Graph = graph1
    self.graph2: Graph = graph2
    self.fileName: str = fileName
    self.todoPDF: bool = todoPDF
    self.printDetails: bool = printDetails
    self.output: list = [[], []]
    self.maxSubs: dict = {}
    self.maxVariants: list = []
    self.partialSubs: list = []
    self.partialVariants: list = []
    self.completeSubs: list = []
    self.analysisResult: int = 0
    self.analysisTime: float = 0

  '''
  (Eng) Resetting attributes before performing a new analysis                |
  ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---|
  (Укр) Скидання атрибутів, що проводиться перед проведенням нового аналізу  |
  '''

  def clear(self):
    self.output = [[], []]
    self.maxSubs = {}
    self.maxVariants = []
    self.partialSubs = []
    self.partialVariants = []
    self.completeSubs = []
    self.analysisResult = 0
    self.analysisTime = 0

  '''
  (Eng) Generating a report in PDF format with the results of the analysis   |
  ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---|
  (Укр) Формування звіту в форматі PDF з результатами зробленого аналізу     |
  '''

  def makePDF(self):
    pdf = Document()

    page1 = Page()
    pdf.add_page(page1)
    layout1 = SingleColumnLayout(page1)

    img = ImgSize.open(f"pngs/{self.graph1.fullName}.png")
    (w, h) = img.size
    if (h > 660):
      w = int(w * 660 / h)
      h = 660
    if (w > 470):
      h = int(h * 470 / w)
      w = 470
    layout1.add(Image(Path(f"pngs/{self.graph1.fullName}.png"),
                      width=w, height=h))

    for line in self.output[0]:
      layout1.add(Paragraph(line, font="Courier"))

    page2 = Page()
    pdf.add_page(page2)
    layout2 = SingleColumnLayout(page2)

    img = ImgSize.open(f"pngs/{self.graph2.fullName}.png")
    (w, h) = img.size
    if (h > 660):
      w = int(w * 660 / h)
      h = 660
    if (w > 470):
      h = int(h * 470 / w)
      w = 470
    layout2.add(Image(Path(f"pngs/{self.graph2.fullName}.png"),
                      width=w, height=h))
    for line in self.output[1]:
      layout2.add(Paragraph(line, font="Courier"))

    page3 = Page()
    pdf.add_page(page3)
    layout3 = SingleColumnLayout(page3)

    for text in self.output[2:]:
      for line in text[0]:
        layout3.add(Paragraph(line, font="Courier"))
      if len(text) == 3:
        for line in text[1]:
          layout3.add(Paragraph(line[0], font="Courier"))
          numCol = len(line[1])
          numRow = 2 * ceil(numCol / 13)
          table = FixedColumnWidthTable(number_of_columns=13,
                                        number_of_rows=numRow)
          for col in range(0, numCol - 1, 13):
            for pos in range(col, col + 13, 1):
              if (pos < numCol):
                table.add(Paragraph(line[1][pos], font="Courier"))
              else:
                table.add(Paragraph('', font="Courier"))
            for pos in range(col, col + 13, 1):
              if (pos < numCol):
                table.add(Paragraph(line[2][pos], font="Courier"))
              else:
                table.add(Paragraph('', font="Courier"))
          table.set_padding_on_all_cells(Decimal(2), Decimal(2),
                                         Decimal(2), Decimal(2))
          layout3.add(table)
          layout3.add(Paragraph('', font="Courier"))
        for line in text[-1]:
          layout3.add(Paragraph(line, font="Courier"))

    with open(Path(f"{self.fileName}.pdf"), "wb") as pdf_file_handle:
      PDF.dumps(pdf_file_handle, pdf)
    print(f"Звіт збережено до файлу {self.fileName}.pdf")

  '''
  (Eng) Function for displaying a substitution to the console.   |
  ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---|
  (Укр) Функція для вивиедення підстановки до консолі.           |
  '''

  @staticmethod
  def printSub(sub):
    print('\t'.join(sub.keys()))
    print('\t'.join(sub.values()))

  '''
  (Eng) A function for outputting text to the console and a pdf report.|
  ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---|
  (Укр) Функція для виводу тексту в консоль та pdf звіт.               |
  '''

  def makeRecord(self, output, engText, ukrText):
    if self.todoPDF:
      for text in engText:
        output.append(text)
    if self.printDetails:
      for text in ukrText:
        print(text)

  '''
  (Eng) Function that checks the possibility of finding a combination (one)  |
  without repeating vertices that satisfies condition A of Theorem 1.        |
  ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---|
  (Укр) Функція, яка перевіряє можливість находження комбінації (однієї)     |
  без повторення вершин, яка задовольняє умові А теореми 1.                  |
  '''

  def findCombCondA(self, variants, current):
    if len(current) == len(variants):
      return True

    for i in variants:
      if i not in current.keys():
        for j in variants[i]:
          if j not in current.values():
            current[i] = j
            res = self.findCombCondA(variants, current)
            if res:
              return True
            current.pop(i)
        break
    return False

  '''
  (Eng) Verification of condition A of Theorem 1     |
  ---   ---   ---   ---   ---   ---   ---   ---   ---|
  (Укр) Перевірка виконання умови А теореми 1        |
  '''

  def checkCondA(self):
    possiblevertexs = {}
    for vertex1 in self.graph1.hd:
      possiblevertexs[vertex1] = list(
        filter(lambda vertex2:
               self.graph1.hd[vertex1][0] <= self.graph2.hd[vertex2][0]
               and self.graph1.hd[vertex1][1] <= self.graph2.hd[vertex2][1],
               self.graph2.hd)
      )
      if len(possiblevertexs[vertex1]) == 0:
        return False
    return self.findCombCondA(possiblevertexs, {})

  '''
  (Eng) Verification of (partial) substitution of condition B of Theorem 1  |
  ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---  ---|
  (Укр) Перевірка виконання (частковою) підстановкою умови B теореми 1      |
  '''

  def conditionB(self, sub):
    for vertex in sub.keys():
      for nextvertex in self.graph1.graph[vertex]:
        if nextvertex in sub.keys() \
                and sub[nextvertex] not in self.graph2.graph[sub[vertex]]:
          if self.printDetails:
            print(f"При перевірці умови B теореми 1 підстановки")
            self.printSub(sub)
            print("виявилено порушення:")
            print(
              f"{nextvertex} є F{vertex}={self.graph1.graph[vertex]}, "
              f"але {sub[nextvertex]} ~є P{sub[vertex]}="
              f"{self.graph2.graph[sub[vertex]]}\n")
          return False
    return True

  '''
  (Eng) Finding all combinations without repetition  |
  that satisfy condition B of Theorem 1              |
  ---   ---   ---   ---   ---   ---   ---   ---   ---|
  (Укр) Знаходження всіх комбінацій без повторення,  | 
  які задовольняють умові B теореми 1                |
  '''

  def findCombCondB(self, variants, res, current):
    if len(current) == len(variants) and self.conditionB(current):
      res.append(current.copy())
      return

    for i in variants:
      if i not in current.keys():
        for j in variants[i]:
          if j not in current.values():
            current[i] = j
            if self.conditionB(current):
              self.findCombCondB(variants, res, current)
            current.pop(i)
        break

  '''
  (Eng) Mapping of vertices with the largest output  |
  half-degree according to condition A of Theorem 1  |
  ---   ---   ---   ---   ---   ---   ---   ---   ---|
  (Укр) Зіставлення вершин із найбільшим             |
  напівступенем виходу згідно умові А теореми 1      |
  '''

  def makeMaxSub(self):
    hdG1SortOut = {k: v for k, v in sorted(self.graph1.hd.items(),
                                           key=lambda elem: elem[1][0],
                                           reverse=True)}
    key = list(
      filter(lambda v:
             hdG1SortOut[v][0] == hdG1SortOut[list(hdG1SortOut.keys())[0]][0],
             hdG1SortOut)
    )
    for vertex1 in key:
      for vertex2 in self.graph2.hd:
        if self.graph2.hd[vertex2][0] >= self.graph1.hd[vertex1][0] and \
                self.graph2.hd[vertex2][1] >= self.graph1.hd[vertex1][1]:
          if vertex1 in self.maxSubs:
            self.maxSubs[vertex1].append(vertex2)
          else:
            self.maxSubs[vertex1] = [vertex2]
    if (self.printDetails):
      for maxVertex in self.maxSubs:
        print(f"Вершину {maxVertex}, з максимальним НВ, можна зіставити "
              f"з вершинами " +
              Graph.stringFormat(f"{self.maxSubs[maxVertex]}", '{', '}'))

  '''
  (Eng) Formation of lists of possible mappings for initial vertices   |
  ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---|
  (Укр) Формування списків можливих зіставлень для початкових вершин   |
  '''

  def makeMaxVariants(self):
    for maxV in self.maxSubs:
      for maxSub in self.maxSubs[maxV]:
        partSub = {maxV: [maxSub]}
        otherVertexs = list(filter(lambda vertex:
                                   vertex != maxV,
                                   self.graph1.graph[maxV]))
        for vToFind in otherVertexs:
          if vToFind != maxV:
            possibleSubs = list(
              filter(lambda v:
                     self.graph1.hd[vToFind][0] <= self.graph2.hd[v][0]
                     and self.graph1.hd[vToFind][1] <= self.graph2.hd[v][1]
                     and v not in partSub[maxV],
                     self.graph2.graph[maxSub])
            )
            if len(possibleSubs) == 0:
              break
            partSub[vToFind] = possibleSubs
            if vToFind == otherVertexs[-1]:
              self.maxVariants.append(partSub)

  '''
  (Eng) Finding partial substitutions that     | 
  satisfy condition B of Theorem 1             |
  ---   ---   ---   ---   ---   ---   ---   ---|
  (Укр) Знаходження часткових підстановок, що  |
  задовольняють умові B теореми 1              |
  '''

  def makePartialSubs(self):
    partSubs = []
    for var in self.maxVariants:
      self.findCombCondB(var, partSubs, {})
    self.partialSubs = partSubs
    if len(self.partialSubs) != 0 and self.printDetails:
      print("Частокві підстановки, що задовольняють умові B теореми 1:")
      for index, sub in enumerate(self.partialSubs):
        print(f"Підстановка №{index + 1}:")
        self.printSub(sub)
        print()

  '''
  (Eng) Generate lists of possible mappings for the remaining vertices |
  ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---|
  (Укр) Формування списків можливих зіставлень для решти вершин        |
  '''

  def makePartialVariants(self):
    for sub in self.partialSubs:
      valuesToCheck = sub.values()
      sub = dict(map(lambda kv: (kv[0], [kv[1]]), sub.items()))
      if sub.keys() != self.graph1.graph.keys():
        otherKeys = list(filter(lambda v1:
                                v1 not in sub.keys(),
                                self.graph1.graph.keys()))
        for vertex1 in otherKeys:
          possibleAdd = []
          for vertex2 in sub.keys():
            if vertex1 in self.graph1.graph[vertex2]:
              for vertex3 in sub[vertex2]:
                possibleAdd.extend(
                  filter(lambda v4:
                         self.graph1.hd[vertex1][0] <= self.graph2.hd[v4][0]
                         and self.graph1.hd[vertex1][1] <= self.graph2.hd[v4][1]
                         and v4 not in valuesToCheck
                         and v4 not in possibleAdd,
                         self.graph2.graph[vertex3])
                )
              break
          if possibleAdd == []:
            possibleAdd = list(
              filter(lambda v3:
                     self.graph1.hd[vertex1][0] <= self.graph2.hd[v3][0]
                     and self.graph1.hd[vertex1][1] <= self.graph2.hd[v3][1]
                     and v3 not in valuesToCheck,
                     self.graph2.graph.keys())
            )
          if len(possibleAdd) == 0:
            break
          sub[vertex1] = possibleAdd
      if len(sub.keys()) == len(self.graph1.graph.keys()):
        self.partialVariants.append(sub)

  '''
  (Eng) Finding the resulting substitutions    |
  that satisfy condition B of Theorem 1        |
  ---   ---   ---   ---   ---   ---   ---   ---|
  (Укр) Знаходження результуючих підстановок,  |
  що задовольняють умову B теореми 1           |
  '''

  def makeCompleteSubs(self):
    completeSubs = []
    for var in self.partialVariants:
      self.findCombCondB(var, completeSubs, {})
    self.completeSubs = completeSubs

  '''
  (Eng) Step-by-step implementation of the algorithm for analyzing the |
  isomorphic embedding of two directed graphs                          |
  ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---|
  (Укр) Покрокове виконання алгоритму аналізу ізоморфного вкладення    |
  двох орієнтованих графів                                             |
  '''

  def algorithm(self):
    output = [[]]
    self.makeRecord(
      output[0],
      [f"Checking the isomorphic embedding of the graph {self.graph1.fullName} "
       f"into the graph {self.graph2.fullName}:"],
      [f"Перевірка ізоморфного вкладення графа {self.graph1.fullName} "
       f"у граф {self.graph2.fullName}"]
    )
    if self.checkCondA():
      self.makeMaxSub()
      self.makeMaxVariants()
      self.makePartialSubs()
      if len(self.partialSubs) != 0:
        self.makePartialVariants()
        self.makeCompleteSubs()
        if len(self.completeSubs) != 0:
          if len(self.completeSubs) != 0 and \
                  (self.printDetails or self.todoPDF):
            self.makeRecord(
              output[0],
              ["The resulting substitutions that satisfy "
               "condition B of Theorem 1:"],
              ["Результуючі підстановки, які задовольняють умові B теореми 1:"]
            )
            outputSub = []
            for index, sub in enumerate(self.completeSubs):
              if self.printDetails:
                print(f"Підстанвока №{index + 1}:")
                self.printSub(sub)
                print()
              if self.todoPDF:
                outputSub.append([f"Substitution {index + 1}:",
                                  list(sub.keys()),
                                  list(sub.values())])
            if self.todoPDF:
              output.append(outputSub)
              output.append([])
              self.makeRecord(
                output[-1],
                [f"The graph {self.graph1.fullName} is isomorphically embedded "
                 f"in the graph {self.graph2.fullName}"],
                [f"Граф {self.graph1.fullName} ізоморфно вкладається у граф "
                 f"{self.graph2.fullName}"]
              )
              self.output.append(output)
            return 0
        else:
          self.makeRecord(
            output[0],
            [f"When trying to complete all possible substitutions, no option "
             f"was found that fulfilled the conditions A and B of Theorem 1, "
             f"therefore the graph {self.graph1.fullName} is NOT isomorphically"
             f" embedded in the graph {self.graph2.fullName}"],
            ["При доповнені часткових підстановок не вдалося знайти варіант, "
             "який задовольняє умовам A та B Теореми 1.",
             f"Граф {self.graph1.fullName} ізоморфно не вкладається у граф "
             f"{self.graph2.fullName}"]
          )
          if self.todoPDF:
            self.output.append(output)
          return 3
      else:
        self.makeRecord(
          output[0],
          ["No partial substitutions were found that satisfy condition B of "
           "Theorem 1.", f"The graph {self.graph1.fullName} is NOT "
                         f"isomorphically embedded in the graph {self.graph2.fullName}."],
          [f"Не вдалося знайти часткові підстановки, які задовольняють умові B "
           f"теореми 1.\nГраф {self.graph1.fullName} ізоморфно НЕ вкладається"
           f" до графа {self.graph2.fullName}."]
        )
        if self.todoPDF:
          self.output.append(output)
        return 2
    else:
      self.makeRecord(
        output[0],
        [f"The given graphs do not satisfy  condition A of Theorem 1.",
         f"The graph {self.graph1.fullName} is NOT isomorphically embedded "
         f"in the graph {self.graph2.fullName}"],
        [f"Введені графи не задовольняють умову A теореми 1.",
         f"Граф {self.graph1.fullName} ізоморфно не вкладається у граф "
         f"{self.graph2.fullName}"]
      )
      if self.todoPDF:
        self.output.append(output)
      return 1

  '''
  (Eng) Function that controls the analysis and returns its result.    |
  (0 - graphs are isomorphically embedded;                             |
  1, 2 and 3 - graphs are not  isomorphically embedded)                |
  ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---|
  (Укр) Функція яка керує аналізом і повертає його результат.          |
  (0 - графи ізоморфно вкладаються;                                    |
  1, 2 та 3 - графи ізоморфно не вкладаються)                          |
  '''

  def makeAnalysis(self):
    self.clear()
    startTime = time.time()
    isomorphicCheck = False
    if self.graph1.size > self.graph2.size:
      self.graph1, self.graph2 = self.graph2, self.graph1
    elif self.graph1.size == self.graph2.size:
      isomorphicCheck = True

    self.graph1.buildGraphImage()
    self.graph2.buildGraphImage()

    self.graph1.printHalfDegreesTable(self.output[0],
                                      self.todoPDF, self.printDetails)
    self.graph2.printHalfDegreesTable(self.output[1],
                                      self.todoPDF, self.printDetails)

    if isomorphicCheck:
      self.analysisResult = self.algorithm()
      self.graph1, self.graph2 = self.graph2, self.graph1
      self.analysisResult += self.algorithm()
      self.graph1, self.graph2 = self.graph2, self.graph1

      if self.analysisResult == 0:
        if self.todoPDF or self.printDetails:
          self.makeRecord(
            self.output[-1][-1],
            [f"The graphs {self.graph1.fullName} and {self.graph2.fullName} "
             f"are isomorphic"],
            [f"Графи {self.graph1.fullName} і {self.graph2.fullName} ізоморфні"]
          )
    else:
      self.analysisResult = self.algorithm()

    endTime = time.time()

    if self.todoPDF:
      self.makeRecord(self.output[-1][-1],
                      [f"Program execution time: {endTime - startTime} seconds"],
                      [f"Час виконання програми: {endTime - startTime} секунд"])
    else:
      self.makeRecord([],
                      [f"Program execution time: {endTime - startTime} seconds"],
                      [f"Час виконання програми: {endTime - startTime} секунд"])

    if self.todoPDF:
      self.makePDF()

    return self.analysisResult


if __name__ == '__main__':
  pass
