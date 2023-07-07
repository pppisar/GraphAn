from random import randint
from math import comb, ceil

from GraphAn import Graph, Analysis

'''
(Eng) A class that is responsible for creating a supergraph    |
and subgraph in accordance with the specified condition of     |
of isomorphic embedding                                        |
---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---|
(Укр) Клас, який відповідає за створення надграфа та підграфа  |
відповідно до заданої умови ізоморфного вкладення              |
'''

class Generator:
  def __init__(self):
    self.supergraph: Graph = None
    self.subgraph: Graph = None
    self.edgesCount: list = None
    self.cuttingSub: dict = None
    self.analysis: Analysis = None
    self.embeddingCondition: bool = None

  '''
  (Eng) Method for getting a value of type Int from the user     |
  ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---|
  (Укр) Метод для отримання від користувача значення типу Int    |
  '''

  @staticmethod
  def inputNum(request, max = None):
    while True:
      inp = input(f"{request}: ")
      if inp.isnumeric():
        if max is None:
          return int(inp)
        elif max is not None and int(inp) < max:
          return int(inp)

  '''
  (Eng) Method for getting a one-character string from the user        |
  ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---|
  (Укр) Метод для отримання односимвольного рядка від користувача      |
  '''

  @staticmethod
  def inputChar(request):
    while True:
      inp = input(f"{request}: ")
      if inp.isalpha() and len(inp) == 1:
        return inp.upper()


  '''
  (Eng) Method for converting user input to a value of type Bool             |
  ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---|
  (Укр) Метод для конвертації рядка, що було введено, до значення типу Bool  |
  '''

  @staticmethod
  def stringToBool(request, condTrue, condFalse):
    while True:
      inp = input(f"{request}: ")
      if inp.lower() == condTrue.lower():
        return True
      elif inp.lower() == condFalse.lower():
        return False

  '''
    (Eng) Method for generating the number of incident edges for each vertex  |
    ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---  ---|
    (Укр) Метод генерації кількості інцедентних ребер для кожної вершини      |
    '''

  def generateEdges(self, size, nullVertex):
    totalSum = comb(size, 2)
    generatedSum = 0
    edgesCount = []
    for i in range(0, size):
      if nullVertex:
        edgesCount.append(randint(0, size))
        generatedSum += edgesCount[i]
      else:
        edgesCount.append(randint(1, size))
        generatedSum += edgesCount[i]

    if generatedSum > totalSum:
      scale = generatedSum / totalSum
      generatedSum = 0
      for i in range(0, size):
        edgesCount[i] = ceil(edgesCount[i] / scale)
        generatedSum += edgesCount[i]
    while generatedSum != totalSum:
      vrtx = randint(0, size - 1)
      if generatedSum < totalSum:
        if edgesCount[vrtx] < size:
          edgesCount[vrtx] += 1
          generatedSum += 1
      else:
        if edgesCount[vrtx] > 1:
          edgesCount[vrtx] -= 1
          generatedSum -= 1
    self.edgesCount = edgesCount

  '''
  (Eng) The method responsible for generating of the supergraph  |
  ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---|
  (Укр) Метод, що відповідає за формування надграфа              |
  '''

  def generateSupergraph(self, size, nullVertex, gSymbol, vSymbol, eSymbol):
    self.generateEdges(size, nullVertex)
    graph = {}
    for i in range(1, size + 1):
      graph[f"{vSymbol.lower()}{i}"] = []
      while len(graph[f"{vSymbol.lower()}{i}"]) != self.edgesCount[i-1]:
        vertex = f"{vSymbol.lower()}{randint(1, size)}"
        if (vertex not in graph[f"{vSymbol.lower()}{i}"]):
          graph[f"{vSymbol.lower()}{i}"].append(vertex)
    self.supergraph = Graph(graph,
                            gSymbol.upper(), vSymbol.upper(), eSymbol.upper())

  '''
  (Eng) A method that creates a substitution according     |
  to the generated set of vertices to be removed.          |
  ---   ---   ---   ---   ---   ---   ---   ---   ---   ---|
  (Укр) Метод, що формує відображення відповідно до        |
  згенерованої множини вершин, що потрібно видалити        |
  '''

  def cutGraph(self, size, vSymbol):
    toDelete = []
    while len(toDelete) != self.supergraph.size - size:
      vertexToDelete = f"{self.supergraph.vertex.lower()}{randint(1, size)}"
      if vertexToDelete not in toDelete:
        toDelete.append(vertexToDelete)
    sub = {}
    for index, vertex in enumerate(filter(lambda vertex: vertex not in toDelete, self.supergraph.graph)):
      sub[vertex] = f"{vSymbol.lower()}{index + 1}"
    self.cuttingSub = sub

  '''
  (Eng) A method that additionally truncates the set of edges          |
  (used if the subgraph must be isomorphically nested)                 |
  ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---|
  (Укр) Метод, який додатково усікає множину ребер підграфа            |
  (використовується у випадку, якщо підграф ізоморфно вкладається)     |
  '''

  def reduceGraph(self, gSymbol, vSymbol, eSymbol):
    graph = {}
    for vertex1 in self.cuttingSub:
      graph[self.cuttingSub[vertex1]] = []
      edgeDeleteCount = int(len(self.supergraph.graph[vertex1]) / 3)
      for vertex2 in self.supergraph.graph[vertex1]:
        if vertex2 in self.cuttingSub:
          edgeStatus = False
          if edgeDeleteCount > 0:
           edgeStatus = randint(0, 1)
           if edgeStatus:
             edgeDeleteCount -= 1
          if not edgeStatus:
            graph[self.cuttingSub[vertex1]].append(self.cuttingSub[vertex2])
    self.subgraph = Graph(graph,
                          gSymbol.upper(), vSymbol.upper(), eSymbol.upper())

  '''
  (Eng) A method that adds edges to the formed subgraph to violate     |
  condition B of Theorem 1 in the resulting substitutions              |
  ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---|
  (Укр) Метод, що додає ребра сформованому підграфу для порушення      |
  умови B теореми 1 в отриманих підстановках                           |
  '''

  def enlargeGraph(self, gSymbol, vSymbol, eSymbol):
    while True:
      self.reduceGraph(gSymbol, vSymbol, eSymbol)

      self.analysis = Analysis(self.subgraph, self.supergraph,
                               "", False, False)
      self.analysis.makeAnalysis()
      for sub in self.analysis.completeSubs:
        if self.analysis.conditionB(sub):
          canBeConnectedFrom = list(
            filter(lambda vrtx:
                   self.subgraph.hd[vrtx][0] < self.supergraph.hd[sub[vrtx]][0],
                   sub.keys()))
          for vrtxFrom in canBeConnectedFrom[-1::-1]:
            canBeConnectedTo = list(
              filter(lambda v:
                     v not in self.subgraph.graph[vrtxFrom] and
                     sub[v] not in self.supergraph.graph[sub[vrtxFrom]]
                     and self.subgraph.hd[v][1] < self.supergraph.hd[sub[v]][1],
                     sub.keys()))
            if len(canBeConnectedTo) > 0:
              self.subgraph.graph[vrtxFrom].append(canBeConnectedTo[0])
              break
      self.analysis.printDetails = False
      if self.analysis.makeAnalysis() == 3:
        return 1

  '''
  (Eng) The method responsible for forming a subgraph| 
  according to the isomorphic embedding condition    |
  ---   ---   ---   ---   ---   ---   ---   ---   ---|
  (Укр) Метод, відповідальний за формування підграфа |
  відповідно до умови ізоморфного вкладення          |
  '''

  def generateSubgraph(self, size, gSymbol, vSymbol, eSymbol):
    self.cutGraph(size, vSymbol)
    if self.embeddingCondition:
      self.reduceGraph(gSymbol, vSymbol, eSymbol)
    else:
      self.enlargeGraph(gSymbol, vSymbol, eSymbol)

  '''
  (Eng) A method that automatically generates a variant                |
  (supergraph and subgraph) and makes a PDF report of the analysis     |
  ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---|
  (Укр) Метод, який автоматично формує варіант (надграф і підграф)     |
  та робить PDF звіт аналізу по згенерованим графам                    |
  '''

  def createVariant(self):
    g1Size = self.inputNum("Введіть розмірність надграфа")
    g1Symbol = self.inputChar("Введіть позначення надграфа")
    v1Symbol = self.inputChar("Введіть позначення множини вершин надграфа")
    e1Symbol = self.inputChar("Введіть позначення множини ребер надграфа")
    nullVrtx = self.stringToBool("Чи може вершина не містити можливих переходів?"
                                 "(\"Так\", або \"Ні\")", "Так", "Ні")
    self.generateSupergraph(g1Size, nullVrtx, g1Symbol, v1Symbol, e1Symbol)
    self.embeddingCondition = self.stringToBool("Чи повинен підграф вкладатися?"
                                                "(\"Так\", або \"Ні\")",
                                                "Так", "Ні")
    g2Size = self.inputNum("Введіть розмірність підграфа", g1Size)
    g2Symbol = self.inputChar("Введіть позначення підграфа")
    v2Symbol = self.inputChar("Введіть позначення множини вершин підграфа")
    e2Symbol = self.inputChar("Введіть позначення множини ребер підграфа")
    self.generateSubgraph(g2Size, g2Symbol, v2Symbol, e2Symbol)
    fileName = "variant" + str(self.inputNum("Введіть номер варіанту"))
    print("Надграф:")
    print(self.supergraph.graph)
    print("Підграф:")
    print(self.subgraph.graph)
    self.analysis = Analysis(self.subgraph, self.supergraph,
                             fileName, True, False)
    self.analysis.makeAnalysis()

if __name__ == '__main__':
  var1 = Generator()
  var1.createVariant()