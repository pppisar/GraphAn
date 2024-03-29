from random import randint
from math import comb, ceil

from GraphAn import Graph, Analysis

'''
A class that is responsible for creating a supergraph 
and subgraph in accordance with the specified condition of isomorphic embedding
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
  Method for getting a value of type Int from the user
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
  Method for getting a one-character string from the user
  '''

  @staticmethod
  def inputChar(request):
    while True:
      inp = input(f"{request}: ")
      if inp.isalpha() and len(inp) == 1:
        return inp.upper()

  '''
  Method for converting user input to a value of type Bool
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
  Method for generating the number of incident edges for each vertex
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
  The method responsible for generating of the supergraph
  '''

  def generateSupergraph(self, size, nullVertex, gSymbol, vSymbol, eSymbol):
    self.generateEdges(size, nullVertex)
    graph = {}
    for i in range(1, size + 1):
      graph[f"{vSymbol.lower()}{i}"] = []
      while len(graph[f"{vSymbol.lower()}{i}"]) != self.edgesCount[i-1]:
        vertex = f"{vSymbol.lower()}{randint(1, size)}"
        if vertex not in graph[f"{vSymbol.lower()}{i}"]:
          graph[f"{vSymbol.lower()}{i}"].append(vertex)
    self.supergraph = Graph(graph,
                            gSymbol.upper(), vSymbol.upper(), eSymbol.upper())

  '''
  A method that creates a substitution according to the generated set of vertices to be removed.
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
  A method that additionally truncates the set of edges  (used if the subgraph must be isomorphically nested)
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
  A method that adds edges to the formed subgraph to violate condition B of Theorem 1 in the resulting substitutions
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
  The method responsible for forming a subgraph according to the isomorphic embedding condition
  '''

  def generateSubgraph(self, size, gSymbol, vSymbol, eSymbol):
    self.cutGraph(size, vSymbol)
    if self.embeddingCondition:
      self.reduceGraph(gSymbol, vSymbol, eSymbol)
    else:
      self.enlargeGraph(gSymbol, vSymbol, eSymbol)

  '''
  A method that automatically generates a variant (supergraph and subgraph) and makes a PDF report of the analysis
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