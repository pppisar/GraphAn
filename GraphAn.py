import numpy as np

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
from PIL import Image as ImgReader

from math import ceil
import time


def readJSON(fileName: str) -> dict:
  """
  A function that reads a graph specified by adjacency lists from a JSON file
  and converts it to a structure of type dictionary

  Args:
    fileName (str): Name of the JSON file

  Returns:
    Adjacency lists in dictionary view
  """
  # TODO: Reading of JSON File
  pass


class Graph:
  """
  Class that contains all the information about a given graph

  Args:
    graph (dict): A graph defined using adjacency lists
    gSymbol (str): Name of the graph (Symbol by which it is denoted)
    vSymbol (str): Name of the vertex set (Symbol by which it is denoted)
    eSymbol (str): Name of the edges set (Symbol by which it is denoted)

  Attributes:
    graph (np.ndarray): Graph adjacency matrix using Numpy array
    size (int): Graph dimensionality
    name (str): Full name of the graph (including the set of vertices and edges)
    vertex (str): Name of the vertex set (Symbol by which it is denoted)
    edge (str): Name of the edges set (Symbol by which it is denoted)
    hd (np.ndarray): The calculated table of outdegrees and indegrees
    transition (dict): Marking a graph vertex with a number (used when switching from the Numpy representation)
  """

  def __init__(self, graph: dict, gSymbol: str, vSymbol: str, eSymbol: str):
    self.size: int = len(graph)
    self.name: str = gSymbol
    self.vertex: str = vSymbol
    self.edge: str = eSymbol
    self.fullName: str = f"{gSymbol} = ({vSymbol}, {eSymbol})"
    self.transition: dict = dict()
    self.graph: np.ndarray = self.transformToMatrix(graph)
    self.hd: np.ndarray = self.countHalfDegrees(graph)

  def transformToMatrix(self, graph: dict) -> np.ndarray:
    """
    A method that converts adjacency lists to an adjacency matrix

    Args:
      graph (dict): Adjacency lists defined with a dictionary

    Returns:
      adjacencyMatrix (np.ndarray): Graph adjacency matrix
    """
    transitionToNumber = dict()

    matrixSize = len(graph.keys())
    adjacencyMatrix = np.zeros((matrixSize, matrixSize), dtype=np.uint32)

    for index, vertex in enumerate(graph.keys()):
      self.transition[index] = vertex
      transitionToNumber[vertex] = index
    for vertexFrom, edges in enumerate(graph.values()):
      for vertexTo in edges:
        adjacencyMatrix[vertexFrom, transitionToNumber[vertexTo]] = 1

    return adjacencyMatrix

  def countHalfDegrees(self, graph: dict) -> np.ndarray:
    """
    A method that calculates the outdegrees and indegrees of a given graph

    Args:
      graph (dict): Adjacency lists defined with a dictionary

    Returns:
      halfDegrees: The calculated table of outdegrees and indegrees
    """
    halfDegrees = np.zeros((len(graph.keys()), 2), dtype=np.uint32)

    for index, vertex in enumerate(graph.keys()):
      hdOut = len(graph[vertex])
      hdIn = np.count_nonzero(self.graph[:, index] == 1)
      halfDegrees[index] = [hdOut, hdIn]
    return halfDegrees

  @staticmethod
  def stringFormat(line, r1, r2) -> str:
    """
    A method that formats a string to display information correctly

    Args:
      line (str): The line in which you need to replace the brackets [ and ]
      r1 (str): The character to replace the bracket [
      r2 (str): The character to replace the bracket ]

    Returns:
      A string with replaced brackets [ and ]
    """
    return line.replace('[', r1).replace(']', r2).replace("'", "")

  def printHalfDegreesTable(self, graph: dict) -> list:
    """
    The method that prints the calculated table of outdegrees
    and indegrees using adjacency lists representing the given graph

    Args:
      graph (dict): Adjacency lists defined with a dictionary
      Uses class attributes: self.fullName, self.size and self.hd,

    Returns:
      A list of strings representing the calculated outdegrees and indegrees table with the specified adjacency lists
    """
    output = list()
    output.append(f"Graph {self.fullName} | Size: {self.size}")
    for index, vertex in enumerate(graph):
      output.append(f"{vertex} | " +
                    self.stringFormat(f"{self.hd[index]} | ", '(', ')') +
                    self.stringFormat(f"{graph[vertex]}", '{', '}'))
    return output

  def buildGraphImage(self, graph) -> str:
    """
    A method that generates a graphical representation of a given graph

    Args:
      graph (dict): Adjacency lists defined with a dictionary
      Uses class attributes: self.fullName

    Returns:
      Full file name of the image
    """
    g = gviz.Digraph('graph', engine='neato',
                     graph_attr={'splines': 'true', 'overlap': 'false',
                                 'sep': str(1.5), 'normalize': 'true',
                                 'label': f"{self.fullName}",
                                 'fontsize': str(20)},
                     node_attr={'shape': 'circle', 'fontsize': str(20)})
    for vertex in graph:
      g.node(vertex, vertex)
      for nextVertex in graph[vertex]:
        g.edge(vertex, nextVertex)
    g.render(filename=f"{self.fullName}", format="png", directory="pngs")
    return f"{self.fullName}.png"


class PDFCreator:
  """
  Class that controls the creation of a PDF report

  Args:
    fileName (str): Name of the report

  Attributes:
    __document (Document): An object containing all pages and information on them
    __pages (list): All pages contained in the PDF document
    __layouts (list): Content of each page
    __fileName (str): Edited (full) name of the report
  """
  def __init__(self, fileName: str):
    self.__document = Document()
    self.__pages = list()
    self.__layouts = list()
    self.__fileName = fileName if fileName[-4:] == ".pdf" else fileName + ".pdf"

  def clear(self) -> None:
    """
    Deleting information related to the analysis
    """
    self.__document = Document()
    self.__pages = self.__pages[:2]
    self.__layouts = self.__layouts[:2]
    self.__document.add_page(self.__pages[0])
    self.__document.add_page(self.__pages[1])

  def addPage(self) -> None:
    """
    Add a new page to the document
    """
    page = Page()
    self.__document.add_page(page)
    self.__pages.append(page)
    layout = SingleColumnLayout(page)
    self.__layouts.append(layout)

  def addImage(self, path: str) -> None:
    """
    Add an image to a new page

    Args:
      path (str): Path (file name) of the image.
    """
    self.addPage()
    img = ImgReader.open(f"pngs/{path}")
    (width, height) = img.size
    if height > 660:
      width = int(width * 660 / height)
      height = 660
    if width > 470:
      height = int(height * 470 / width)
      width = 470
    self.__layouts[-1].add(Image(Path(f"pngs/{path}"), width=Decimal(width), height=Decimal(height)))

  def addLine(self, line: str) -> None:
    """
    Add a string to the last created page

    Args:
      line (str): The string to add
    """
    self.__layouts[-1].add(Paragraph(line, font="Courier"))

  def addMultiLine(self, text: list) -> None:
    """
    Add strings to the last created page

    Args:
      text (list): Strings to add
    """
    for line in text:
      self.__layouts[-1].add(Paragraph(line, font="Courier"))

  def addSubstitutions(self, substitutions: list) -> None:
    """
    Add substitutions in the form of a table to the last created page

    Args:
      substitutions (list): List of substitutions that have been found
    """
    for sub in substitutions:
      self.__layouts[-1].add(Paragraph(sub[0], font="Courier"))
      numCol = len(sub[1])
      numRow = 2 * ceil(numCol / 13)
      table = FixedColumnWidthTable(number_of_columns=13, number_of_rows=numRow)
      for col in range(0, numCol - 1, 13):
        for pos in range(col, col + 13, 1):
          if pos < numCol:
            table.add(Paragraph(sub[1][pos], font="Courier"))
          else:
            table.add(Paragraph('', font="Courier"))
        for pos in range(col, col + 13, 1):
          if pos < numCol:
            table.add(Paragraph(sub[2][pos], font="Courier"))
          else:
            table.add(Paragraph('', font="Courier"))
      table.set_padding_on_all_cells(Decimal(2), Decimal(2),
                                     Decimal(2), Decimal(2))
      self.__layouts[-1].add(table)
      self.__layouts[-1].add(Paragraph('', font="Courier"))

  def saveFile(self) -> None:
    """
    Save the generated report to a file with the .pdf extension
    """
    with open(Path(self.__fileName), "wb") as pdfFileHandler:
     PDF.dumps(pdfFileHandler, self.__document)


class Analysis:
  """
  An analysis class that implements all the steps of the isomorphic embedding analysis algorithm

  Args:
    graph1 (dict): A graph defined using adjacency lists
    graph2 (dict): Name of the graph (Symbol by which it is denoted)
    fileName (str): Name of the vertex set (Symbol by which it is denoted)

  Attributes:
    __graph1 (dict): Graph representation using a numpy array
    __graph2 (dict): Graph representation using a numpy array
    __fileName (str): Graph dimensionality
    __todoPDF (list): Full name of the graph (including the set of vertices and edges)
    completeSubs (list): The calculated table of semi-powers, which is specified using a numpy array
    analysisResult (int):
    analysisResult (float):
  """

  def __init__(self, graph1: dict, graph2: dict, fileName: str = None):
    self.__graph1: Graph = Graph(graph1, "G", "F", "E")
    self.__graph2: Graph = Graph(graph2, "H", "P", "L")
    self.__todoPDF: bool = False if fileName is None else True

    self.completeSubs: list = []
    self.analysisResult: int = 0
    self.analysisTime: float = 0

    if self.__todoPDF:
      self.__report = PDFCreator(fileName)

      self.__report.addImage(self.__graph1.buildGraphImage(graph1))
      self.__report.addMultiLine(self.__graph1.printHalfDegreesTable(graph1))
      self.__report.addImage(self.__graph2.buildGraphImage(graph2))
      self.__report.addMultiLine(self.__graph2.printHalfDegreesTable(graph2))

      self.__output: list = []

  def __clear(self) -> None:
    """
    Resetting attributes before performing a new analysis
    """
    self.completeSubs = []
    self.analysisResult = 0
    self.analysisTime = 0

    if self.__todoPDF:
      self.__report.clear()
      self.__output: list = []

  def __makePDF(self) -> None:
    """
    Generating a report in PDF format with the results of the analysis
    """
    print("Creating a report...")

    self.__report.addPage()
    self.__report.addMultiLine(self.__output[0][0])
    if len(self.__output[0]) > 1:
      self.__report.addSubstitutions(self.__output[0][1])

    if len(self.__output) > 1:
      self.__report.addPage()
      self.__report.addMultiLine(self.__output[1][0])
      if len(self.__output[1]) > 1:
        self.__report.addSubstitutions(self.__output[0][1])

    self.__report.addLine(f"Execution time: {self.analysisTime}")

    self.__report.saveFile()

    print("The report has been created!")

  def __findCombCondA(self, variants: dict, current: dict) -> bool:
    """
    Function that checks the possibility of finding a combination (one)
    without repeating vertices that satisfies condition A (For each vertex of a subgraph, there exists a vertex of the
    supergraph whose outdegree and indegree are not less than the outdegree and indegree of the vertex of the subgraph)

    Args:
      variants (dict): Variants of graph vertex mappings
      current (dict): Current substitution

    Returns:
      True if there is a mapping that satisfies condition A, otherwise False
    """
    if len(current) == len(variants):
      return True

    for i in variants:
      if i not in current.keys():
        for j in variants[i]:
          if j not in current.values():
            current[i] = j
            if self.__findCombCondA(variants, current):
              return True
            current.pop(i)
        break
    return False

  def __checkCondA(self) -> bool:
    """
    Verification of condition A (For each vertex of a subgraph, there exists a vertex of the supergraph whose
    outdegree and indegree are not less than the outdegree and indegree of the vertex of the subgraph)

    Returns:
       True if there is a substitution that satisfies condition A, otherwise False
    """
    possibleVertices = {}
    for vertexG1 in range(self.__graph1.size):
      possibleVertices[vertexG1] = np.uint32(list(filter(lambda vertexG2:
                                                         self.__graph1.hd[vertexG1][0] <= self.__graph2.hd[vertexG2][
                                                           0] and
                                                         self.__graph1.hd[vertexG1][1] <= self.__graph2.hd[vertexG2][1],
                                                         range(self.__graph2.size))))
      if possibleVertices[vertexG1].size == 0:
        return False
    return self.__findCombCondA(possibleVertices, {})

  def __conditionB(self, sub):
    """
    Verification of (partial) substitution of condition B (Preserving subgraph edges when nesting in a supergraph)

    Args:
      sub (dict): Substitution to be checked

    Returns:
      True if condition B is satisfied, otherwise False
    """
    for vertex in sub.keys():
      for nextVertex in np.argwhere(self.__graph1.graph[vertex] == 1)[:, 0]:
        if nextVertex in sub.keys() and sub[nextVertex] not in np.argwhere(self.__graph2.graph[sub[vertex]] == 1)[:, 0]:
          return False
    return True

  def __findCombCondB(self, variants: dict, res: list, current: dict) -> None:
    """
    Finding all combinations without repetition that satisfy condition B

    Args:
      variants (dict): Variants of graph vertex mappings
      res (list): The resulting substitutions that satisfy the condition B
      current (dict): The current substitution, which is found recursively
    """
    if len(current) == len(variants) and self.__conditionB(current):
      res.append(current.copy())
      return

    for i in variants:
      if i not in current.keys():
        for j in variants[i]:
          if j not in current.values():
            current[i] = j
            if self.__conditionB(current):
              self.__findCombCondB(variants, res, current)
            current.pop(i)
        break

  def __makeMaxSub(self) -> dict:
    """
    Mapping of vertices with the largest output half-degree according to condition A

    Returns:
      Dictionary with possible mappings for the vertices with highest outdegree
    """
    hdG1SortOut = self.__graph1.hd[np.argsort(self.__graph1.hd[:, 0])]
    keys = np.uint32(list(filter(lambda position:
                                 position[1] == 0,
                                 np.argwhere(self.__graph1.hd == hdG1SortOut[-1][0]))))[:, 0]
    maxSubs = {}
    for vertex in keys:
      maxSubs[vertex] = np.uint32(np.argwhere((self.__graph2.hd[:, 0] >= self.__graph1.hd[vertex, 0]) &
                                              (self.__graph2.hd[:, 1] >= self.__graph1.hd[vertex, 1])))[:, 0]
    return maxSubs

  def __makeMaxVariants(self, maxSubs: dict) -> list:
    """
    Formation of lists of possible mappings for initial vertices

    Args:
      maxSubs (dict): Possible mappings for the vertices with highest outdegree

    Returns:
      Possible mappings for other vertices relative to the mappings of vertices with the highest outdegree
    """
    maxVariants = list()
    for maxVertex in maxSubs:
      for maxSub in maxSubs[maxVertex]:
        partSub = {maxVertex: np.uint32([maxSub])}
        otherVertices = np.uint32(np.argwhere(self.__graph1.graph[maxVertex] == 1)[:, 0])
        for nextVertex in otherVertices:
          if nextVertex != maxVertex:
            possibleSubs = np.uint32(list(filter(lambda vertexG2:
                                                 self.__graph1.hd[nextVertex][0] <= self.__graph2.hd[vertexG2][0] and
                                                 self.__graph1.hd[nextVertex][1] <= self.__graph2.hd[vertexG2][1] and
                                                 vertexG2 not in partSub[maxVertex],
                                                 np.argwhere(self.__graph2.graph[maxSub] == 1)[:, 0])))
            if possibleSubs.size == 0:
              break
            partSub[nextVertex] = possibleSubs
            if nextVertex == otherVertices[-1]:
              maxVariants.append(partSub)
    return maxVariants

  def __makePartialSubs(self, maxVariants: list) -> list:
    """
    Finding partial substitutions that satisfy condition B

    Args:
      maxVariants (list): Possible mappings for vertices that can be reached from the vertex with the highest outdegree

    Returns:
      List of the partial substitutions that have been found
    """
    partSubs = list()
    for variant in maxVariants:
      self.__findCombCondB(variant, partSubs, {})
    return partSubs

  def __makePartialVariants(self, partialSubs: list) -> list:
    """
    Generate lists of possible mappings for the remaining vertices

    Args:
      partialSubs (list): Partial substitutions that have been found

    Returns:
      List of the possible mappings for vertices that are not included in the found partial substitutions
    """
    partialVariants = list()
    for sub in partialSubs:
      keysToCheck = np.uint32(list(sub.keys()))
      valuesToCheck = np.uint32(list(sub.values()))
      subVariant = dict(map(lambda pair: (pair[0], np.uint32([pair[1]])), sub.items()))
      for nextVertex in np.delete(np.uint32(range(self.__graph1.size)), keysToCheck):
        verticesWithTransition = np.uint32(np.intersect1d(np.argwhere(self.__graph1.graph[:, nextVertex] == 1)[:, 0],
                                                          keysToCheck))
        for vertexTo in verticesWithTransition:
          variants = np.setdiff1d(np.uint32(list(filter(lambda vertex:
                                                        self.__graph2.hd[vertex, 0] >= self.__graph1.hd[
                                                          nextVertex, 0] and
                                                        self.__graph2.hd[vertex, 1] >= self.__graph1.hd[
                                                          nextVertex, 1],
                                                        np.argwhere(self.__graph2.graph[sub[vertexTo]] == 1)[:, 0]))),
                                  valuesToCheck)
        if variants.size == 0:
          variants = np.setdiff1d(np.argwhere((self.__graph2.hd[:, 0] >= self.__graph1.hd[nextVertex, 0]) &
                                              (self.__graph2.hd[:, 1] >= self.__graph1.hd[nextVertex, 1]))[:, 0],
                                  valuesToCheck)

        if variants.size == 0:
          break
        subVariant[nextVertex] = np.uint32(variants)
      if len(subVariant.keys()) == self.__graph1.size:
        partialVariants.append(subVariant)
    return partialVariants

  def __makeCompleteSubs(self, partialVariants: list) -> list:
    """
    Finding the resulting substitutions that satisfy conditions A and B

    Args:
      partialVariants (list): Possible mappings for vertices that are not included in the found partial substitutions

    Returns:
      List of resulting substitutions that satisfy conditions A and B
    """
    completeSubs = []
    for variant in partialVariants:
      self.__findCombCondB(variant, completeSubs, {})
    return completeSubs

  def __algorithm(self) -> int:
    """
    Step-by-step implementation of the algorithm for analyzing the isomorphic embedding of two directed graphs

    Returns:
      The result of the analysis
      (0 - graphs are isomorphically nested, 1, 2 or 3 - graphs are not isomorphically nested)
    """
    if self.__todoPDF:
      output = [[]]
      output[0].append(f"""Checking the isomorphic embedding of the graph {self.__graph1.fullName} 
      into the graph {self.__graph2.fullName}:""")
    if self.__checkCondA():
      maxSubs = self.__makeMaxSub()
      maxVariants = self.__makeMaxVariants(maxSubs)
      partialSubs = self.__makePartialSubs(maxVariants)
      if len(partialSubs) > 0:
        partialVariants = self.__makePartialVariants(partialSubs)
        completeSubs = self.__makeCompleteSubs(partialVariants)
        if len(completeSubs) > 0:
          for sub in completeSubs:
            translatedSub = {}
            for vertex in sub:
              translatedSub[self.__graph1.transition[vertex]] = self.__graph2.transition[sub[vertex]]
            self.completeSubs.append(translatedSub)
          if self.__todoPDF:
            output[0].append(f"""The graph {self.__graph1.fullName} is isomorphically embedded 
            in the graph {self.__graph2.fullName}""")
            output[0].append("The resulting substitutions that satisfy condition B:")
            output.append(list())
            for index, sub in enumerate(self.completeSubs):
              output[1].append([f"Substitution {index + 1}:", list(sub.keys()), list(sub.values())])
            self.__output.append(output)
          return 0
        else:
          if self.__todoPDF:
            output[0].append(f"""When trying to complete all possible substitutions, no option was found that 
            fulfilled the conditions A and B, therefore the graph {self.__graph1.fullName} is NOT 
            isomorphically embedded in the graph {self.__graph2.fullName}""")
            self.__output.append(output)
          return 3
      else:
        if self.__todoPDF:
          output[0].append(f"""No partial substitutions were found that satisfy condition B. 
          The graph {self.__graph1.fullName} is NOT isomorphically embedded in the graph {self.__graph2.fullName}.""")
          self.__output.append(output)
        return 2
    else:
      if self.__todoPDF:
        output[0].append(f"""The given graphs do not satisfy  condition A. 
        The graph {self.__graph1.fullName} is NOT isomorphically embedded in the graph {self.__graph2.fullName}""")
        self.__output.append(output)
      return 1

  def makeAnalysis(self):
    """
    Function that controls the analysis and returns its result.

    Returns:
      The result of the analysis
      (0 - graphs are isomorphically nested; 1, 2 and 3 - graphs are not isomorphically nested)
    """
    self.__clear()
    startTime = time.time()
    isomorphicCheck = False
    if self.__graph1.size > self.__graph2.size:
      self.__graph1, self.__graph2 = self.__graph2, self.__graph1
    elif self.__graph1.size == self.__graph2.size:
      isomorphicCheck = True

    if isomorphicCheck:
      self.analysisResult = self.__algorithm()
      self.__graph1, self.__graph2 = self.__graph2, self.__graph1
      self.analysisResult += self.__algorithm()
      self.__graph1, self.__graph2 = self.__graph2, self.__graph1

      if self.analysisResult == 0 and self.__todoPDF:
        self.__output.append([f"The graphs {self.__graph1.fullName} and {self.__graph2.fullName} are isomorphic"])
    else:
      self.analysisResult = self.__algorithm()

    endTime = time.time()
    self.analysisTime = endTime - startTime

    if self.__todoPDF:
      self.__makePDF()

    return self.analysisResult


if __name__ == '__main__':
  vartest = {
    "GAL1": {'y1': ['y6', 'y8', 'y4', 'y2'], 'y2': ['y6'], 'y3': ['y1', 'y2', 'y6', 'y5', 'y3', 'y4'],
             'y4': ['y6', 'y8', 'y2', 'y1', 'y5'], 'y5': ['y8'], 'y6': ['y3', 'y7', 'y5', 'y4'],
             'y7': ['y3', 'y8', 'y2'], 'y8': ['y6', 'y8', 'y2']},
    "GAL2": {'x1': ['x4', 'x2'], 'x2': ['x5'], 'x3': ['x1', 'x5', 'x3', 'x4'], 'x4': ['x5', 'x1'], 'x5': ['x4']}
  }
  # testGraph = {
  #   "x1": ["x2"],
  #   "x2": ["x1", "x4"],
  #   "x3": ["x2", "x3", "x5"],
  #   "x4": ["x1"],
  #   "x5": ["x2", "x5"]
  # }
  # g1 = Graph(testGraph, "G", "V", "E")

  testAn = Analysis(vartest["GAL1"], vartest["GAL2"], "test")

  testAn.makeAnalysis()

  print()
