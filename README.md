![GraphAn](/images/Logo.png)

---

`GraphAn` is an application for analyzing the isomorphic embedding of two oriented graphs.

## 0. About GraphAn

GraphAn is a pure python application (in the future, a library) for analyzing the isomorphic embedding of two oriented graphs.

The input graphs are specified using adjacency lists in the form of a dictionary. 
The application changes the representation to matrix and further analysis is performed using the matrix representation in the form of nested numpy arrays.

If, as a result of the analysis, one graph is isomorphically embedded in another, a list of substitutions (bijective mappings) in the form of dictionaries will be obtained.

The application was developed as part of a bachelor's thesis at Taras Shevchenko National University of Kyiv.

## 1. Installing

### Installing graphviz
To create a graphical representation of graphs, the **graphviz** package is used. In addition to the library, you need to install it on your operating system. You can find graphviz for your operating system [here](https://graphviz.org/download/).

### Installing libraries
The application uses the following python libraries:
|Library|Installation|
|-------|------------|
|Numpy|`pip install numpy`|
|graphviz|`pip install graphviz`|
|borb|`pip install borb`|

## 2. How to use

First you need to clone the repository:
```git
git clone https://github.com/pppisar/GraphAn.git
```

After that, you need to import the Analysis and Graph classes into your project
```python
from GraphAn import Graph, Analysis
```

If your project is located in a different directory, then move the GraphAn.py file, or add the cloned repository to your project 
>Example from the file tests/examples.ipynb:
```python
import sys
                # Your path to the cloned repository
sys.path.append('../')

from GraphAn import Graph, Analysis
```

## 3. Examples

...

## 4. Analysis algorithm

### Necessary and sufficient conditions for isomorphic embedding
For an isomorphic embedding of a graph G1 = (V1, E1) into a graph G2 = (V2, E2), the following two conditions must be necessary and simultaneously sufficient:

>Condition A
>- For each vertex in the set V1 of graph G1, there must exist at least one such vertex in the set V2 of graph G2, whose outdegree and indegree are not less than the outdegree and indegree of the vertex in the set V1.

>Condition B
>- There must be at least one bijective mapping 𝑓:V1 → V2', where V2' ⊆ V2, that transforms the graph G1 into the graph G2' = (V2', E2'), where E2' ⊆ E2.

### Algorithm

The analysis algorithm that implements this application was developed by me on the basis of an existing algorithm for recognizing the isomorphic embedding of two oriented graphs.
Let there be two oriented graphs G1 = (V1, E1) and G2 = (V2, E2) such that |V1| ≤ |V2| and |E1| ≤ |E2|. 
The algorithm for analyzing the isomorphic embedding of graph G1 in graph G2 contains the following 7 steps:

1) In the natural order, all the vertices of graphs G1 and G2 with the specified pairs of outdegrees and indegrees are written out.
2) The satisfaction of condition A is checked. If the condition is satisfied, then go to the next step, otherwise to step 7.
3) The vertex v𝑖 ∈ V1 with the maximum outdegree is matched with such vertices w𝑗 ∈ V2 for which condition A is satisfied. If no such vertices exist, then proceed to step 7, otherwise proceed to the next step of the algorithm.
4) Each vertex v𝑚 ∈ E1 (v𝑖) is mapped to a single vertex w𝑛 ∈ E2(w𝑗) so that condition A is satisfied. The result is partial substitutions of the set V1 by V2' ⊆ V2, which are checked to see if condition B is satisfied. If no such substitutions exist, proceed to step 7. Otherwise, proceed to the next step of the algorithm.
5) The remaining partial substitutions are completed for the last vertices of V1 so that condition A is satisfied. The result is the completed substitutions of the set V1 by V2' ⊆ V2, which are checked for the fulfillment of condition B. If there are no substitutions that satisfy both conditions, then we proceed to step 7, otherwise we proceed to the next step.
6) Graph G1 is isomorphically embedded in graph G2. At least one substitution of V1 by V2' ⊆ V2 is found that satisfies conditions A and B.
7) The graph G1 = (V1, E1) does not isomorphically embedded in the graph G2 = (V2, E2). As a result of the analysis, we did not find any substitution(mapping) of the set V1 to V2' ⊆ V2 that fulfills conditions A and B.

If |V1| = |V2|, then the following results of the analysis are possible:
- The graph G1 = (V1, E1) is isomorphically embedded in the graph G2 = (V2, E2);
- The graph G2 = (V2, E2) is isomorphically embedded in the graph G1 = (V1, E1);
- The graphs G1 = (V1, E1) and G2 = (V2, E2) are not isomorphically embedded;
- The graphs G1 = (V1, E1) and G2 = (V2, E2) are mutually isomorphically embedded. In this case, the graphs are isomorphic, and the analysis algorithm must be performed twice: when the graph G1 is isomorphically embedded in G2 and when the graph G2 is isomorphically embedded in G1.
---