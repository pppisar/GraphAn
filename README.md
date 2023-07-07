# GraphAn
Final version of application

before running the application you need to:
### 1) Install the libraries:
```
pip install graphviz
pip install borb
```

### 2) Import the libraries
>For GraphAn.py code:
```python
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
```

>For generator.py code:
```python
from random import randint
from math import comb, ceil

from GraphAn import Graph, Analysis
```