# Qfrom_slim
Qfrom provides a unified and simple query language for sets of data.
This Project is based on Python 3.10.0

---

# Qfrom Documentation

## Contents
 - [Imports](#imports)
 - [Qfrom](#qfrom)
 - [col](#col)
 - [func](#func)
 - [agg](#agg)
 - [plot](#plot)
 - [out](#out)
 - [trans](#trans)

---

## Imports
 - [Qfrom](#qfrom-imports)
 - [3d party](#3d-party-imports)

### Qfrom imports
```python
from QfromPackage.Qfrom_slim import col, func, agg, plot, out, trans, Qfrom
```


### 3d party imports
```python
import numpy as np
import pandas as pd
```

[Contents](#contents)

---

## Qfrom

### Methods
 - [import list](#import-list)
 - [import dict](#import-dict)
 - [import set](#import-set)
 - [import array](#import-array)
 - [import matrix](#import-mtx)
 - import_dataframe
 - import_csv
 - (import_json)
 - import_generator
 * eq
 * str
 * repr
 * append
 * setitem -> more dim slice support
 * getitem
 * contains
 * iter
 * len
 - keys
 - values
 - items
 - (stats)
 * remove(selection: str|tuple[str]|list[str])
 * rename(map: dict[str, str])
 * select(selection: str|tuple[str]|list[str])
 * map(args: str|tuple[str]|list[str], func: callable, out=str|tuple[str]|list[str])
 * orderby(selection: str|tuple[str]|list[str], func: callable, reverse: bool)
 - where(selection: str|tuple[str]|list[str], predicate: callable)
 - groupby(selection: str|tuple[str]|list[str], func: callable)
 - flatten
 - unique
 - value_counts
 * agg
 - join
 - join_cross
 - join_outer
 - join_outer_left
 - join_outer_right
 - join_id
 - join_id_outer
 - join_id_outer_left
 - join_id_outer_right
 * concat
 * concat_outer
 * concat_outer_left
 * concat_outer_right
 - calc
 - call


### import list
```python
l = [1, 2, 3]
Qfrom(l)
```
```
> Qfrom
> 0
> 1
> 2
> 3
```
```python
l = [(1, 4), (2, 5), (3, 6)]
Qfrom(l)
```
```
> Qfrom
> 0 1
> 1	4
> 2	5
> 3	6
```
```python
l = [{'a': 1, 'b': 4}, {'a': 2, 'b': 5}, {'a': 3, 'b': 6}]
Qfrom(l)
```
```
> Qfrom
> a	b
> 1	4
> 2	5
> 3	6
```

### import dict
```python
d = {'a': [1, 2, 3], 'b': [4, 5, 6]}
Qfrom(d)
```
```
> Qfrom
> a	b
> 1	4
> 2	5
> 3	6
```

### import set
```python
s = {1, 2, 3}
Qfrom(s)
```
```
> Qfrom
> 0
> 1
> 2
> 3
```

### import array
```python
a = np.array([1, 2, 3])
Qfrom(a)
```
```
> Qfrom
> 0
> 1
> 2
> 3
```

### import matrix
```python
mtx = np.array([[1, 4], [2, 5], [3, 6]])
Qfrom(mtx)
```
```
> Qfrom
> 0	1
> 1	4
> 2	5
> 3	6
```

[Contents](#contents)

---

## col

### Methods

1 -> 1
 - pass_none
 - normalize
 - abs
 - center -> set a new origin for a column: [1, 2, 3], origin=2 -> [-1, 0, 1]
 - shift(steps=...)
 - not
 - id

n -> 1
 - any
 - all
 - min
 - min_colname
 - max
 - max_colname
 - sum
 - mean
 - median
 - var
 - eq
 - agg(colcount) -> combines multible cols to one 2d col
 - state(rules: func|dict[func]) -> iterates over col. for each item the result state of the last item is feed in. 
 - lod_and
 - lod_or
 - lod_xor

1 -> n
 - copy(n)
 - flatten -> autodetect out count

n -> m
 - ml_models


[Contents](#contents)

---

## func

### Methods
 - __ call __(func, in: int, out: int) -> verpackt func in lambda, so dass lambda-parameter in-count entsprechen und output tuple out-count entspricht.
 - vec(func) -> vectorize func, autodetect in and out counts
 - vec(func, in: int, out: int)
 - multicol(repetitioncount: int)


[Contents](#contents)

---

## agg

### Methods
 - any
 - all
 - min
 - min_id
 - max
 - max_id
 - sum
 - mean
 - median
 - var
 - len
 - size
 - state(rules: func|dict[func]) -> returns the last state of col.state


[Contents](#contents)

 ---

## plot

### Methods
 - plot
 - bar
 - hist
 - box
 - scatter


[Contents](#contents)

---

## out

### Methods
 - tolist
 - (toset)
 - todict
 - toarray
 - (tomtx)
 - todf
 - tocsv
 - (tocsvfile)
 - (tojson)
 - (tojsonfile)


[Contents](#contents)

---

## trans

### Methods
 - (shuffle)

[Contents](#contents)
