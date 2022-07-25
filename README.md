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

---

## Qfrom

### Methods
 - import_list
 - import_dict
 - (import_set)
 - (import_array)
 - (import_mtx)
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

---

## func

### Methods
 - __ call __(func, in: int, out: int) -> verpackt func in lambda, so dass lambda-parameter in-count entsprechen und output tuple out-count entspricht.
 - vec(func) -> vectorize func, autodetect in and out counts
 - vec(func, in: int, out: int)
 - multicol(repetitioncount: int)

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

 ---

## plot

### Methods
 - plot
 - bar
 - hist
 - box
 - scatter

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

---

## trans

### Methods
 - (shuffle)