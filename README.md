# Qfrom_slim
Qfrom provides a unified and simple query language for sets of data.
This Project is based on Python 3.10.0

---

# Qfrom Documentation

## Contents
- [Qfrom](#qfrom)
- [col](#col)
- [func](#func)
- [agg](#agg)
- [plot](#plot)
- [out](#out)
- [trans](#trans)
- [Performance Tests](#performance-tests)
---

## Qfrom

import Qfrom like this

```python
from QfromPackage.Qfrom_slim import Qfrom
```

### Methods
- [import list](#import-list)
- [import dict](#import-dict)
- [import set](#import-set)
- [import array](#import-array)
- [import matrix](#import-mtx)
- [import DataFrame](#import-dataframe)
- [import csv](#import-csv)
- [import json](#import-json)
- [import generator](#import-generator)
* [eq](#eq)
* [str](#str)
* [repr](#repr)
* [append](#append)
* [setitem](#setitem)
  * [set row](#set-row)
  * [set column](#set-column)
  * [set cell](#set-cell)
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
> y
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
> y1 y2
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
> y
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
> y
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
> y1    y2
> 1 4
> 2 5
> 3 6
```

### import DataFrame
```python
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
Qfrom(df)
```
```
> Qfrom
> a	b
> 1	4
> 2	5
> 3	6
```

### import csv
```python
csv = '''
a,b
1,4
2,5
3,6
'''

Qfrom(csv)
```
```
> Qfrom
> a	b
> 1	4
> 2	5
> 3	6
```

### import json
```python
json = "{'a': [1, 2, 3], 'b': [4, 5, 6]}"
Qfrom(json)
```
```
> Qfrom
> a	b
> 1	4
> 2	5
> 3	6
```

### import generator
```python
Qfrom(range(3))
```
```
> Qfrom
> y
> 0
> 1
> 2
```

### eq
```python
q1 = Qfrom({
    'a': [1, 2, 3],
    'b': [4, 5, 6]
})
q2 = Qfrom([
    {'a': 1, 'b': 4},
    {'a': 2, 'b': 5},
    {'a': 3, 'b': 6}
])

q1 == q2
```
```
> True
```

### str
```python
q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
str(q)
```
```
> 'Qfrom\na\tb\n1\t4\n2\t5\n3\t6'
```

### repr
```python
q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
print(q)
```
```
> Qfrom
> a	b
> 1	4
> 2	5
> 3	6
```

### append
```python
q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})

q.append((4, 7))
q
```
```
> Qfrom
> a	b
> 1	4
> 2	5
> 3	6
> 4 7
```

```python
q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})

q.append({'a': 4, 'b':7})
q
```
```
> Qfrom
> a	b
> 1	4
> 2	5
> 3	6
> 4 7
```

### setitem

#### set row
```python
q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})

q[1] = (7, 8)
q
```
```
> Qfrom
> a	b
> 1	4
> 7 8
> 3	6
```

```python
q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})

q[1] = {'a': 7, 'b': 8}
q
```
```
> Qfrom
> a	b
> 1	4
> 7 8
> 3	6
```

#### set column

set single column
```python
q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})

q['a'] = [7, 8, 9]
q
```
```
> Qfrom
> a	b
> 7	4
> 8 5
> 9	6
```

add new column
```python
q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})

q['c'] = [7, 8, 9]
q
```
```
> Qfrom
> a	b   c
> 1	4   7
> 2 5   8
> 3	6   9
```

set multiple columns
```python
q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})

q['a, b'] = [(4, 1), (5, 2), (6, 3)]
q
```
```
> Qfrom
> a	b
> 4	1
> 5 2
> 6	3
```

the order of the key-value-pairs in the dictionary does not matter.
```python
q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})

q['a, b'] = {'b': [1, 2, 3], 'a': [4, 5, 6]}
q
```
```
> Qfrom
> a	b
> 4	1
> 5 2
> 6	3
```


#### set cell

```python
q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})

q['a', 1] = 7
q
```
```
> Qfrom
> a	b
> 1	4
> 7 5
> 3	6
```

```python
q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})

q[1] = {'a': 7}
q
```
```
> Qfrom
> a	b
> 1	4
> 7 5
> 3	6
```

[Contents](#contents)

---

## col

import col like this

```python
from QfromPackage.Qfrom_slim import col
```

### Methods
- 1 -> 1
  - pass_none
  - normalize
  - abs
  - center -> set a new origin for a column: [1, 2, 3], origin=2 -> [-1, 0, 1]
  - shift(steps=...)
  - not
  - id
- n -> 1
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
- 1 -> n
  - copy(n)
  - flatten -> autodetect out count
- n -> m
  - ml_models


[Contents](#contents)

---

## func

import func like this

```python
from QfromPackage.Qfrom_slim import func
```

### Methods
- __ call __(func, in: int, out: int) -> verpackt func in lambda, so dass lambda-parameter in-count entsprechen und output tuple out-count entspricht.
- vec(func) -> vectorize func, autodetect in and out counts
- vec(func, in: int, out: int)
- multicol(repetitioncount: int)


[Contents](#contents)

---

## agg

import agg like this

```python
from QfromPackage.Qfrom_slim import agg
```

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

import plot like this

```python
from QfromPackage.Qfrom_slim import plot
```

### Methods
- plot
- bar
- hist
- box
- scatter


[Contents](#contents)

---

## out

import out like this

```python
from QfromPackage.Qfrom_slim import out
```

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

import trans like this

```python
from QfromPackage.Qfrom_slim import trans
```

### Methods
- (shuffle)

[Contents](#contents)

---

## Performance Tests


[Contents](#contents)