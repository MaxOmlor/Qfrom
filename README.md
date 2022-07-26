# Qfrom_slim
Qfrom provides a unified and simple query language for sets of data.
This Project is based on Python 3.10.0

---

# Contents
- [Qfrom_slim](#qfrom_slim)
- [Contents](#contents)
- [class Qfrom](#class-qfrom)
  - [import list](#import-list)
  - [import dict](#import-dict)
  - [import set](#import-set)
  - [import array](#import-array)
  - [import matrix](#import-matrix)
  - [import DataFrame](#import-dataframe)
  - [import csv](#import-csv)
  - [import json](#import-json)
  - [import generator](#import-generator)
  - [eq](#eq)
  - [str](#str)
  - [repr](#repr)
  - [append](#append)
  - [setitem](#setitem)
    - [set row](#set-row)
    - [set column](#set-column)
    - [set cell](#set-cell)
  - [getitem](#getitem)
    - [get row](#get-row)
    - [get column](#get-column)
    - [get cell](#get-cell)
  - [contains](#contains)
  - [iter](#iter)
  - [len](#len)
  - [keys](#keys)
  - [values](#values)
  - [items](#items)
  - [remove](#remove)
  - [rename](#rename)
  - [select](#select)
    - [string](#string)
    - [dynamic column selection](#dynamic-column-selection)
    - [tuple](#tuple)
  - [map](#map)
    - [out not specified](#out-not-specified)
    - [args not specified](#args-not-specified)
    - [args dynamic column selection](#args-dynamic-column-selection)
    - [func not specified respectively copying column](#func-not-specified-respectively-copying-column)
    - [vectorize function](#vectorize-function)
    - [function returning multiple columns](#function-returning-multiple-columns)
    - [function returning a scalar](#function-returning-a-scalar)
    - [function returning a generator](#function-returning-a-generator)
  - [orderby](#orderby)
  - [where](#where)
  - [groupy](#groupy)
  - [flatten](#flatten)
  - [unique](#unique)
  - [value counts](#value-counts)
  - [agg](#agg)
  - [join](#join)
  - [join cross](#join-cross)
  - [join outer](#join-outer)
  - [join outer left](#join-outer-left)
  - [join outer right](#join-outer-right)
  - [join id](#join-id)
  - [join id outer](#join-id-outer)
  - [join id outer left](#join-id-outer-left)
  - [join id outer right](#join-id-outer-right)
  - [concat](#concat)
  - [concat outer](#concat-outer)
  - [concat outer left](#concat-outer-left)
  - [concat outer right](#concat-outer-right)
  - [calculate](#calculate)
  - [call](#call)
- [class col](#class-col)
  - [0 -> 1 functions](#0---1-functions)
    - [id](#id)
  - [1 -> 1 functions](#1---1-functions)
    - [normalize](#normalize)
    - [abs](#abs)
    - [shift](#shift)
    - [not](#not)
  - [n -> 1 functions](#n---1-functions)
    - [any](#any)
    - [all](#all)
    - [min](#min)
    - [min_colname](#min_colname)
    - [max](#max)
    - [max_colname](#max_colname)
    - [sum](#sum)
    - [mean](#mean)
    - [median](#median)
    - [var](#var)
    - [eq](#eq-1)
    - [agg](#agg-1)
    - [state](#state)
    - [lod_and](#lod_and)
    - [lod_or](#lod_or)
    - [lod_xor](#lod_xor)
  - [1 -> n functions](#1---n-functions)
    - [copy](#copy)
    - [flatten](#flatten-1)
  - [n -> m functions](#n---m-functions)
    - [ml_models](#ml_models)
- [class func](#class-func)
  - [vec](#vec)
  - [multicol](#multicol)
- [class agg](#class-agg)
  - [any](#any-1)
  - [all](#all-1)
  - [min](#min-1)
  - [min_id](#min_id)
  - [max](#max-1)
  - [max_id](#max_id)
  - [sum](#sum-1)
  - [mean](#mean-1)
  - [median](#median-1)
  - [var](#var-1)
  - [len](#len-1)
  - [size](#size)
  - [state](#state-1)
- [class plot](#class-plot)
  - [plot](#plot)
  - [bar](#bar)
  - [hist](#hist)
  - [box](#box)
  - [scatter](#scatter)
- [class out](#class-out)
  - [list](#list)
  - [set](#set)
  - [dict](#dict)
  - [array](#array)
  - [mtx](#mtx)
  - [df](#df)
  - [csv](#csv)
  - [csv file](#csv-file)
  - [json](#json)
  - [json file](#json-file)
- [class trans](#class-trans)
  - [shuffle](#shuffle)
- [Performance Tests](#performance-tests)
  - [setup](#setup)
  - [runtime tests](#runtime-tests)
    - [append](#append-1)
    - [getitem](#getitem-1)
    - [iter](#iter-1)
    - [select](#select-1)
    - [map](#map-1)
    - [orderby](#orderby-1)
    - [where](#where-1)
    - [groupby](#groupby)
    - [agg](#agg-2)
---

# class Qfrom

import Qfrom like this

```python
from QfromPackage.Qfrom_slim import Qfrom
```

## import list
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
> y0 y1
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

[Contents](#contents)

---

## import dict
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

[Contents](#contents)

---

## import set
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

[Contents](#contents)

---

## import array
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

[Contents](#contents)

---

## import matrix
```python
mtx = np.array([[1, 4], [2, 5], [3, 6]])
Qfrom(mtx)
```
```
> Qfrom
> y0    y1
> 1 4
> 2 5
> 3 6
```

[Contents](#contents)

---

## import DataFrame
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

[Contents](#contents)

---

## import csv
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

[Contents](#contents)

---

## import json
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

[Contents](#contents)

---

## import generator
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

[Contents](#contents)

---

## eq
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

[Contents](#contents)

---

## str
```python
q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
str(q)
```
```
> 'Qfrom\na\tb\n1\t4\n2\t5\n3\t6'
```

[Contents](#contents)

---

## repr
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

[Contents](#contents)

---

## append
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

[Performance test](#append-1)

[Contents](#contents)

---

## setitem

### set row
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

q[1] = 7, 8
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

### set column

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


### set cell

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

## getitem

[Performance test](#getitem-1)

### get row
```python
q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})

print(q[1])

a, b = q[1]
print(f'{a=}, {b=}')
```
```
> (2, 5)
> a=2, b=5
```

it is posssible to use slice notation. the result is returned as a new Qfrom.
```python
q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
q[1:]
```
```
> Qfrom
> a	b
> 2	5
> 3	6
```

### get column
```python
q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
q['a']
```
```
> Qfrom
> a
> 1
> 2
> 3
```

select multiple columns
```python
q = Qfrom({
    'a': [1, 2, 3],
    'b': [4, 5, 6],
    'c': [7, 8, 9],
    'd': [10, 11, 12],
    'e': [13, 14, 15],
    })
q['a,c']
```
```
> Qfrom
> a c
> 1 7
> 2 8
> 3 9
```

it is possible to use dynamic column selection. More information in section [dynamic column selection](#dynamic-column-selection)
```python
q = Qfrom({
    'a': [1, 2, 3],
    'b': [4, 5, 6],
    'c': [7, 8, 9],
    'd': [10, 11, 12],
    'e': [13, 14, 15],
    })
q['...,c']
```
```
> Qfrom
> a b   c
> 1 4   7
> 2 5   8
> 3 6   9
```


### get cell
```python
q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
q['a', 0]
```
```
> 1
```

[Contents](#contents)

---

## contains
```python
q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
(2, 5) in q
```
```
> True
```

the order of the key-value-pairs in the dictionary does not matter.
```python
q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
{'b':5, 'a': 2} in q
```
```
> True
```

[Contents](#contents)

---

## iter
```python
q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})

for a, b in q:
    print(a, b)
```
```
> 1 4
> 2 5
> 3 6
```

[Performance test](#iter-1)

[Contents](#contents)

---

## len
```python
q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
len(q)
```
```
> 3
```

[Contents](#contents)

---

## keys
```python
q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})

for k in q.keys():
    print(k)
```
```
> a
> b
```

[Contents](#contents)

---

## values
```python
q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})

for v in q.values():
    print(v)
```
```
> [1 2 3]
> [4 5 6]
```

[Contents](#contents)

---

## items
```python
q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})

for v in q.values():
    print(v)
```
```
> a [1 2 3]
> b [4 5 6]
```

[Contents](#contents)

---

## remove
```python
q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
q.remove('a')
```
```
> Qfrom
> b
> 4
> 5
> 6
```

remove multiple columns
```python
q = Qfrom({
    'a': [1, 2, 3],
    'b': [4, 5, 6],
    'c': [7, 8, 9],
    'd': [10, 11, 12],
    'e': [13, 14, 15],
    })
q.remove('a, c')
```
```
> Qfrom
> b	d	e
> 4	10	13
> 5	11	14
> 6	12	15
```

it is possible to use dynamic column selection. More information in section [dynamic column selection](#dynamic-column-selection)
```python
q = Qfrom({
    'a': [1, 2, 3],
    'b': [4, 5, 6],
    'c': [7, 8, 9],
    'd': [10, 11, 12],
    'e': [13, 14, 15],
    })
q.remove('...,c')
```
```
> Qfrom
> d	e
> 10	13
> 11	14
> 12	15
```

[Contents](#contents)

---

## rename
```python
q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
q.rename({'b': 'c'})
```
```
> Qfrom
> a	c
> 1	4
> 2	5
> 3	6
```

rename multiple columns
```python
q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
q.rename({'b': 'c'})
```
```
> Qfrom
> a	y
> 1	4
> 2	5
> 3	6
```

[Contents](#contents)

---

## select

args
- selection: str|tuple[str]|list[str]
  
    -> determents which columns will be passed to new Qfrom
- return: Qfrom

[Performance test](#select-1)

### string
```python
q = Qfrom({
    'a': [1, 2, 3],
    'b': [4, 5, 6],
    'c': [7, 8, 9],
    'd': [10, 11, 12],
    'e': [13, 14, 15],
    })
q.select('a')
```
```
> Qfrom
> a
> 1
> 2
> 3
```

select multiple columns
```python
q = Qfrom({
    'a': [1, 2, 3],
    'b': [4, 5, 6],
    'c': [7, 8, 9],
    'd': [10, 11, 12],
    'e': [13, 14, 15],
    })
q.select('a, c')
```
```
> Qfrom
> a	c
> 1	7
> 2	8
> 3	9
```

### dynamic column selection

... notation for a slice of the keys
```python
q = Qfrom({
    'a': [1, 2, 3],
    'b': [4, 5, 6],
    'c': [7, 8, 9],
    'd': [10, 11, 12],
    'e': [13, 14, 15],
    })
q.select('...,c')
```
```
> Qfrom
> a	b	c
> 1	4	7
> 2	5	8
> 3	6	9
```

```python
q = Qfrom({
    'a': [1, 2, 3],
    'b': [4, 5, 6],
    'c': [7, 8, 9],
    'd': [10, 11, 12],
    'e': [13, 14, 15],
    })
q.select('b,...,d')
```
```
> Qfrom
> a	b	c	d
> 1	4	7	10
> 2	5	8	11
> 3	6	9	12
```

. will be replaced by next occuring key
```python
q = Qfrom({
    'a': [1, 2, 3],
    'b': [4, 5, 6],
    'c': [7, 8, 9],
    'd': [10, 11, 12],
    'e': [13, 14, 15],
    })
q.select('a,.,c')
```
```
> Qfrom
> a	b	c
> 1	4	7
> 2	5	8
> 3	6	9
```

\* will be replaced by all keys
```python
q = Qfrom({
    'a': [1, 2, 3],
    'b': [4, 5, 6],
    'c': [7, 8, 9],
    'd': [10, 11, 12],
    'e': [13, 14, 15],
    })
q.select('*')
```
```
> Qfrom
> a	b	c	d	e
> 1	4	7	10	13
> 2	5	8	11	14
> 3	6	9	12	15
```

### tuple
```python
q = Qfrom({
    'a': [1, 2, 3],
    'b': [4, 5, 6],
    'c': [7, 8, 9],
    'd': [10, 11, 12],
    'e': [13, 14, 15],
    })
q.select(('a', 'c'))
```
```
> Qfrom
> a	c
> 1	7
> 2	8
> 3	9
```

[Contents](#contents)

---

## map

args
- args: str | tuple[str] | list[str] = None
  
    -> determents which columns will be passed to func
- func: callable = None,

    -> function mappes passed columns to one or more new columns
- out: str | tuple[str] | list[str] = None

    -> names for the output columns
- return: Qfrom

```python
q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
q.map('a,b', lambda x,y: x+y, 'c')
```
```
> Qfrom
> a	b	c
> 1	4	5
> 2	5	7
> 3	6	9
```

[Performance test](#map-1)

### out not specified

if out is not speecified the result will be written into the first column from the specified args
```python
q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
q.map('a,b', lambda x,y: x+y)
```
```
> Qfrom
> a	b
> 5	4
> 7	5
> 9	6
```

### args not specified

if args is not speecified the passed columns will be choosen by ne names of arguments of the given function.
```python
q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
q.map(func=lambda b,a: b+a, out='c')
```
```
> Qfrom
> a	b	c
> 1	4	5
> 2	5	7
> 3	6	9
```

if * notation is used in the args of the given function, all not used columns will be passed to the function.
```python
q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
q.map(func=lambda a, *args: a+sum(args), out='d')
```
```
> Qfrom
> a	b	c	d
> 1	4	7	12
> 2	5	8	15
> 3	6	9	18
```

if ** notation is used  in the args of the given function, all not used columns will be passed as a dict to the function.
```python
q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
q.map(func=lambda a, **kwrgs: kwrgs['c'], out='d')
```
```
> Qfrom
> a	b	c	d
> 1	4	7	7
> 2	5	8	8
> 3	6	9	9
```

### args dynamic column selection

it is possible to use dynamic column selection to specify the parameter args. More information in section [dynamic column selection](#dynamic-column-selection)
```python
q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
q.map('*', lambda x, *args: x+sum(args), out='d')
```
```
> Qfrom
> a	b	c	d
> 1	4	7	12
> 2	5	8	15
> 3	6	9	18
```

### func not specified respectively copying column

if func is not specified map will write the selected columns to the specified out keys
```python
q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
q.map('a, b', out='c, d')
```
```
> Qfrom
> a	b	c	d
> 1	4	1	4
> 2	5	2	5
> 3	6	3	6
```

### vectorize function

by default the columns which will be passed to the function are of type np.ndarray. if the given function is defined vor single element, not for whole columns, the function must first be vectorized. More information in section [vec](#vec)
```python
q = Qfrom({'a': ['ab', 'cd', 'fg']})
q.map('a', func.vec(lambda x: x.upper()))
```
```
> Qfrom
> a
> AB
> CD
> FG
```

### function returning multiple columns

if a function is returning a tuple or a dict of np.ndarray the result will be treated as multible columns.
```python
q = Qfrom({'a': [1, 2, 3]})
q.map('a', lambda x: (x+1, x+2))
```
```
> Qfrom
> a	a0	a1
> 1	2	3
> 2	3	4
> 3	4	5
```

```python
q = Qfrom({'a': [1, 2, 3]})
q.map('a', lambda x: {'b': x+1, 'c': x+2})
```
```
> Qfrom
> a	b	c
> 1	2	3
> 2	3	4
> 3	4	5
```

multible keys can be specified in the out parameter
```python
q = Qfrom({'a': [1, 2, 3]})
q.map('a', lambda x: (x+1, x+2), 'b, c')
```
```
> Qfrom
> a	b	c
> 1	2	3
> 2	3	4
> 3	4	5
```

### function returning a scalar

if the function is returning a scalar insted of a np.ndarray, the scalar will be broadcasted to a np.ndarray of the size of a column
```python
q = Qfrom({'a': [1, 2, 3]})
q.map(func=lambda: 1, out='b')
```
```
> Qfrom
> a	b
> 1	1
> 2	1
> 3	1
```

### function returning a generator

if the function is returning a generator insted of a np.ndarray, map will pull as many elements from the generator as needed to fill a np.ndarray of the size of a column
```python
q = Qfrom({'a': [1, 2, 3]})
q.map(func=lambda: (c for c in 'python'), out='b')
```
```
> Qfrom
> a	b
> 1	p
> 2	y
> 3	t
```

using the generator col.id is a simple way to get a id column. More information in section [id](#id)
```python
q = Qfrom({'a': [1, 2, 3]})
q.map(func=col.id, out='i')
```
```
> Qfrom
> a	i
> 1	0
> 2	1
> 3	2
```

[Contents](#contents)

---

## orderby
```python
data = [
    {'a': 3, 'b': 4, 'c': 1},
    {'a': 2, 'b': 3, 'c': 2},
    {'a': 2, 'b': 2, 'c': 3},
    {'a': 1, 'b': 1, 'c': 4},
]
q = Qfrom(data)
q.orderby('a')
```
```
> Qfrom
> a	b	c
> 1	1	4
> 2	3	2
> 2	2	3
> 3	4	1
```

```python
data = [
    {'a': 3, 'b': 4, 'c': 1},
    {'a': 2, 'b': 3, 'c': 2},
    {'a': 2, 'b': 2, 'c': 3},
    {'a': 1, 'b': 1, 'c': 4},
]
q = Qfrom(data)
q.orderby('a', reverse=True)
```
```
> Qfrom
> a	b	c
> 3	4	1
> 2	2	3
> 2	3	2
> 1	1	4
```

it is possible to order by multiple keys.
```python
data = [
    {'a': 3, 'b': 4, 'c': 1},
    {'a': 2, 'b': 3, 'c': 2},
    {'a': 2, 'b': 2, 'c': 3},
    {'a': 1, 'b': 1, 'c': 4},
]
q = Qfrom(data)
q.orderby('a, b')
```
```
> Qfrom
> a	b	c
> 1	1	4
> 2	2	3
> 2	3	2
> 3	4	1
```

it is possible to transform the key column through a function.
```python
data = [
    {'a': 3, 'b': 4, 'c': 1},
    {'a': 2, 'b': 3, 'c': 2},
    {'a': 2, 'b': 2, 'c': 3},
    {'a': 1, 'b': 1, 'c': 4},
]
q = Qfrom(data)
q.orderby('a', lambda x: x%2)
```
```
> Qfrom
> a	b	c
> 2	3	2
> 2	2	3
> 3	4	1
> 1	1	4
```

if selection is not speecified the passed columns will be choosen by ne names of arguments of the given function.
```python
data = [
    {'a': 3, 'b': 4, 'c': 1},
    {'a': 2, 'b': 3, 'c': 2},
    {'a': 2, 'b': 2, 'c': 3},
    {'a': 1, 'b': 1, 'c': 4},
]
q = Qfrom(data)
q.orderby(func=lambda a: a%2)
```
```
> Qfrom
> a	b	c
> 2	3	2
> 2	2	3
> 3	4	1
> 1	1	4
```

[Performance test](#orderby-1)

[Contents](#contents)

---

## where
```python
q = Qfrom({
    'a': [True, False, True, False, True],
    'b': [1, 1, 1, 1, 0],
    'c': [1, 2, 3, 4, 5]
})
q.where('a')
```
```
> Qfrom
> a	b	c
> True	1	1
> True	1	3
> True	0	5
```

it is possible to pass multiple keys into where method. The values in all selected columns will first be parst to booleans. The parsed columns will be combined through a logical and operation to resive the final boolean key array which determines which rows will be passed to the result Qfrom.
```python
q = Qfrom({
    'a': [True, False, True, False, True],
    'b': [1, 1, 1, 1, 0],
    'c': [1, 2, 3, 4, 5]
})
q.where('a, b')
```
```
> Qfrom
> a	b	c
> True	1	1
> True	1	3
```

it is possible to transform the key column through a function.
```python
q = Qfrom({
    'a': [True, False, True, False, True],
    'b': [1, 1, 1, 1, 0],
    'c': [1, 2, 3, 4, 5]
})
q.where('c', lambda x: x < 3)
```
```
> Qfrom
> a	b	c
> True	1	1
> False	1	2
```

if selection is not speecified the passed columns will be choosen by ne names of arguments of the given function.
```python
q = Qfrom({
    'a': [True, False, True, False, True],
    'b': [1, 1, 1, 1, 0],
    'c': [1, 2, 3, 4, 5]
})
q.where(func=lambda c: c < 3)
```
```
> Qfrom
> a	b	c
> True	1	1
> False	1	2
```

[Performance test](#where-1)

[Contents](#contents)

---

## groupy
```python
q = Qfrom({
    'a': [1, 1, 2, 2],
    'b': [3, 3, 3, 4],
    'c': [5, 6, 7, 8]
})
q.groupby('a')
```
```
> Qfrom
> key	group
> 1	Qfrom
> a	b	c
> 1	3	5
> 1	3	6
> 2	Qfrom
> a	b	c
> 2	3	7
> 2	4	8
```

it is possible to group by multiple keys. Therefore the selcted columns will be transforemd to one column full of tuples holding the items from the selected columns.
```python
q = Qfrom({
    'a': [1, 1, 2, 2],
    'b': [3, 3, 3, 4],
    'c': [5, 6, 7, 8]
})
q.groupby('a, b')
```
```
> Qfrom
> key	group
> (1, 3)	Qfrom
> a	b	c
> 1	3	5
> 1	3	6
> (2, 3)	Qfrom
> a	b	c
> 2	3	7
> (2, 4)	Qfrom
> a	b	c
> 2	4	8
```

it is possible to transform the key column through a function.
```python
q = Qfrom({
    'a': [1, 1, 2, 2],
    'b': [3, 3, 3, 4],
    'c': [5, 6, 7, 8]
})
q.groupby('c', lambda x: x%2)
```
```
> Qfrom
> key	group
> 1	Qfrom
> a	b	c
> 1	3	5
> 2	3	7
> 0	Qfrom
> a	b	c
> 1	3	6
> 2	4	8
```

if selection is not speecified the passed columns will be choosen by ne names of arguments of the given function.
```python
q = Qfrom({
    'a': [1, 1, 2, 2],
    'b': [3, 3, 3, 4],
    'c': [5, 6, 7, 8]
})
q.groupby(func=lambda c: c%2)
```
```
> Qfrom
> key	group
> 1	Qfrom
> a	b	c
> 1	3	5
> 2	3	7
> 0	Qfrom
> a	b	c
> 1	3	6
> 2	4	8
```

[Performance test](#groupby)

[Contents](#contents)

---

## flatten
```python
q = Qfrom({
    'a': [1, 2],
    'b': [[3, 4], [5, 6]]
})
q.flatten('b')
```
```
> Qfrom
> a	b
> 1	3
> 1	4
> 2	5
> 2	6
```

[Contents](#contents)

---

## unique
collects first appearing items in Qfrom with a unique key.
```python
q = Qfrom({
    'a': [1, 2, 2, 3, 3],
    'b': [4, 5, 5, 6, 7]
})
q.unique('a')
```
```
> Qfrom
> a	b
> 1	4
> 2	5
> 3	6
```

it is possible to pass multiple keys.
```python
q = Qfrom({
    'a': [1, 2, 2, 3, 3],
    'b': [4, 5, 5, 6, 7]
})
q.unique('a, b')
```
```
> Qfrom
> a	b
> 1	4
> 2	5
> 3	6
> 3	7
```

## value counts
count how often each key appears in the given Qfrom.
```python
q = Qfrom({
    'a': [1, 2, 2, 3, 3],
    'b': [4, 5, 5, 6, 7]
})
q.unique('a')
```
```
> Qfrom
> value	count
> 1	1
> 2	2
> 3	2
```

it is possible to pass multiple keys. Therefore the selcted columns will be transforemd to one column full of tuples holding the items from the selected columns.
```python
q = Qfrom({
    'a': [1, 2, 2, 3, 3],
    'b': [4, 5, 5, 6, 7]
})
q.value_counts('a, b')
```
```
> Qfrom
> value	count
> (1, 4)	1
> (2, 5)	2
> (3, 6)	1
> (3, 7)	1
```

[Contents](#contents)

---

## agg
if one function is passed to agg, the function will be applied to every column.
```python
q = Qfrom({
    'a': [1, 2, 3],
    'b': [4, 5, 6]
})
q.agg(agg.sum)
```
```
> (6, 15)
```

multiple functions can be passed as a tuple of functions. Each function will be applied to the corresponding column in order of key apperances in the Qfrom.
```python
q = Qfrom({
    'a': [1, 2, 3],
    'b': [4, 5, 6]
})
q.agg((agg.max, agg.min))
```
```
> (3, 4)
```

[Performance test](#agg-2)

[Contents](#contents)

---

## join

[Contents](#contents)

---

## join cross

[Contents](#contents)

---

## join outer

[Contents](#contents)

---

## join outer left

[Contents](#contents)

---

## join outer right

[Contents](#contents)

---

## join id

[Contents](#contents)

---

## join id outer

[Contents](#contents)

---

## join id outer left

[Contents](#contents)

---

## join id outer right

[Contents](#contents)

---

## concat

[Contents](#contents)

---

## concat outer

[Contents](#contents)

---

## concat outer left

[Contents](#contents)

---

## concat outer right

[Contents](#contents)

---

## calculate

[Contents](#contents)

---

## call


[Contents](#contents)

---

# class col

class is a colection of functions which can easily be applied to colums of in a Qfrom. 

import col like this

```python
from QfromPackage.Qfrom_slim import col
```

## 0 -> 1 functions

### id
```python
g = col.id()
print(next(g))
print(next(g))
print(next(g))
```
```
> 0
> 1
> 2
```

```python
q = Qfrom({'a': ['x', 'y', 'z']})
q.map(func=col.id, out='id')
```
```
> Qfrom
> a	id
> x	0
> y	1
> z	2
```

[Contents](#contents)

---

## 1 -> 1 functions
function which resive one np.ndarray and return one np.ndarray of same lenght.

### normalize
```python
a = np.array([1, 2, 3, 4])
col.normalize(a)
```
```
> array([0.25, 0.5 , 0.75, 1.  ])
```

```python
q = Qfrom({'a': [1, 2, 3, 4]})
q.map('a', col.normalize)
```
```
> Qfrom
> a
> 0.25
> 0.5
> 0.75
> 1.0
```

```python
a = np.array([1, -2, 3, -4])
col.normalize(a)
```
```
> array([ 0.25, -0.5 ,  0.75, -1.  ])
```

[Contents](#contents)

---

### abs
```python
a = np.array([1, -2, 3, -4])
col.abs(a)
```
```
> array([1, 2, 3, 4])
```

[Contents](#contents)

---

### shift
```python
a = np.array([1, 2, 3, 4])
col.shift(steps=1, default_value=0)(a)
```
```
> array([0, 1, 2, 3])
```

[Contents](#contents)

---

### not

[Contents](#contents)

---

## n -> 1 functions
function which resive one np.ndarray and return multiple np.ndarray of same lenght.

### any

[Contents](#contents)

---
### all

[Contents](#contents)

---
### min

[Contents](#contents)

---
### min_colname
```python
a = np.array([1, 2, 3, 4])
b = np.array([4, 3, 2, 1])

col.min_colname(a=a, b=b)
```
```
> array(['a', 'a', 'b', 'b'], dtype=object)
```

```python
q = Qfrom({
    'a': [1, 2, 3, 4],
    'b': [4, 3, 2, 1]
})
q.map('*', col.min_colname, 'min')
```
```
> Qfrom
> a	b	min
> 1	4	a
> 2	3	a
> 3	2	b
> 4	1	b
```

[Contents](#contents)

---

### max

[Contents](#contents)

---
### max_colname
```python
a = np.array([1, 2, 3, 4])
b = np.array([4, 3, 2, 1])

col.max_colname(a=a, b=b)
```
```
> array(['b', 'b', 'a', 'a'], dtype=object)
```

```python
q = Qfrom({
    'a': [1, 2, 3, 4],
    'b': [4, 3, 2, 1]
})
q.map('*', col.max_colname, 'max')
```
```
> Qfrom
> a	b	max
> 1	4	b
> 2	3	b
> 3	2	a
> 4	1	a
```

[Contents](#contents)

---

### sum

[Contents](#contents)

---
### mean

[Contents](#contents)

---
### median

[Contents](#contents)

---
### var

[Contents](#contents)

---
### eq

[Contents](#contents)

---
### agg

[Contents](#contents)

---
### state

[Contents](#contents)

---
### lod_and

[Contents](#contents)

---
### lod_or

[Contents](#contents)

---
### lod_xor

[Contents](#contents)

---

## 1 -> n functions
### copy

[Contents](#contents)

---
### flatten

[Contents](#contents)

---

## n -> m functions
### ml_models

[Contents](#contents)

---

# class func

import func like this

```python
from QfromPackage.Qfrom_slim import func
```

## vec

[Contents](#contents)

---
## multicol

[Contents](#contents)

---

# class agg

import agg like this

```python
from QfromPackage.Qfrom_slim import agg
```

## any

[Contents](#contents)

---
## all

[Contents](#contents)

---
## min

[Contents](#contents)

---
## min_id

[Contents](#contents)

---
## max

[Contents](#contents)

---
## max_id

[Contents](#contents)

---
## sum

[Contents](#contents)

---
## mean

[Contents](#contents)

---
## median

[Contents](#contents)

---
## var

[Contents](#contents)

---
## len

[Contents](#contents)

---
## size

[Contents](#contents)

---
## state

[Contents](#contents)

 ---

# class plot

import plot like this

```python
from QfromPackage.Qfrom_slim import plot
```

## plot

[Contents](#contents)

---
## bar

[Contents](#contents)

---
## hist

[Contents](#contents)

---
## box

[Contents](#contents)

---
## scatter

[Contents](#contents)

---

# class out

import out like this

```python
from QfromPackage.Qfrom_slim import out
```

## list

[Contents](#contents)

---
## set

[Contents](#contents)

---
## dict

[Contents](#contents)

---
## array

[Contents](#contents)

---
## mtx

[Contents](#contents)

---
## df

[Contents](#contents)

---
## csv

[Contents](#contents)

---
## csv file

[Contents](#contents)

---
## json

[Contents](#contents)

---
## json file

[Contents](#contents)

---

# class trans

import trans like this

```python
from QfromPackage.Qfrom_slim import trans
```

## shuffle

[Contents](#contents)

---

# Performance Tests

In this section several modules for data manipulation will be getting compared. Therefore several common methods for data manipulation are getting explored in regard to there runtimes.

The explored modules are numpy, pandas, python lists and Qfrom. 

## setup

this section discripes the test data.

data set generation
```python
def get_p(n):
    return [1/(2**(i+1)) if i+1 != n else 1/(2**(i)) for i in range(n)]

def get_tab_data(n):
    name_list = ['Ann', 'Steven', 'Max', 'Jack', 'Julia', 'Clara', 'Emma', 'Bob', 'Anna' 'Lena']
    job_list = ['employee', 'jobless', 'freelancer', 'artist', 'technician', 'leader', 'coach', 'manager']
    max_age = 100
    max_salary = 1_000_000

    return {
        'name': np.random.choice(name_list, n, p=get_p(len(name_list))),
        'age': np.random.randint(max_age, size=n),
        'job': np.random.choice(job_list, n, p=get_p(len(job_list))),
        'salary': np.random.randint(max_salary, size=n),
        }
```

How data gets transformed to meet the requirements of the different modules.
```python
class setup():
    @classmethod
    def np(cls, data: dict[str, numpy.ndarray]):
        return {k: np.copy(v) for k,v in data.items()}
    @classmethod
    def np_tpl(cls, data: dict[str, numpy.ndarray]):
        return ({k:np.copy(v) for k,v in data[0].items()}, *data[1:])
    @classmethod
    def np_mtx(cls, data: dict[str, numpy.ndarray]):
        cols = list(data.values())
        return np_ext.col_stack(cols)
    @classmethod
    def df(cls, data: dict[str, numpy.ndarray]):
        return pd.DataFrame(data)
    @classmethod
    def df_tpl(cls, data: dict[str, numpy.ndarray]):
        return (pd.DataFrame(data[0]), *data[1:])
    @classmethod
    def l(cls, data: dict[str, numpy.ndarray]):
        return {key: list(col) for key, col in data.items()}
    @classmethod
    def l_tpl(cls, data: dict[str, numpy.ndarray]):
        return ({key: list(col) for key, col in data[0].items()}, *data[1:])
    @classmethod
    def qs(cls, data: dict[str, numpy.ndarray]):
        return Qfrom_slim(data)
    @classmethod
    def qs_tpl(cls, data: dict[str, numpy.ndarray]):
        return (Qfrom_slim(data[0]), *data[1:])
    @classmethod
    def list_items(cls, data: dict[str, numpy.ndarray]):
        return list(iter_table_dict(data))
```

[Contents](#contents)

---

## runtime tests

the different data manipulation methods get executed on multiple datasets of varying sizes

### append

append method implementations
```python
def append_np(data):
    result = {
        'name': np.array([data[0][0]]),
        'age': np.array([data[0][1]]),
        'job': np.array([data[0][2]]),
        'salary': np.array([data[0][3]]),
        }
    for name, age, job, salary in data[1:]:
        result['name'] = np.append(result['name'], [name])
        result['name'] = np.append(result['age'], [age])
        result['name'] = np.append(result['job'], [job])
        result['name'] = np.append(result['salary'], [salary])
    return result

def append_df(data):
    result = pd.DataFrame([data[0]], columns=['name', 'age', 'job', 'salary'])
    for t in data[1:]:
        row = pd.DataFrame([t], columns=['name', 'age', 'job', 'salary'])
        result.append([row], ignore_index=True)
    return result

def append_qs(data):
    result = Qfrom_slim()
    for t in data:
        result.append(t)
    return result(out.dict)

def append_l(data):
    result = {
        'name': [],
        'age': [],
        'job': [],
        'salary': [],
    }
    for name, age, job, salary in data:
        result['name'].append(name)
        result['age'].append(age)
        result['job'].append(job)
        result['salary'].append(salary)
    return result
```

measured runtimes dependend on the size of the input data sets
![](Images/append%20comparison.png)

runtimes for max data set size n=10 000

||||
|---|---|---|
|np	|0.174 s	|1.0%	|
|df	|8.22 s	|47.184%	|
|qs	|0.492 s	|2.823%	|
|l	|0.003 s	|0.016%	|

[Contents](#contents)

---

### getitem

getitem method implementations
```python
def getitem_np(t):
    data, ids = t
    for id in ids:
        tuple(col[id] for col in data.values())

def getitem_df(t):
    df, ids = t
    for id in ids:
        df.iloc[id]

def getitem_qs(t):
    q, ids = t
    for id in ids:
        q[id]

def getitem_l(t):
    data, ids = t
    for id in ids:
        tuple(col[id] for col in data.values())
```

![](Images/getitem%20comparison.png)

runtimes for max data set size n=100 000

||||
|---|---|---|
|np	|0.144 s	|1.0%	|
|df	|5.486 s	|38.089%	|
|qs	|0.161 s	|1.119%	|
|l	|0.119 s	|0.826%	|

[Contents](#contents)

---

### iter

iter method implementations
```python
def iter_np(data):
    row_count = 0
    for _ in np.nditer(list(data.values()), flags=["refs_ok"]):
        row_count += 1
    return row_count
def iter_np_mtx(data):
    row_count = 0
    for _ in np.nditer(data, flags=["refs_ok"]):
        row_count += 1
    return row_count

def iter_df(df: pd.DataFrame):
    row_count = 0
    for _ in df.values:
        row_count += 1
    return row_count

def iter_qs(q: Qfrom_slim):
    row_count = 0
    for _ in q:
        row_count += 1
    return row_count

def iter_l(data):
    row_count = 0
    for _ in zip(*data.values()):
        row_count += 1
    return row_count
```

![](Images/iter%20comparison.png)

runtimes for max data set size n=1 000 000

||||
|---|---|---|
|np	|0.16 s	|1.0%	|
|df	|0.163 s	|1.015%	|
|qs	|0.438 s	|2.736%	|
|l	|0.044 s	|0.275%	|
|np_mtx	|0.06 s	|0.375%	|

[Contents](#contents)

---

### select

select method implementations
```python
cols = 'name'

def select_np(data):
    return {key:value for key, value in data.items() if key in cols}

def select_df(df):
    return df[cols]

def select_qs(q: Qfrom_slim):
    return q.select(cols)(out.dict)

def select_l(data):
    return {key:value for key, value in data.items() if key in cols}
```

![](Images/select%20comparison.png)

runtimes for max data set size n=1 000 000

||||
|---|---|---|
|np	|0.0 s	|0%	|
|df	|0.0 s	|0%	|
|qs	|0.0 s	|0%	|
|l	|0.0 s	|0%	|

---

select multiple columns method implementations
```python
cols = ['name', 'age']

def select_np(data):
    return {key:value for key, value in data.items() if key in cols}

def select_df(df):
    return df[cols]

def select_qs(q: Qfrom_slim):
    return q.select(cols)(out.dict)

def select_l(data):
    return {key:value for key, value in data.items() if key in cols}
```

![](Images/select%20mult%20col%20comparison.png)

runtimes for max data set size n=10 000 000

||||
|---|---|---|
|np	|0.0 s	|0%	|
|df	|0.106 s	|0%	|
|qs	|0.0 s	|0%	|
|l	|0.0 s	|0%	|

[Contents](#contents)

---

### map

map add method implementations
```python
def map_add_np(data):
    data['age'] = data['age']+10
    return data

def map_add_df(df: pd.DataFrame):
    data['age'] = data['age']+10
    return df

def map_add_qs(q: Qfrom_slim):
    q = q.map(func=lambda age: age+10)
    return q(out.dict)

def map_add_l(data):
    data['age'] = [x+10 for x in data['age']]
    return data
```

![](Images/map%20add%20comparison.png)

runtimes for max data set size n=10 000 000

||||
|---|---|---|
|np	|0.008 s	|1.0%	|
|df	|0.016 s	|2.05%	|
|qs	|0.008 s	|1.0%	|
|l	|1.223 s	|152.791%	|

---

map by func method implementations
```python
def test_func(x): f'i am {x} years old'

def map_func_np(data):
    map_age = np.frompyfunc(test_func, 1, 1)
    data['age'] = map_age(data['age'])
    return data

def map_func_df(df: pd.DataFrame):
    df['age'] = df['age'].apply(test_func)
    return df

def map_func_qs(q: Qfrom_slim):
    q = q.map('age', func.vec(test_func))
    return q(out.dict)

def map_func_l(data):
    data['age'] = [test_func(x) for x in data['age']]
    return data
```

![](Images/map%20by%20func%20comparison.png)

runtimes for max data set size n=10 000 000

||||
|---|---|---|
|np	|1.29 s	|1.0%	|
|df	|2.021 s	|1.567%	|
|qs	|1.28 s	|0.993%	|
|l	|2.547 s	|1.975%	|

---

map by func two arguments method implementations
```python
def test_func(x, y): return f'My name is {x} and i am {y} years old'

def map_func_np(data):
    map_age = np.frompyfunc(test_func, 2, 1)
    data['msg'] = map_age(data['name'], data['age'])
    return data

def map_func_df(df: pd.DataFrame):
    df['msg'] = df.apply(lambda x: test_func(x['name'], x['age']), axis=1)
    return df

def map_func_qs(q: Qfrom_slim):
    q = q.map('name, age', func.vec(test_func), 'msg')
    return q(out.dict)

def map_func_l(data):
    data['msg'] = [test_func(*a) for a in zip(data['name'], data['age'])]
    return data

def map_func_df_lcph(df: pd.DataFrame):
    df['msg'] = [test_func(*a) for a in zip(df['name'], df['age'])]
    return df

def map_func_df_np(df: pd.DataFrame):
    map_age = np.frompyfunc(test_func, 2, 1)
    df['msg'] = map_age(df['name'], df['age'])
    return df
```

![](Images/map%20%20by%20func%202%20args%20comparison.png)

runtimes for max data set size n=1 000 000

||||
|---|---|---|
|np	|0.225 s	|1.0%	|
|df	|7.09 s	|31.476%	|
|qs	|0.216 s	|0.959%	|
|l	|0.404 s	|1.793%	|
|df_lcph	|0.355 s	|1.575%	|
|df_np	|0.194 s	|0.861%	|

[Contents](#contents)

---

### orderby

orderby method implementations
```python
def orderby_np(data):
    sorted_ids = np.argsort(data['age'])
    return {key: value[sorted_ids] for key, value in data.items()}

def orderby_df(df):
    return df.sort_values('age')

def orderby_qs(q: Qfrom_slim):
    return q.orderby('age')(out.dict)

def orderby_l(data):
    sorted_ids = sorted(range(len(data['age'])), key=lambda x: data['age'][x])
    return {key: [value[i] for i in sorted_ids] for key, value in data.items()}
```

![](Images/orderby%20comparison.png)

runtimes for max data set size n=1 000 000

||||
|---|---|---|
|np	|0.094 s	|1.0%	|
|df	|0.155 s	|1.648%	|
|qs	|0.116 s	|1.242%	|
|l	|1.156 s	|12.33%	|

---

orderby multiple columns method implementations
```python
def orderby_mult_np(data):
    sorted_ids = np.lexsort([data['age'], data['name']])
    return {key:value[sorted_ids] for key, value in data.items()}

def orderby_mult_df(df):
    return df.sort_values(['name', 'age'])

def orderby_mult_qs(q: Qfrom_slim):
    return q.orderby('name, age')(out.dict)
```

![](Images/orderby%20mult%20cols%20comparison.png)

runtimes for max data set size n=10 000 000

||||
|---|---|---|
|np	|8.315 s	|1.0%	|
|df	|2.378 s	|0.286%	|
|qs	|7.396 s	|0.889%	|

[Contents](#contents)

---

### where

where method implementations
```python
def where_np(data):
    job_filter = np.where(data['job']=='manager')
    data = {key: value[job_filter] for key, value in data.items()}
    return data

def where_df(df):
    return df[df['job']=='manager']

def where_qs(q: Qfrom_slim):
    return q.where('job', lambda x: x=="manager")(out.dict)

def where_l(data):
    job_filter = [i for i in range(len(data['job'])) if data['job'][i]=='manager']
    return {key: [value[i] for i in job_filter] for key, value in data.items()}
```

![](Images/where%20comparison.png)

runtimes for max data set size n=10 000 000

||||
|---|---|---|
|np	|0.232 s	|1.0%	|
|df	|0.619 s	|2.672%	|
|qs	|0.232 s	|1.002%	|
|l	|1.428 s	|6.166%	|

[Contents](#contents)

---

### groupby

not easy compareable bacause pandas groupby is only returning ids.

groupby method implementations
```python
cols = 'job'

def groupby_np(data):
    sorted_ids = np.argsort(data[cols])
    sorted_key_array = data[cols][sorted_ids]
    unique_keys, unique_key_ids = np.unique(sorted_key_array, return_index=True)
    id_groups = np.split(sorted_ids, unique_key_ids[1:])
    group_dict = {
        'key': unique_keys,
        'group': np.array([{key:col[ids] for key, col in data.items()} for ids in id_groups])
        }

    return group_dict

def groupby_df(df: pd.DataFrame):
    return df.groupby(cols).groups

def groupby_qs(q: Qfrom_slim):
    return q.groupby(cols)(out.dict)
```

![](Images/groupby%20comparison.png)

runtimes for max data set size n=10 000 000

||||
|---|---|---|
|np	|2.466 s	|1.0%	|
|df	|0.627 s	|0.254%	|
|qs	|7.511 s	|3.046%	|

---

groupby multiple columns method implementations
```python
cols = ['job', 'name']

def groupby_np(data):
    sorted_ids = np.lexsort([data[c] for c in cols[::-1]])
    sorted_key_array = array_tuple_to_tuple_array([c for k, c in data.items() if k in cols])
    unique_keys, unique_key_ids = np.unique(sorted_key_array, return_index=True)
    id_groups = np.split(sorted_ids, unique_key_ids[1:])
    group_dict = {
        'key': unique_keys,
        'group': np.array([{key:col[ids] for key, col in data.items()} for ids in id_groups])
        }
    
    return group_dict


def groupby_df(df: pd.DataFrame):
    return df.groupby(cols).groups

def groupby_qs(q: Qfrom_slim):
    return q.groupby(cols)(out.dict)
```

![](Images/groupby%20mult%20cols%20comparison.png)

runtimes for max data set size n=1 000 000

||||
|---|---|---|
|np	|3.431 s	|1.0%	|
|df	|0.917 s	|0.267%	|
|qs	|1.089 s	|0.318%	|

[Contents](#contents)

---

### agg

agg method implementations
```python
def agg_np(data):
    return np.mean(data['age'])

def agg_df(df: pd.DataFrame):
    return df['age'].agg('mean')

def agg_qs(q: Qfrom_slim):
    return q['age'].agg(agg.mean)

def agg_l(data):
    return sum(data['age']) / len(data['age'])
```

![](Images/agg%20comparison.png)

runtimes for max data set size n=10 000 000

||||
|---|---|---|
|np	|0.005 s	|1.0%	|
|df	|0.006 s	|1.2%	|
|qs	|0.005 s	|1.0%	|
|l	|0.372 s	|74.305%	|

[Contents](#contents)