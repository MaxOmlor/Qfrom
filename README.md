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
  - [aa](#aa)
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
  - [1 -> 1 functions](#1---1-functions)
    - [pass_none](#pass_none)
    - [normalize](#normalize)
    - [abs](#abs)
    - [center](#center)
    - [shift](#shift)
    - [not](#not)
    - [id](#id)
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
    - [agg](#agg)
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
  - [results](#results)
    - [append](#append-1)
    - [getitem](#getitem-1)
    - [iter](#iter-1)
    - [select](#select-1)
    - [map](#map-1)
    - [orderby](#orderby-1)
    - [where](#where-1)
    - [groupby](#groupby)
    - [agg](#agg-1)
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

## str
```python
q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
str(q)
```
```
> 'Qfrom\na\tb\n1\t4\n2\t5\n3\t6'
```

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

## getitem

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

## len
```python
q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
len(q)
```
```
> 3
```

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

## select

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

## map

args
- args: str | tuple[str] | list[str] = None,
  
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

if args is not speecified the passed columns will be choosen by ne names of arguments from the given function.
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

## orderby

## where

## groupy

## flatten

## unique

## value counts

## aa

## join

## join cross

## join outer

## join outer left

## join outer right

## join id

## join id outer

## join id outer left

## join id outer right

## concat

## concat outer

## concat outer left

## concat outer right

## calculate

## call


[Contents](#contents)

---

# class col

import col like this

```python
from QfromPackage.Qfrom_slim import col
```

## 1 -> 1 functions

### pass_none
### normalize
### abs
### center
### shift
### not
### id

## n -> 1 functions

### any
### all
### min
### min_colname
### max
### max_colname
### sum
### mean
### median
### var
### eq
### agg
### state
### lod_and
### lod_or
### lod_xor

## 1 -> n functions
### copy
### flatten

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
## multicol


[Contents](#contents)

---

# class agg

import agg like this

```python
from QfromPackage.Qfrom_slim import agg
```

## any
## all
## min
## min_id
## max
## max_id
## sum
## mean
## median
## var
## len
## size
## state


[Contents](#contents)

 ---

# class plot

import plot like this

```python
from QfromPackage.Qfrom_slim import plot
```

## plot
## bar
## hist
## box
## scatter


[Contents](#contents)

---

# class out

import out like this

```python
from QfromPackage.Qfrom_slim import out
```

## list
## set
## dict
## array
## mtx
## df
## csv
## csv file
## json
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

## setup

get_tab_data
setup

## results

### append
### getitem
### iter
### select
### map
### orderby
### where
### groupby
### agg

[Contents](#contents)