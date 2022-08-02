import unittest
from Qfrom_slim import col, func, agg, plot, out, trans, Qfrom, parse
import pandas as pd
import numpy as np

class TestColClass(unittest.TestCase):
    # = 0 -> 1
    #   (- <Any> -> singleton to array)
    #
    # = 1 -> 1
    #   - pass_none
    #   - normalize
    def test_normalize(self):
        a1 = np.array([1, 2, 3, 4])
        a2 = np.array([1, -2, 3, -4])

        result1 = np.array([.25, .5, .75, 1])
        result2 = np.array([.25, -.5, .75, -1])

        self.assertTrue(np.array_equal(col.normalize(a1), result1))
        self.assertTrue(np.array_equal(col.normalize(a2), result2))
    #   - abs
    #   - center -> set a new origin for a column: [1, 2, 3], origin=2 -> [-1, 0, 1]
    #   - shift(steps=...)
    def test_shift(self):
        a1 = np.array([1, 2, 3, 4])

        result1 = np.array([0, 1, 2, 3])

        self.assertTrue(np.array_equal(col.shift(1, 0)(a1), result1))
    #   - not
    #   - id
    #
    # = n -> 1
    #   - any
    #   - all
    #   - min
    #   - min_colname
    def test_min_colname(self):
        a = np.array([1, 2, 3, 4])
        b = np.array([4, 3, 2, 1])

        result = np.array(['a', 'a', 'b', 'b'])

        self.assertTrue(np.array_equal(col.min_colname(a=a, b=b), result))
    #   - max
    #   - max_colname
    def test_max_colname(self):
        a = np.array([1, 2, 3, 4])
        b = np.array([4, 3, 2, 1])

        result = np.array(['b', 'b', 'a', 'a'])

        self.assertTrue(np.array_equal(col.max_colname(a=a, b=b), result))
    #   - sum
    #   - mean
    #   - median
    #   - var
    #   - eq
    #   - agg(colcount) -> combines multible cols to one 2d col
    #   - state(rules: func|dict[func]) -> iterates over col. for each item the result state of the last item is feed in. 
    #   - lod_and
    #   - lod_or
    #   - lod_xor
    #
    # = 1 -> n
    #   - copy(n)
    #   - flatten -> autodetect out count
    #
    # = n -> m
    #   - ml_models
    pass

class TestFuncClass(unittest.TestCase):
    # - __call__(func, in: int, out: int) -> verpackt func in lambda, so dass lambda-parameter in-count entsprechen und output tuple out-count entspricht.
    # - vec(func) -> vectorize func, autodetect in and out counts
    def test_vec(self):
        a = np.array([1, 2, 3])

        f1 = lambda x: x+1
        f2 = lambda x: (x, x+1)
        f3 = lambda x: {'x1': x, 'x2': x+1}

        result1 = np.array([2, 3, 4])
        result2 = (np.array([1, 2, 3]), np.array([2, 3, 4]))
        result3 = {'x1': np.array([1, 2, 3]), 'x2': np.array([2, 3, 4])}

        self.assertTrue(np.array_equal(func.vec(f1)(a), result1))
        self.assertTrue(np.array_equal(x1, x2) for x1, x2 in zip(func.vec(f2)(a), result2))
        self.assertTrue(k1==k2 and np.array_equal(v1, v2) for (k1, v1), (k2, v2) in zip(func.vec(f3)(a).items(), result3.items()))
    # - vec(func, in: int, out: int)
    # - multicol(repetitioncount: int)
    # (- args(func))
    # (   -> ex. args(lambda a,b: a+b) -> {('a', 'b'): lambda a,b: a+b})
    # (   -> ex. args(lambda a,b, *args: (a+b, *args)) -> {('a', 'b', '*'): lambda a,b, *args: (a+b, *args)})

class TestAggClass(unittest.TestCase):
    # - any
    # - all
    # - min
    # - min_id
    # - max
    # - max_id
    # - sum
    # - mean
    # - median
    # - var
    # - len
    # - size
    # - state(rules: func|dict[func]) -> returns the last state of col.state
    pass

#class TestPlotClass(unittest.TestCase):
    # - plot
    # - bar
    # - hist
    # - box
    # - scatter

class TestOutClass(unittest.TestCase):
    # - tolist
    # - (toset)
    # - todict
    # - toarray
    # - (tomtx)
    # - todf
    # - tocsv
    # - (tocsvfile)
    # - (tojson)
    # - (tojsonfile)
    pass

class TestTransClass(unittest.TestCase):
    # - (shuffle)
    pass

class TestQfromClass(unittest.TestCase):
    # - import_list
    def test_init_list(self):
        self.assertEqual(Qfrom([1, 2, 3])(out.list), [1, 2, 3])

        l1 = [
            (1, 4),
            (2, 5),
            (3, 6),
            ]
        l2 = [
            {'a': 1, 'b': 4},
            {'a': 2, 'b': 5},
            {'a': 3, 'b': 6},
            ]
        class t:
            def __init__(self, *args):
                self.data = list(args)
            
            def __getitem__(self, key):
                return self.data[key]
            
            def __len__(self):
                return len(self.data)

            def __eq__(self, other):
                if type(self) != type(other):
                    return False
                return self.data == other.data
            
            def __str__(self):
                data_str = ', '.join(str(item) for item in self.data)
                return f't({data_str})'
            def __repr__(self):
                return str(self)
        l3 = [t(1, 2, 3), t(2, 3, 4), t(3, 4, 5)]
        
        q_result1 = Qfrom({'y0': [1, 2, 3], 'y1': [4, 5, 6]})
        q_result2 = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
        arr = np.empty(len(l3), dtype=object)
        for i, item in enumerate(l3):
            arr[i] = item
        q_result3 = Qfrom({'y': arr})

        self.assertEqual(Qfrom(l1), q_result1)
        self.assertEqual(Qfrom(l1)(out.list), l1)
        self.assertEqual(Qfrom(l2), q_result2)
        self.assertEqual(Qfrom(l3), q_result3)
    # - import_dict
    def test_init_dict(self):
        d = {'a': [1, 2, 3], 'b': [4, 5, 6]}

        self.assertEqual({key:list(col) for key,col in Qfrom(d)(out.dict).items()}, d)
    # - (import_set)
    # - (import_array)
    def test_init_array(self):
        a = np.array([1, 2, 3])

        q_result = Qfrom({'y': [1, 2, 3]})

        self.assertEqual(Qfrom(a), q_result)
    # - (import_mtx)
    # - import_dataframe
    def test_init_df(self):
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

        q_result = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})

        self.assertEqual(Qfrom(df), q_result)
    # - import_csv
    def test_init_csv(self):
        csv = '''
        a,b
        1,4
        2,5
        3,6
        '''

        q_result = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})

        self.assertEqual(Qfrom(csv), q_result)
    # - (import_json)
    # - import_generator
    def test_init_generator(self):
        g1 = ({'a': i} for i in range(3))
        g2 = ((i,) for i in range(3))

        q_result1 = Qfrom({'a': [0, 1, 2]})
        q_result2 = Qfrom({'y': [0, 1, 2]})

        self.assertEqual(Qfrom(g1), q_result1)
        self.assertEqual(Qfrom(g2), q_result2)

    # - eq
    def test_eq(self):
        self.assertEqual(Qfrom([1, 2, 3]), Qfrom([1, 2, 3]))
        self.assertNotEqual(Qfrom([1, 2, 3]), Qfrom([3, 2, 1]))

        q1 = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
        q2 = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
        q3 = Qfrom({'a': [1, 2, 3], 'b': [6, 5, 4]})
        q4 = Qfrom({'a': [1, 2, 3]})
        class t:
            def __init__(self, *args):
                self.data = list(args)
            
            def __getitem__(self, key):
                return self.data[key]
            
            def __len__(self):
                return len(self.data)

            def __eq__(self, other):
                if type(self) != type(other):
                    return False
                return self.data == other.data
            
            def __str__(self):
                data_str = ', '.join(str(item) for item in self.data)
                return f't({data_str})'
            def __repr__(self):
                return str(self)
        q5 = Qfrom([t(1, 2, 3), t(2, 3, 4), t(3, 4, 5)])
        q6 = Qfrom([t(1, 2, 3), t(2, 3, 4), t(3, 4, 5)])
        
        self.assertEqual(q1, q1)
        self.assertEqual(q1, q2)
        self.assertNotEqual(q1, q3)
        self.assertNotEqual(q1, q4)
        self.assertEqual(q5, q6)
    # - str
    def test_str(self):
        q1 = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
        qs1 = 'Qfrom\na\tb\n1\t4\n2\t5\n3\t6'''
        q2 = Qfrom()
        qs2 = 'Qfrom\nempty'

        self.assertEqual(str(q1), qs1)
        self.assertEqual(str(q2), qs2)
    # - repr
    # - append
    def test_append(self):
        q1 = Qfrom()
        q2 = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
        q1_result1 = Qfrom({'y': [1]})
        q1_result2 = Qfrom({'y': [1, 2]})
        q1_result3 = Qfrom({'y0': [1], 'y1': [2]})
        q1_result4 = Qfrom({'a': [1], 'b': [2]})
        q2_result = Qfrom({'a': [1, 2, 3, 7], 'b': [4, 5, 6, 8]})

        q1_1 = Qfrom(q1)
        q1_1.append(1)
        q1_2 = Qfrom(q1)
        q1_2.append((1,))
        q1_3 = Qfrom(q1)
        q1_3.append(1)
        q1_3.append(2)
        q1_4 = Qfrom(q1)
        q1_4.append((1, 2))
        q1_5 = Qfrom(q1)
        q1_5.append({'a': 1, 'b':2})

        q2_1 = Qfrom(q2)
        q2_1.append((7, 8))
        q2_2 = Qfrom(q2)
        q2_2.append({'a': 7, 'b':8})

        self.assertEqual(q1_1, q1_result1)
        self.assertEqual(q1_2, q1_result1)
        self.assertEqual(q1_3, q1_result2)
        self.assertEqual(q1_4, q1_result3)
        self.assertEqual(q1_5, q1_result4)
        self.assertEqual(q2_1, q2_result)
        self.assertEqual(q2_2, q2_result)
    # - setitem -> more dim slice support
    def test_setitem(self):
        q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
        q_result = Qfrom({'a': [1, 7, 3], 'b': [4, 8, 6]})
        
        q1 = Qfrom(q)
        q1[1] = (7, 8)
        q2 = Qfrom(q)
        q2[1] = {'a':7, 'b':8}
        q3 = Qfrom(q)
        q3[1] = {'a':7}
        q3[1] = {'b':8}

        self.assertEqual(q1, q_result)
        self.assertEqual(q2, q_result)
        self.assertEqual(q3, q_result)
    def test_setitem_col(self):
        q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
        q_result1 = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
        q_result2 = Qfrom({'a': [1, 2, 3], 'b': [7, 8, 9]})
        q_result3 = Qfrom({'a': [4, 5, 6], 'b': [1, 2, 3]})
        
        q1_1 = Qfrom(q)
        q1_1['c'] = [7, 8, 9]
        #q1_2 = Qfrom(q)
        #q1_2['c'] = np.array([7, 8, 9])
        q1_3 = Qfrom(q)
        q1_3[('c', )] = [7, 8, 9]
        q1_4 = Qfrom(q)
        q1_4['c'] = {'c': [7, 8, 9]}
        q1_5 = Qfrom(q)
        q1_5['c'] = [(7,), (8,), (9,)]
        #input Qfrom

        q2_1 = Qfrom(q)
        q2_1['b'] = [7, 8, 9]

        q3_1 = Qfrom(q)
        q3_1['a, b'] = [(4, 1), (5, 2), (6, 3)]
        q3_2 = Qfrom(q)
        q3_2[('a', 'b')] = [(4, 1), (5, 2), (6, 3)]
        #q3_3 = Qfrom(q)
        #q3_3[('a', 'b')] = np.array([[4, 1], [5, 2], [6, 3]])
        q3_4 = Qfrom(q)
        q3_4['a, b'] = {'a': [4, 5, 6], 'b': [1, 2, 3]}
        #q3_5 = Qfrom(q)
        #q3_5['...,b'] = {'a': [4, 5, 6], 'b': [1, 2, 3]}
        #q3_6 = Qfrom(q)
        #q3_6['*'] = {'a': [4, 5, 6], 'b': [1, 2, 3]}

        self.assertEqual(q1_1, q_result1)
        #self.assertEqual(q1_2, q_result1)
        self.assertEqual(q1_3, q_result1)
        self.assertEqual(q1_4, q_result1)
        self.assertEqual(q1_5, q_result1)

        self.assertEqual(q2_1, q_result2)

        self.assertEqual(q3_1, q_result3)
        self.assertEqual(q3_2, q_result3)
        #self.assertEqual(q3_3, q_result3)
        self.assertEqual(q3_4, q_result3)
        #self.assertEqual(q3_5, q_result3)
        #self.assertEqual(q3_6, q_result3)
    def test_setitem_col_id(self):
        q1 = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
        q2 = Qfrom({'a': ['a', 'b', 'c'], 'b': ['aa', 'bb', 'cc']})
        q3 = Qfrom({'a': ['', '', ''], 'b': ['aa', 'bb', 'cc']})
        q1_result1 = Qfrom({'a': [10, 2, 3], 'b': [4, 5, 6]})
        q1_result2 = Qfrom({'a': [10, 11, 3], 'b': [12, 13, 6]})
        q1_result3 = Qfrom({'a': [10, 12, 3], 'b': [12, 13, 6]})
        q2_result1 = Qfrom({'a': ['aa', 'b', 'c'], 'b': ['aa', 'bb', 'cc']})
        q2_result2 = Qfrom({'a': ['aa', 'bb', 'c'], 'b': ['aa', 'bb', 'cc']})
        q2_result3 = Qfrom({'a': ['a', 'b', 'c'], 'b': ['aa', 'bb', 'c']})
        q2_result4 = Qfrom({'a': ['', 'b', 'c'], 'b': ['aa', 'bb', 'cc']})
        q3_result1 = Qfrom({'a': ['a', '', ''], 'b': ['aa', 'bb', 'cc']})

        q1_1 = Qfrom(q1)
        q1_1['a', 0] = 10
        #q1_2 = Qfrom(q1)
        #q1_2['a', 0:2] = [10, 11]
        #q1_3 = Qfrom(q1)
        #q1_3['a,b', 0:2] = [(10, 11), (12, 13)]

        q2_1 = Qfrom(q2)
        q2_1['a', 0] = 'aa'
        #q2_2 = Qfrom(q2)
        #q2_2['a,b', 0] = ('aa','bb')
        #q2_3 = Qfrom(q2)
        #q2_3['a,b', 0] = {'b':'bb', 'a':'aa'}
        q2_4 = Qfrom(q2)
        q2_4['b', 2] = 'c'
        q2_5 = Qfrom(q2)
        q2_5['a', 0] = ''

        q3_1 = Qfrom(q3)
        q3_1['a', 0] = 'a'

        self.assertEqual(q1_1, q1_result1)
        #self.assertEqual(q1_2, q1_result2)
        #self.assertEqual(q1_3, q1_result3)
        self.assertEqual(q2_1, q2_result1)
        #self.assertEqual(q2_2, q2_result2)
        #self.assertEqual(q2_3, q2_result2)
        self.assertEqual(q2_4, q2_result3)
        self.assertEqual(q2_5, q2_result4)
        self.assertEqual(q3_1, q3_result1)
    # - getitem
    def test_getitem(self):
        self.assertEqual(Qfrom([1, 2, 3])[1], 2)
        self.assertEqual(Qfrom([1, 2, 3])[1:], Qfrom([2, 3]))
        self.assertEqual(Qfrom([1, 2, 3])[:-1], Qfrom([1, 2]))
        self.assertEqual(Qfrom([1, 2, 3])[1], 2)
        #self.assertEqual(Qfrom([[1, 2], [3, 4]])[1], np.array([3, 4]))
        #self.assertEqual(Qfrom([1, 2, 3, 4])[[1, 3]], Qfrom([2, 4]))

        q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
        self.assertEqual(q[1], (2, 5))
        self.assertEqual(q[1:], Qfrom({'a': [2, 3], 'b': [5, 6]}))
        self.assertEqual(q['a', 0], 1)
        self.assertEqual(q['...,b'], q)
        self.assertEqual(q['a,.'], q)
        self.assertEqual(q['*'], q)
        #self.assertEqual(q[lambda a,i: (i,a), 'i,a'], Qfrom({'i':[0,1,2], 'a':[1,2,3]}))
    # - contains
    def test_contains(self):
        q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})

        self.assertTrue((2, 5) in q)
        self.assertTrue({'a': 2, 'b':5} in q)
        self.assertFalse((0, 0) in q)
        self.assertFalse({'a': 0, 'b':0} in q)
        self.assertFalse((2, 4) in q)
        self.assertFalse({'a': 2, 'b':4} in q)
    # - iter
    def test_iter(self):
        self.assertEqual([item for item in Qfrom([1, 2, 3])], [1, 2, 3])
        self.assertEqual(next(iter(Qfrom([1, 2, 3]))), 1)

        q1 = Qfrom({'a': ['a', 'b', 'c'], 'b': ['d', 'e', 'f']})
        self.assertEqual([a+b for a,b in q1], ['ad', 'be', 'cf'])

        q2 = Qfrom().concat_outer(q1)
        self.assertEqual([a+b for a,b in q2], ['ad', 'be', 'cf'])

    # - keys
    def test_keys(self):
        q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})

        self.assertEqual(tuple(q.keys()), ('a', 'b'))
    # - values
    def test_values(self):
        q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})

        self.assertEqual(tuple(list(col) for col in q.values()), ([1, 2, 3], [4, 5, 6]))
    # - items
    # - (stats)
    # - len

    # - remove(selection: str|tuple[str]|list[str])
    def test_remove(self):
        q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
        
        q_result1 = Qfrom({'b': [4, 5, 6]})

        self.assertEqual(q.remove('a'), q_result1)
    # - rename(map: dict[str, str])
    def test_rename(self):
        q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
        
        q_result1 = Qfrom({'a': [1, 2, 3], 'c': [4, 5, 6]})
        q_result2 = Qfrom({'x': [1, 2, 3], 'y': [4, 5, 6]})

        self.assertEqual(q.rename({'b': 'c'}), q_result1)
        self.assertEqual(q.rename({'a': 'x', 'b': 'y'}), q_result2)
    # - select(selection: str|tuple[str]|list[str])
    def test_select_str(self):
        a = [1, 2, 3]
        b = [4, 5, 6]
        c = [7, 8, 9]
        d = [10, 11, 12]
        q = Qfrom({'a': a, 'b': b, 'c': c, 'd': d})

        q_result1 = Qfrom({'a': a})
        q_result2 = Qfrom({'a': a, 'c': c, 'd': d})
        q_result3 = Qfrom({'a': a, 'b': b})
        q_result4 = Qfrom({'b': b, 'a': a})
        q_result5 = Qfrom({'a': a, 'd': d})
        q_result7 = Qfrom({'d': d})
        q_result8 = Qfrom({'a': a, 'b': b, 'c': c})

        self.assertEqual(q.select('a'), q_result1)
        #self.assertEqual(q.select('-b'), q_result2)
        self.assertEqual(q.select('a, b'), q_result3)
        self.assertEqual(q.select('b, a'), q_result4)
        #self.assertEqual(q.select('-b, -c'), q_result5)
        self.assertEqual(q.select('d, ...'), q_result7)
        self.assertEqual(q.select('a,...,c'), q_result8)
        self.assertEqual(q.select('...,c'), q_result8)
        self.assertEqual(q.select('.,.,c'), q_result8)
        #self.assertEqual(q1.select('(a, b), *'), q_result3)
    def test_select_tuple(self):
        a = [1, 2, 3]
        b = [4, 5, 6]
        c = [7, 8, 9]
        d = [10, 11, 12]
        q = Qfrom({'a': a, 'b': b, 'c': c, 'd': d})

        q_result1 = Qfrom({'a': a})
        q_result2 = Qfrom({'a': a, 'c': c, 'd': d})
        q_result3 = Qfrom({'a': a, 'b': b})
        q_result4 = Qfrom({'b': b, 'a': a})
        q_result5 = Qfrom({'x': d, 'a': a, 'b': b, 'c': c, 'd': d})
        q_result6 = Qfrom({'d': d})
        q_result7 = Qfrom({'a': a, 'b': b, 'c': c})
        q_result8 = Qfrom({'a': a, 'b': b, 'c': c})
        q_result9 = Qfrom({'a': a, 'e': b})
        q_result10 = Qfrom({'x': a, 'y': a, 'z': a})

        self.assertEqual(q.select(('a',)), q_result1)
        #self.assertEqual(q.select(('-b', )), q_result2)
        self.assertEqual(q.select(('a', 'b')), q_result3)
        self.assertEqual(q.select(('b', 'a')), q_result4)
        self.assertEqual(q.select(('d', '...')), q_result6)
        self.assertEqual(q.select(('.', '.', 'c')), q_result7)
        self.assertEqual(q.select(('...', 'c')), q_result8)
    # - map(args: str|tuple[str]|list[str], func: callable, out=str|tuple[str]|list[str])
    def test_map(self):
        q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})

        q_result1 = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9], 'd': [5, 5, 5]})
        q_result2 = Qfrom({'a': [6, 6, 6], 'b': [4, 5, 6], 'c': [7, 8, 9]})
        q_result3 = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9], 'd': [6, 6, 6]})
        q_result4 = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9], 'i': [0, 1, 2]})
        q_result5 = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9], 'd': [7, 8, 9]})
        q_result6 = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9], 'd': [1, 2, 3]})
        q_result7 = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9], 'd': [5, 5, 5], 'i': [0, 1, 2]})

        self.assertEqual(q.map(func=lambda: 5, out='d'), q_result1)
        self.assertEqual(q.map('a', agg.sum), q_result2)
        self.assertEqual(q.map('a', agg.sum, 'd'), q_result3)
        self.assertEqual(q.map('a', agg.sum, ['d']), q_result3)
        #self.assertEqual(q.map(col.id), q_result1)
        self.assertEqual(q.map(func=col.id, out='i'), q_result4)
        self.assertEqual(q.map('a,b,c', col.max, 'd'), q_result5)
        self.assertEqual(q.map(['a','b','c'], col.max, 'd'), q_result5)
        self.assertEqual(q.map('*', col.max, 'd'), q_result5)
        self.assertEqual(q.map(func=lambda c: c), q)
        self.assertEqual(q.map(func=lambda c: (c,)), q)
        self.assertEqual(q.map(func=lambda c: {'c': c}), q)
        self.assertEqual(q.map('c', col.sum), q)
        self.assertEqual(q.map('a', out='d'), q_result6)
        self.assertEqual(q.map(func=lambda **kwrgs: kwrgs), q)
        
        #self.assertEqual(q.map(func=lambda: (5,col.id), out='d,i'), q_result7)
        #self.assertEqual(q.map(func=lambda: {'d':5, 'i': col.id}), q_result7)

    # - orderby(selection: str|tuple[str]|list[str], func: callable, reverse: bool)
    def test_orderby(self):
        q1 = Qfrom('''
        a,b,c
        3,4,1
        2,3,2
        2,2,3
        1,1,4
        ''')
        q_result1 = Qfrom('''
        a,b,c
        1,1,4
        2,3,2
        2,2,3
        3,4,1
        ''')
        q_result2 = Qfrom('''
        a,b,c
        3,4,1
        2,3,2
        2,2,3
        1,1,4
        ''')
        q_result3 = Qfrom('''
        a,b,c
        3,4,1
        2,2,3
        2,3,2
        1,1,4
        ''')
        q_result4 = Qfrom('''
        a,b,c
        1,1,4
        2,2,3
        2,3,2
        3,4,1
        ''')

        self.assertEqual(q1.orderby('a'), q_result1)
        self.assertEqual(q1.orderby('a', lambda x:-x), q_result2)
        self.assertEqual(q1.orderby(func=lambda a:-a), q_result2)
        self.assertEqual(q1.orderby('a', reverse=True), q_result3)
        self.assertEqual(q1.orderby('a, b'), q_result4)
        self.assertEqual(q1.orderby(('a', 'b')), q_result4)

    # - where(selection: str|tuple[str]|list[str], predicate: callable)
    def test_where(self):
        q1 = Qfrom({'a': [True, False, True, False, True], 'b': [5, 4, 3, 2, 1]})
        q2 = Qfrom({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
        q3 = Qfrom({'a': [True, False, True, False, True], 'b': [True, True, True, True, False], 'c': [5, 4, 3, 2, 1]})
        q4 = Qfrom({'a': [1, 0, 1, 0, 1], 'b': [5, 4, 3, 2, 1]})
        q5 = Qfrom({'a': [1, 0, 1, 0, 1], 'b': [1, 1, 1, 1, 0], 'c': [5, 4, 3, 2, 1]})

        q_result1 = Qfrom({'a': [True, True, True], 'b': [5, 3, 1]})
        q_result2 = Qfrom({'a': [1, 2], 'b': [5, 4]})
        q_result3 = Qfrom({'a': [True, True], 'b': [True, True], 'c': [5, 3]})
        q_result4 = Qfrom({'a': [1, 1, 1], 'b': [5, 3, 1]})
        q_result5 = Qfrom({'a': [1, 1], 'b': [1, 1], 'c': [5, 3]})

        self.assertEqual(q1.where('a'), q_result1)
        self.assertEqual(q2.where('a',lambda x: x<3), q_result2)
        self.assertEqual(q2.where(func=lambda a: a<3), q_result2)
        self.assertEqual(q3.where('a,b'), q_result3)
        self.assertEqual(q3.where(('a', 'b')), q_result3)
        self.assertEqual(q4.where('a'), q_result4)
        self.assertEqual(q5.where('a, b'), q_result5)

    # - groupby(selection: str|tuple[str]|list[str], func: callable)
    def test_groupby(self):
        q1 = Qfrom({'a': [1, 2, 1, 2], 'b': [5, 6, 7, 8]})
        q2 = Qfrom({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8]})
        q3 = Qfrom({'a': [1, 1, 2, 2], 'b': [3, 3, 3, 4], 'c': [5, 6, 7, 8]})
        
        q_result1 = Qfrom({'key':[1, 2], 'group': parse.list_to_array([Qfrom({'a':[1, 1], 'b':[5, 7]}), Qfrom({'a':[2, 2], 'b':[6, 8]})])})
        q_result2 = Qfrom({'key':[1, 0], 'group': parse.list_to_array([Qfrom({'a':[1, 3], 'b':[5, 7]}), Qfrom({'a':[2, 4], 'b':[6, 8]})])})
        q_result3 = Qfrom({'key':[1, 2], 'group': parse.list_to_array([Qfrom({'b':[5, 7]}), Qfrom({'b':[6, 8]})])})
        q_result5 = Qfrom({'key':[(1, 3), (2, 3), (2, 4)], 'group': parse.list_to_array([Qfrom({'a': [1, 1], 'b': [3, 3], 'c':[5, 6]}), Qfrom({'a': [2], 'b': [3], 'c':[7]}), Qfrom({'a': [2], 'b': [4], 'c':[8]})])})

        self.assertEqual(q1.groupby('a'), q_result1)
        self.assertEqual(q1.groupby(('a',)), q_result1)
        self.assertEqual(q1.groupby('a', lambda x:x), q_result1)
        self.assertEqual(q2.groupby(func=lambda a: a%2), q_result2)
        self.assertEqual(q3.groupby('a, b'), q_result5)
        self.assertEqual(q3.groupby(('a', 'b')), q_result5)
    # - flatten
    def test_flatten(self):
        q1 = Qfrom({'a': [1, 2], 'b': [[3, 4], [5, 6]]})
        q2 = Qfrom({'a': [1, 2], 'b': [[(3, 4), (5, 6)], [(7, 8), (9, 10)]]})
        q3 = Qfrom({'a': [1, 2], 'b': [3, 4]})

        q_result1 = Qfrom({'a': [1, 1, 2, 2], 'b': [3, 4, 5, 6]})
        q_result2 = Qfrom({'a': [1, 1, 2, 2], 'b': [[3, 4], [3, 4], [5, 6], [5, 6]], 'c': [3, 4, 5, 6]})
        q_result3 = Qfrom({'a': [1, 1, 2, 2], 'b': [(3, 4), (5, 6), (7, 8), (9, 10)]})
        q_result4 = Qfrom({'a': [1, 1, 2, 2], 'b': [3, 3, 4, 4], 'c': [1, 3, 2, 4]})

        self.assertEqual(q1.flatten('b'), q_result1)
        self.assertEqual(q1.flatten('b', 'c'), q_result2)
        self.assertEqual(q2.flatten('b'), q_result3)
        self.assertEqual(q3.flatten('a, b', 'c'), q_result4)
    # - unique
    def test_unique(self):
        q = Qfrom({'a': [1, 2, 2, 3, 3], 'b': [4, 5, 5, 6, 7]})

        q_result1 = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
        q_result2 = Qfrom({'a': [1, 2, 3, 3], 'b': [4, 5, 6, 7]})

        self.assertEqual(q.unique('a'), q_result1)
        self.assertEqual(q.unique(('a',)), q_result1)
        self.assertEqual(q.unique('a, b'), q_result2)
        self.assertEqual(q.unique(('a', 'b')), q_result2)
    # - value_counts
    def test_value_counts(self):
        q = Qfrom({'a': [1, 2, 2, 3, 3], 'b': [4, 5, 5, 6, 7]})

        q_result1 = Qfrom({'value': [1, 2, 3], 'count': [1, 2, 2]})
        q_result2 = Qfrom({'value': [(1, 4), (2, 5), (3, 6), (3, 7)], 'count': [1, 2, 1, 1]})

        self.assertEqual(q.value_counts('a'), q_result1)
        self.assertEqual(q.value_counts(('a',)), q_result1)
        self.assertEqual(q.value_counts('a, b'), q_result2)
        self.assertEqual(q.value_counts(('a', 'b')), q_result2)
    # - agg
    def test_agg(self):
        q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})

        self.assertEqual(q.agg(agg.sum), (6, 15))
        self.assertEqual(q.agg((agg.sum, agg.max)), (6, 6))
        self.assertEqual(q['a'].agg(agg.sum), 6)

    # - join
    def test_join(self):
        q1 = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
        q2 = Qfrom({'a': [1, 2, 3], 'c': [7, 8, 9]})
        q3 = Qfrom({'a': [1, 2, 3, 4], 'b': [4, 5, 6, 10]})
        q4 = Qfrom({'a': [1, 2, 3, 4], 'c': [7, 8, 9, 10]})
        q5 = Qfrom({'a': [1, 2, 3, None], 'c': [7, 8, 9, 10]})
        q6 = Qfrom({'d': [1, 2, 3], 'c': [7, 8, 9]})
        q7 = Qfrom({'a': [1, 1, 2], 'b': [3, 4, 4], 'c': [5, 6, 7]})
        q8 = Qfrom({'a': [1, 2], 'b': [3, 4], 'd': [8, 9]})
        q9 = Qfrom({'e': [1, 2], 'f': [3, 4], 'd': [8, 9]})

        q_result1 = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
        q_result2 = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6], 'd': [1, 2, 3], 'c': [7, 8, 9]})
        q_result3 = Qfrom({'a': [1, 2], 'b': [3, 4], 'c': [5, 7], 'd': [8, 9]})
        q_result4 = Qfrom({'a': [1, 2], 'b': [3, 4], 'c': [5, 7], 'e': [1, 2], 'f': [3, 4], 'd': [8, 9]})

        self.assertEqual(q1.join(q2), q_result1)
        self.assertEqual(q1.join(q2), q_result1)
        self.assertEqual(q3.join(q2), q_result1)
        self.assertEqual(q1.join(q4), q_result1)
        self.assertEqual(q3.join(q5), q_result1)
        self.assertEqual(q1.join(q6, {'a':'d'}), q_result2)
        self.assertEqual(q7.join(q8), q_result3)
        self.assertEqual(q7.join(q9, {'a':'e', 'b':'f'}), q_result4)
    # - join_cross
    def test_join_cross(self):
        q1 = Qfrom({'a': [1, 2]})
        q2 = Qfrom({'b': [3, 4]})

        q_result = Qfrom({'a': [1, 1, 2, 2], 'b': [3, 4, 3, 4]})
        
        self.assertEqual(q1.join_cross(q2), q_result)
    # - join_outer
    def test_join_outer(self):
        q1 = Qfrom({'a': [0, 1, 2, 3], 'b': [0, 4, 5, 6]})
        q2 = Qfrom({'a': [1, 2, 3, 4], 'c': [7, 8, 9, 10]})
        q3 = Qfrom({'a': [0, 1, 2, 3, 4], 'c': [None, 7, 8, 9, 10]})

        q_result = Qfrom({'a': [0, 1, 2, 3, 4], 'b': [0, 4, 5, 6, None], 'c': [None, 7, 8, 9, 10]})

        self.assertEqual(q1.join_outer(q2), q_result)
        self.assertEqual(q1.join_outer(q3), q_result)
    # - join_outer_left
    def test_join_outer_left(self):
        q1 = Qfrom({'a': [0, 1, 2, 3], 'b': [0, 4, 5, 6]})
        q2 = Qfrom({'a': [1, 2, 3, 4], 'c': [7, 8, 9, 10]})

        q_result = Qfrom({'a': [0, 1, 2, 3], 'b': [0, 4, 5, 6], 'c': [None, 7, 8, 9]})

        self.assertEqual(q1.join_outer_left(q2), q_result)
    # - join_outer_right
    def test_join_outer_right(self):
        q1 = Qfrom({'a': [0, 1, 2, 3], 'b': [0, 4, 5, 6]})
        q2 = Qfrom({'a': [1, 2, 3, 4], 'c': [7, 8, 9, 10]})

        q_result = Qfrom({'a': [1, 2, 3, 4], 'b': [4, 5, 6, None], 'c': [7, 8, 9, 10]})

        self.assertEqual(q1.join_outer_right(q2), q_result)
    # - join_id
    def test_join_id(self):
        q1 = Qfrom({'a': [1, 2, 3]})
        q2 = Qfrom({'b': [4, 5, 6]})
        q3 = Qfrom({'b': [4, 5, 6, 7]})

        q_result = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})

        self.assertEqual(q1.join_id(q2), q_result)
        self.assertEqual(q1.join_id(q3), q_result)
    # - join_id_outer
    def test_join_id_outer(self):
        q1 = Qfrom({'a': [1, 2, 3]})
        q2 = Qfrom({'b': [4, 5, 6]})
        q3 = Qfrom({'b': [4, 5, 6, 7]})
        q4 = Qfrom({'a': [1, 2, 3, None]})

        q_result1 = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
        q_result2 = Qfrom({'a': [1, 2, 3, None], 'b': [4, 5, 6, 7]})

        self.assertEqual(q1.join_id_outer(q2), q_result1)
        self.assertEqual(q1.join_id_outer(q3), q_result2)
        self.assertEqual(q4.join_id_outer(q3), q_result2)
    # - join_id_outer_left
    def test_join_id_outer_left(self):
        q1 = Qfrom({'a': [1, 2, 3]})
        q2 = Qfrom({'b': [4, 5, 6]})
        q3 = Qfrom({'b': [4, 5, 6, 7]})
        q4 = Qfrom({'a': [1, 2, 3, 4]})
        q5 = Qfrom({'b': [5, 6, 7]})

        q_result1 = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
        q_result2 = Qfrom({'a': [1, 2, 3, 4], 'b': [5, 6, 7, None]})

        self.assertEqual(q1.join_id_outer_left(q2), q_result1)
        self.assertEqual(q1.join_id_outer_left(q3), q_result1)
        self.assertEqual(q4.join_id_outer_left(q5), q_result2)
    # - join_id_outer_right
    def test_join_id_outer_right(self):
        q1 = Qfrom({'a': [1, 2, 3]})
        q2 = Qfrom({'b': [4, 5, 6]})
        q3 = Qfrom({'b': [4, 5, 6, 7]})
        q4 = Qfrom({'a': [1, 2, 3, 4]})
        q5 = Qfrom({'b': [5, 6, 7]})

        q_result1 = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
        q_result2 = Qfrom({'a': [1, 2, 3, None], 'b': [4, 5, 6, 7]})
        q_result3 = Qfrom({'a': [1, 2, 3], 'b': [5, 6, 7]})

        self.assertEqual(q1.join_id_outer_right(q2), q_result1)
        self.assertEqual(q1.join_id_outer_right(q3), q_result2)
        self.assertEqual(q4.join_id_outer_right(q5), q_result3)

    # - concat
    def test_concat(self):
        q1 = Qfrom({'a': [1, 2], 'b': [5, 6]})
        q2 = Qfrom({'a': [3, 4], 'b': [7, 8]})
        q3 = Qfrom({'c': [9, 10]})
        q4 = Qfrom({'d': [11, 12]})
        q5 = Qfrom({'a': [3, 4], 'c': [9, 10]})
        q6 = Qfrom({'a': [2, 3, 4], 'b': [6, 7, 8]})
        q7 = Qfrom({'a': [9, 10], 'b': [11, 12]})

        q_result1 = Qfrom({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8]})
        q_result2 = Qfrom()
        q_result3 = Qfrom({'a': [1, 2, 3, 4]})
        q_result4 = Qfrom({'a': [1, 2, 2, 3, 4], 'b': [5, 6, 6, 7, 8]})
        q_result5 = Qfrom({'a': [1, 2, 3, 4, 9, 10], 'b': [5, 6, 7, 8, 11, 12]})

        self.assertEqual(q1.concat(q2), q_result1)
        self.assertEqual(q3.concat(q4), q_result2)
        self.assertEqual(q1.concat(q5), q_result3)
        self.assertEqual(q1.concat(q6), q_result4)
        self.assertEqual(q1.concat([q2, q7]), q_result5)
    # - concat_outer
    def test_concat_outer(self):
        q1 = Qfrom({'a': [1, 2], 'b': [5, 6]})
        q2 = Qfrom({'a': [3, 4], 'b': [7, 8]})
        q3 = Qfrom({'c': [9, 10]})
        q4 = Qfrom({'d': [11, 12]})
        q5 = Qfrom({'a': [3, 4], 'c': [9, 10]})
        q6 = Qfrom({'a': [2, 3, 4], 'b': [6, 7, 8]})

        q_result1 = Qfrom({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8]})
        q_result2 = Qfrom({'c': [9, 10, None, None], 'd': [None, None, 11, 12]})
        q_result3 = Qfrom({'a': [1, 2, 3, 4], 'b': [5, 6, None, None], 'c': [None, None, 9, 10]})
        q_result4 = Qfrom({'a': [1, 2, 2, 3, 4], 'b': [5, 6, 6, 7, 8]})

        self.assertEqual(q1.concat_outer(q2), q_result1)
        self.assertEqual(Qfrom().concat_outer([q1,q2]), q_result1)
        self.assertEqual(q3.concat_outer(q4), q_result2)
        self.assertEqual(q1.concat_outer(q5), q_result3)
        self.assertEqual(q1.concat_outer(q6), q_result4)
    # - concat_outer_left
    def test_concat_outer_left(self):
        q1 = Qfrom({'a': [1, 2], 'b': [5, 6]})
        q2 = Qfrom({'a': [3, 4], 'b': [7, 8]})
        q3 = Qfrom({'c': [9, 10]})
        q4 = Qfrom({'d': [11, 12]})
        q5 = Qfrom({'a': [3, 4], 'c': [9, 10]})
        q6 = Qfrom({'a': [2, 3, 4], 'b': [6, 7, 8]})

        q_result1 = Qfrom({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8]})
        q_result2 = Qfrom({'c': [9, 10, None, None]})
        q_result3 = Qfrom({'a': [1, 2, 3, 4], 'b': [5, 6, None, None]})
        q_result4 = Qfrom({'a': [1, 2, 2, 3, 4], 'b': [5, 6, 6, 7, 8]})

        self.assertEqual(q1.concat_outer_left(q2), q_result1)
        self.assertEqual(q3.concat_outer_left(q4), q_result2)
        self.assertEqual(q1.concat_outer_left(q5), q_result3)
        self.assertEqual(q1.concat_outer_left(q6), q_result4)
    # - concat_outer_right
    def test_concat_outer_right(self):
        q1 = Qfrom({'a': [1, 2], 'b': [5, 6]})
        q2 = Qfrom({'a': [3, 4], 'b': [7, 8]})
        q3 = Qfrom({'c': [9, 10]})
        q4 = Qfrom({'d': [11, 12]})
        q5 = Qfrom({'a': [3, 4], 'c': [9, 10]})
        q6 = Qfrom({'a': [2, 3, 4], 'b': [6, 7, 8]})

        q_result1 = Qfrom({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8]})
        q_result2 = Qfrom({'d': [None, None, 11, 12]})
        q_result3 = Qfrom({'a': [1, 2, 3, 4], 'c': [None, None, 9, 10]})
        q_result4 = Qfrom({'a': [1, 2, 2, 3, 4], 'b': [5, 6, 6, 7, 8]})

        self.assertEqual(q1.concat_outer_right(q2), q_result1)
        self.assertEqual(q3.concat_outer_right(q4), q_result2)
        self.assertEqual(q1.concat_outer_right(q5), q_result3)
        self.assertEqual(q1.concat_outer_right(q6), q_result4)

    # - calc
    # - call