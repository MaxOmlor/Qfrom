import unittest
from Qfrom_tab_mod import predicatestr_to_funcstr, selectarg_to_funcstr, list_to_array, Qfrom
import pandas
import numpy as np


class TestTransFuncStr(unittest.TestCase):
    '''def test_get_func_type(self):
        self.assertEqual('a', 'select')
        self.assertEqual('a, b', 'select')
        self.assertEqual('a, b as c', 'select')
        self.assertEqual('-a', 'select')
        self.assertEqual('a, b<3 as c', 'select')

        self.assertEqual('a == "a"', 'predicate')
        self.assertEqual('a != "a"', 'predicate')
        self.assertEqual('a<b', 'predicate')
        self.assertEqual('a is None', 'predicate')
        self.assertEqual('a is not None', 'predicate')
        self.assertEqual('a and b', 'predicate')
        self.assertEqual('a or b', 'predicate')
        self.assertEqual('a == 1 or b == 1', 'predicate')'''
    def test_predicatestr_to_funcstr(self):
        keys = ('a', 'b', 'c')
        self.assertEqual(predicatestr_to_funcstr('a', keys), 'lambda a: a')
        self.assertTrue(predicatestr_to_funcstr('a, b', keys) in ['lambda a, b: (a) and (b)', 'lambda b, a: (a) and (b)'])
        self.assertEqual(predicatestr_to_funcstr('a == "a"', keys), 'lambda a: a == "a"')
        self.assertEqual(predicatestr_to_funcstr('"a" == a', keys), 'lambda a: "a" == a')
        self.assertEqual(predicatestr_to_funcstr('a != "a"', keys), 'lambda a: a != "a"')
        self.assertEqual(predicatestr_to_funcstr('a in ["a"]', keys), 'lambda a: a in ["a"]')
        self.assertEqual(predicatestr_to_funcstr('a is None', keys), 'lambda a: a is None')
        self.assertEqual(predicatestr_to_funcstr('a is not None', keys), 'lambda a: a is not None')
        self.assertEqual(predicatestr_to_funcstr('a==1 or a==2', keys), 'lambda a: a==1 or a==2')
        self.assertTrue(predicatestr_to_funcstr('a==1 and b==2', keys) in ['lambda a, b: a==1 and b==2', 'lambda b, a: a==1 and b==2'])
        self.assertEqual(predicatestr_to_funcstr('2<a<5', keys), 'lambda a: 2<a<5')
        self.assertEqual(predicatestr_to_funcstr('2<=a<=5', keys), 'lambda a: 2<=a<=5')
        self.assertTrue(predicatestr_to_funcstr('a or b', keys) in ['lambda a, b: a or b', 'lambda b, a: a or b'])
        self.assertEqual(predicatestr_to_funcstr('not not a', keys), 'lambda a: not not a')
        self.assertTrue(predicatestr_to_funcstr('a%2, b==1', keys) in ['lambda a, b: (a%2) and (b==1)', 'lambda b, a: (a%2) and (b==1)'])
    def test_selectstr_to_funcstr(self):
        keys = ('a', 'b', 'c')
        self.assertEqual(selectarg_to_funcstr('a', keys), (
            None,
            None,
            'lambda x: {"a": x["a"]}'))
        self.assertEqual(selectarg_to_funcstr(('a',), keys), (
            None,
            None,
            'lambda x: {"a": x["a"]}'))
        self.assertEqual(selectarg_to_funcstr('a, b', keys), (
            None,
            None,
            'lambda x: {"a": x["a"], "b": x["b"]}'))
        self.assertEqual(selectarg_to_funcstr(('a', 'b'), keys), (
            None,
            None,
            'lambda x: {"a": x["a"], "b": x["b"]}'))
        self.assertEqual(selectarg_to_funcstr('a, b as c', keys), (
            None,
            None,
            'lambda x: {"a": x["a"], "c": x["b"]}'))
        self.assertEqual(selectarg_to_funcstr(('a', 'b as c'), keys), (
            None,
            None,
            'lambda x: {"a": x["a"], "c": x["b"]}'))
        self.assertEqual(selectarg_to_funcstr('-a', keys), (
            None,
            None,
            'lambda x: {key:value for key, value in x.items() if key != "a"}'))
        self.assertEqual(selectarg_to_funcstr('-a, -b', keys), (
            None,
            None,
            'lambda x: {key:value for key, value in x.items() if key not in ["a", "b"]}'))
        self.assertEqual(selectarg_to_funcstr(('-a', '-b'), keys), (
            None,
            None,
            'lambda x: {key:value for key, value in x.items() if key not in ["a", "b"]}'))
        self.assertEqual(selectarg_to_funcstr('a, b < 3', keys), (
            'lambda b: b < 3',
            (0,),
            'lambda x: {"a": x["a"], 0: x[0]}'))
        self.assertEqual(selectarg_to_funcstr('a, b<3', keys), (
            'lambda b: b<3',
            (0,),
            'lambda x: {"a": x["a"], 0: x[0]}'))
        self.assertEqual(selectarg_to_funcstr(('a', 'b<3'), keys), (
            'lambda b: b<3',
            (0,),
            'lambda x: {"a": x["a"], 0: x[0]}'))
        self.assertEqual(selectarg_to_funcstr('a, b<3 as c', keys), (
            'lambda b: b<3',
            ('c',),
            'lambda x: {"a": x["a"], "c": x["c"]}'))
        self.assertEqual(selectarg_to_funcstr('a, b<3 as c, b>=3 as d', keys), (
            'lambda b: (b<3, b>=3)',
            ('c', 'd'),
            'lambda x: {"a": x["a"], "c": x["c"], "d": x["d"]}'))

'''class TestTransFunc(unittest.TestCase):
    def test_func(self):

        with self.assertRaises(SyntaxError):
            trans_func(None)'''



class TestQfromClass(unittest.TestCase):

    ## import_list
    ## import_dict
    ## (import_set)
    ## (import_array)
    ## (import_mtx)
    ## import_dataframe
    ## import_csv
    ## (import_json)

    ## eq
    ## str
    ## repr

    ## append
    ## setitem
    ## getitem
    ## contains
    ## iter
    
    ## rename
    ## orderby
    ## orderby_pn
    ## (shuffle)
    ## first
    ## columns
    ## (stats)
    ## copy
    ## (deep_copy)

    ## where
    ## select
    ## select_pn -> pass None values
    ## select_join -> map-op which gets joined directly
    ## select_join_pn
    ## (normalize)
    
    ## join
    ## join_cross
    ## join_outer
    ## join_outer_left
    ## join_outer_right
    ## join_id
    ## join_id_outer
    ## join_id_outer_left
    ## join_id_outer_right

    ## (union)
    ## (intersect)
    ## (difference)
    ## (symmetric_difference)
    ## (partition)
    ## concat
    ## concat_outer
    ## concat_outer_left
    ## concat_outer_right

    ## groupby
    ## groupby_pn
    ## flatten
    ## flatten_pn
    ## flatten_join
    ## flatten_join_pn
    ## unique
    
    ## agg
    ## agg_pairs
    ## any
    ## all
    ## min
    ## min_id
    ## min_item
    ## max
    ## max_id
    ## max_item
    ## sum
    ## mean
    ## median
    ## var
    ## len
    ## size

    ## calc
    ## call

    ## plot
    ## plot_bar
    ## plot_hist
    ## plot_box
    ## plot_scatter

    ## tolist
    ## (toset)
    ## todict
    ## (toarray)
    ## (tomtx)
    ## todf
    ## tocsv
    ## (tocsvfile)
    ## (tojson)
    ## (tojsonfile)
    ## __array__





    ## import_list
    def test_init_list(self):
        self.assertEqual(Qfrom([1, 2, 3]).tolist(), [1, 2, 3])

        l = [
            (1, 4),
            (2, 5),
            (3, 6),
            ]
        q_result = Qfrom({0: [1, 2, 3], 1: [4, 5, 6]})

        self.assertEqual(Qfrom(l), q_result)
        self.assertEqual(Qfrom(l).tolist(), l)
    ## import_dict
    '''def test_init_dict(self):
        d = {'a': [1, 2, 3], 'b': [4, 5, 6]}

        self.assertEqual(Qfrom(d).todict(), d)'''
    ## (import_set)
    ## (import_array)
    ## (import_mtx)
    ## import_dataframe
    ## import_csv
    ## (import_json)
    
    ## eq
    def test_eq(self):
        self.assertEqual(Qfrom([1, 2, 3]), Qfrom([1, 2, 3]))
        self.assertNotEqual(Qfrom([1, 2, 3]), Qfrom([3, 2, 1]))

        q1 = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
        q2 = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
        q3 = Qfrom({'a': [1, 2, 3], 'b': [6, 5, 4]})
        q4 = Qfrom({'a': [1, 2, 3]})
        
        self.assertEqual(q1, q1)
        self.assertEqual(q1, q2)
        self.assertNotEqual(q1, q3)
        self.assertNotEqual(q1, q4)
    ## str
    def test_str(self):
        q1 = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
        qs1 = 'Qfrom\na\tb\n1\t4\n2\t5\n3\t6'''
        q2 = Qfrom()
        qs2 = 'Qfrom\nempty'

        self.assertEqual(str(q1), qs1)
        self.assertEqual(str(q2), qs2)
    ## repr
    
    ## append
    def test_append(self):
        q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
        q_result = Qfrom({'a': [1, 2, 3, 7], 'b': [4, 5, 6, 8]})

        self.assertEqual(q.append((7, 8)), q_result)
        self.assertEqual(q.append({'a': 7, 'b':8}), q_result)
    ## setitem
    def test_setitem(self):
        q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
        q_result = Qfrom({'a': [1, 7, 3], 'b': [4, 8, 6]})
        
        q1 = q.copy()
        q1[1] = (7, 8)
        q2 = q.copy()
        q2[1] = {'a':7, 'b':8}
        q3 = q.copy()
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
        
        q1_1 = q.copy()
        q1_1['c'] = [7, 8, 9]
        #q1_2 = q.copy()
        #q1_2['c'] = np.array([7, 8, 9])
        q1_3 = q.copy()
        q1_3[('c', )] = [7, 8, 9]
        q1_4 = q.copy()
        q1_4['c'] = {'c': [7, 8, 9]}
        q1_5 = q.copy()
        q1_5['c'] = [(7,), (8,), (9,)]
        #input Qfrom

        q2_1 = q.copy()
        q2_1['b'] = [7, 8, 9]

        q3_1 = q.copy()
        q3_1['a, b'] = [(4, 1), (5, 2), (6, 3)]
        q3_2 = q.copy()
        q3_2[('a', 'b')] = [(4, 1), (5, 2), (6, 3)]
        #q3_3 = q.copy()
        #q3_3[('a', 'b')] = np.array([[4, 1], [5, 2], [6, 3]])
        q3_4 = q.copy()
        q3_4['a, b'] = {'a': [4, 5, 6], 'b': [1, 2, 3]}

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
    #def test_setitem_errors(self):
        # set to short/to long list
    ## getitem
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
    '''def test_getitem_col(self):
        q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})

        result1 = np.array([1, 2, 3])
        result3 = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6], 'i': [0, 1, 2]})

        self.assertEqual(q['a'], result1)
        self.assertEqual(q[('a',)], result1)
        self.assertEqual(q['a, b'], q)
        self.assertEqual(q[('a', 'b')], q)
        self.assertEqual(q['a, b, i'], result3)
        self.assertEqual(q[('a', 'b', 'i')], result3)'''
    ## contains
    def test_contains(self):
        q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})

        self.assertTrue((2, 5) in q)
        self.assertTrue({'a': 2, 'b':5} in q)
        self.assertFalse((0, 0) in q)
        self.assertFalse({'a': 0, 'b':0} in q)
        self.assertFalse((2, 4) in q)
        self.assertFalse({'a': 2, 'b':4} in q)
    ## iter
    def test_iter(self):
        self.assertEqual([item for item in Qfrom([1, 2, 3])], [1, 2, 3])
        self.assertEqual(next(iter(Qfrom([1, 2, 3]))), 1)

        q1 = Qfrom({'a': ['a', 'b', 'c'], 'b': ['d', 'e', 'f']})
        self.assertEqual([a+b for a,b in q1], ['ad', 'be', 'cf'])

        q2 = Qfrom().concat_outer(q1)
        self.assertEqual([a+b for a,b in q2], ['ad', 'be', 'cf'])
    
    ## rename
    def test_rename(self):
        q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
        q_result1 = Qfrom({'c': [1, 2, 3], 'b': [4, 5, 6]})
        q_result2 = Qfrom({'c': [1, 2, 3], 'd': [4, 5, 6]})

        self.assertEqual(q.rename({'a':'c'}), q_result1)
        self.assertEqual(q.rename('a as c'), q_result1)
        self.assertEqual(q.rename({'a':'c', 'b':'d'}), q_result2)
        self.assertEqual(q.rename('a as c, b as d'), q_result2)
    ## orderby
    def test_orderby(self):
        q1 = Qfrom('''
        a,b,c
        3,4,1
        2,3,2
        2,2,3
        1,1,4
        ''')
        q2 = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
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
        2,2,3
        2,3,2
        1,1,4
        ''')
        q_result3 = Qfrom('''
        a,b,c
        1,1,4
        2,2,3
        2,3,2
        3,4,1
        ''')
        q_result4 = Qfrom({'a': [3, 2, 1], 'b': [6, 5, 4]})
        q_result5 = Qfrom({'b': [6, 5, 4]})


        self.assertEqual(q1.orderby('a'), q_result1)
        self.assertEqual(q1.orderby('a', reverse=True), q_result2)
        self.assertEqual(q1.orderby('a, b'), q_result3)
        self.assertEqual(q1.orderby('a, b', lambda x, y: (x, y)), q_result3)
        self.assertEqual(q1.orderby(('a', 'b')), q_result3)
        self.assertEqual(q2.orderby('3-a'), q_result4)
        self.assertEqual(q2.orderby(lambda a: 3-a), q_result4)
        self.assertEqual(q2.orderby('a', lambda x: 3-x), q_result4)
        self.assertEqual(q2.orderby('3-a', 'b'), q_result5)
        self.assertEqual(q2.orderby('a', lambda x: 3-x, 'b'), q_result5)
    ## orderby_pn
    ## (shuffle)
    ## first
    def test_first(self):
        q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
        
        self.assertEqual(q.first(), (1, 4))
    def test_first_predicate(self):
        q = Qfrom({'a': [None, 2, 3], 'b': [4, None, 6]})

        self.assertEqual(q.first('a'), (2,None))
        self.assertEqual(q.first(lambda a: a is not None), (2,None))
        self.assertEqual(q.first('a', lambda x: x is not None), (2,None))
        self.assertEqual(q.first(('a',), lambda x: x is not None), (2,None))
        self.assertEqual(q.first('a, b'), (3, 6))
        self.assertEqual(q.first(('a', 'b'), lambda x,y: x is not None and y is not None), (3, 6))
    ## columns
    def test_columns(self):
        self.assertEqual(Qfrom([1, 2, 3]).columns(), (0,))

        q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})

        self.assertEqual(q.columns(), ('a', 'b'))
    ## (stats)
    ## copy
    def test_copy(self):
        d = {'a': [1, 2, 3], 'b': [4, 5, 6]}
        d_result = {'a': [7, 8, 9], 'b': [4, 5, 6]}

        q1 = Qfrom(dict(d))
        q2 = q1.copy()
        q2['a'] = [7, 8, 9]

        self.assertEqual({k:list(v) for k,v in q1.todict().items()}, d)
        self.assertEqual({k:list(v) for k,v in q2.todict().items()}, d_result)
    ## (deep_copy)

    ## where
    def test_where(self):
        q1 = Qfrom({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
        q2 = Qfrom({'a': [None, 2, 3], 'b': [4, None, 6]})
        q3 = Qfrom({'a': [None, 2, 3], 'b': [4, None, 6], 'c': [7, 8, None]})

        q_result1 = Qfrom({'a': [1, 2], 'b': [5, 4]})
        q_result2 = Qfrom({'a': [2, 3, 4], 'b': [4, 3, 2]})
        q_result3 = Qfrom({'a': [2, 3], 'b': [None, 6]})
        q_result4 = Qfrom({'a': [3], 'b': [6]})
        q_result5 = Qfrom({'a': [2, 3], 'b': [None, 6], 'c': [8, None]})

        self.assertEqual(q1.where(lambda a: a<3), q_result1)
        self.assertEqual(q1.where('a<3'), q_result1)
        self.assertEqual(q1.where('a<5 and b<5'), q_result2)
        self.assertEqual(q1.where('a<5, b<5'), q_result2)
        self.assertEqual(q1.where(('a<5', 'b<5')), q_result2)
        self.assertEqual(q2.where(lambda a: a), q_result3)
        self.assertEqual(q2.where('a', lambda x: x), q_result3)
        self.assertEqual(q2.where(('a',), lambda x: x), q_result3)
        self.assertEqual(q2.where('a'), q_result3)
        self.assertEqual(q2.where(('a',)), q_result3)
        #self.assertEqual(q2.where(lambda a,b: (a, b)), q_result4)
        self.assertEqual(q2.where('a, b'), q_result4)
        self.assertEqual(q2.where(('a', 'b')), q_result4)
        self.assertEqual(q2.where(lambda a, b: a and b), q_result4)
        self.assertEqual(q3.where('a, b or c'), q_result5)
        self.assertEqual(q3.where(('a', 'b or c')), q_result5)
    ## select
    def test_select(self):
        q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
        q_result1 = Qfrom({'a': [1, 2, 3]})
        q_result2 = Qfrom({'a': [1, 2, 3], 'c': [7, 8, 9]})
        q_result3 = Qfrom({'a': [1, 2, 3], 'd': [4, 5, 6]})
        q_result4 = Qfrom({'b': [4, 5, 6], 'a': [1, 2, 3]})
        q_result5 = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
        q_result6 = Qfrom({'i': [0, 1, 2]})
        q_result7 = Qfrom({'i': [0, 1, 2], 'a': [1, 2, 3]})

        self.assertEqual(q.select('a'), q_result1)
        self.assertEqual(q.select(('a',)), q_result1)
        self.assertEqual(q.select('-b'), q_result2)
        self.assertEqual(q.select(('-b', )), q_result2)
        self.assertEqual(q.select('a, b as d'), q_result3)
        self.assertEqual(q.select(('a', 'b as d')), q_result3)
        self.assertEqual(q.select('b, a'), q_result4)
        self.assertEqual(q.select(('b', 'a')), q_result4)
        self.assertEqual(q.select(('a', 'b')), q_result5)
        self.assertEqual(q.select('i'), q_result6)
        self.assertEqual(q.select('i, a'), q_result7)
        self.assertEqual(q.select(('i', 'a')), q_result7)
    def test_select_predicate(self):
        q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
        q_result1 = Qfrom({0: [False, True, True]})
        q_result2 = Qfrom({'p': [False, True, True]})
        q_result3 = Qfrom({'p': [False, True, True], 'a': [1, 2, 3], 'b': [4, 5, 6]})

        self.assertEqual(q.select('a>1'), q_result1)
        self.assertEqual(q.select(('a>1',)), q_result1)
        self.assertEqual(q.select('a>1 as p'), q_result2)
        self.assertEqual(q.select(('a>1 as p',)), q_result2)
        self.assertEqual(q.select(('a>1 as p', 'a', 'b')), q_result3)
    def test_select_func(self):
        q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
        q_result1 = Qfrom({0: [5, 7, 9]})
        q_result2 = Qfrom({'c': [5, 7, 9]})
        q_result3 = Qfrom({0: [1, 2, 3]})
        q_result4 = Qfrom({0: [0, 1, 2]})
        q_result5 = Qfrom({0: [0, 1, 2], 1: [1, 2, 3], 2: [4, 5, 6]})
        q_result6 = Qfrom({'i': [0, 1, 2], 'a': [1, 2, 3], 'b': [4, 5, 6]})

        self.assertEqual(q.select(lambda a,b: a+b), q_result1)
        self.assertEqual(q.select('a, b', lambda x,y: x+y), q_result1)
        self.assertEqual(q.select(('a', 'b'), lambda x,y: x+y), q_result1)
        self.assertEqual(q.select('a, b', lambda x,y: x+y, 'c'), q_result2)
        self.assertEqual(q.select('a, b', lambda x,y: x+y, ('c',)), q_result2)
        self.assertEqual(q.select(lambda a: a), q_result3)
        self.assertEqual(q.select(lambda i: i), q_result4)
        self.assertEqual(q.select(lambda i,a,b: (i,a,b)), q_result5)
        self.assertEqual(q.select(lambda i,a,b: (i,a,b), ('i', 'a', 'b')), q_result6)
        self.assertEqual(q.select(lambda i,a,b: (i,a,b), 'i, a, b'), q_result6)
    ## select_pn -> pass None values
    def test_select_pn(self):
        q = Qfrom({'a': [1, None, 3], 'b': [4, 5, 6]})
        q_result1 = Qfrom({0: [1, None, 9]})
        q_result2 = Qfrom({0: [5, None, 9]})

        self.assertEqual(q.select_pn(lambda a:a**2), q_result1)
        self.assertEqual(q.select_pn(lambda a,b:a+b), q_result2)
    ## select_join -> map-op which gets joined directly
    def test_select_join(self):
        q = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})

        q_result1 = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6], 0:[0, 1, 2]})
        q_result2 = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6], 'i':[0, 1, 2]})
        q_result3 = Qfrom({'b': [4, 5, 6], 'a': [7, 8, 9]})
        q_result4 = Qfrom({'b': [4, 5, 6], 'a': [7, 8, 9], 'i':[0, 1, 2]})
        q_result5 = Qfrom({'b': [4, 5, 6], 'a': [False, True, True]})

        self.assertEqual(q.select_join(lambda i: i), q_result1)
        self.assertEqual(q.select_join(lambda i: i, ('i',)), q_result2)
        self.assertEqual(q.select_join(lambda i: i, 'i'), q_result2)
        self.assertEqual(q.select_join('i'), q_result2)
        self.assertEqual(q.select_join(('i',)), q_result2)
        self.assertEqual(q.select_join(lambda a: a+6, ('a',)), q_result3)
        self.assertEqual(q.select_join(lambda a: a+6, 'a'), q_result3)
        self.assertEqual(q.select_join('a', lambda x: x+6, 'a'), q_result3)
        self.assertEqual(q.select_join(('a',), lambda x: x+6, 'a'), q_result3)
        self.assertEqual(q.select_join(lambda a, i: (a+6, i), ('a', 'i')), q_result4)
        self.assertEqual(q.select_join(lambda a, i: (a+6, i), 'a, i'), q_result4)
        self.assertEqual(q.select_join('a>1 as a'), q_result5)
    ## select_join_pn
    def test_select_join_pn(self):
        q1 = Qfrom({'a': [None, 2, 3], 'b': [4, 5, 6]})
        q2 = Qfrom({'a': [None, 2, 3], 'b': [4, None, 6]})
        q_result1 = Qfrom({'a': [None, 2, 3], 'b': [4, 5, 6], 'c': [None, 7, 9]})
        q_result2 = Qfrom({'a': [None, 2, 3], 'b': [4, None, 6], 'c': [None, None, 9]})

        self.assertEqual(q1.select_join_pn(lambda a,b:a+b, 'c'), q_result1)
        self.assertEqual(q2.select_join_pn(lambda a,b:a+b, 'c'), q_result2)
    ## (normalize)
    
    ## join
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
    ## join_cross
    def test_join_cross(self):
        q1 = Qfrom({'a': [1, 2]})
        q2 = Qfrom({'b': [3, 4]})

        q_result = Qfrom({'a': [1, 1, 2, 2], 'b': [3, 4, 3, 4]})
        
        self.assertEqual(q1.join_cross(q2), q_result)
    ## join_outer
    def test_join_outer(self):
        q1 = Qfrom({'a': [0, 1, 2, 3], 'b': [0, 4, 5, 6]})
        q2 = Qfrom({'a': [1, 2, 3, 4], 'c': [7, 8, 9, 10]})
        q3 = Qfrom({'a': [0, 1, 2, 3, 4], 'c': [None, 7, 8, 9, 10]})

        q_result = Qfrom({'a': [0, 1, 2, 3, 4], 'b': [0, 4, 5, 6, None], 'c': [None, 7, 8, 9, 10]})

        self.assertEqual(q1.join_outer(q2), q_result)
        self.assertEqual(q1.join_outer(q3), q_result)
    ## join_outer_left
    def test_join_outer_left(self):
        q1 = Qfrom({'a': [0, 1, 2, 3], 'b': [0, 4, 5, 6]})
        q2 = Qfrom({'a': [1, 2, 3, 4], 'c': [7, 8, 9, 10]})

        q_result = Qfrom({'a': [0, 1, 2, 3], 'b': [0, 4, 5, 6], 'c': [None, 7, 8, 9]})

        self.assertEqual(q1.join_outer_left(q2), q_result)
    ## join_outer_right
    def test_join_outer_right(self):
        q1 = Qfrom({'a': [0, 1, 2, 3], 'b': [0, 4, 5, 6]})
        q2 = Qfrom({'a': [1, 2, 3, 4], 'c': [7, 8, 9, 10]})

        q_result = Qfrom({'a': [1, 2, 3, 4], 'b': [4, 5, 6, None], 'c': [7, 8, 9, 10]})

        self.assertEqual(q1.join_outer_right(q2), q_result)
    ## join_id
    def test_join_id(self):
        q1 = Qfrom({'a': [1, 2, 3]})
        q2 = Qfrom({'b': [4, 5, 6]})
        q3 = Qfrom({'b': [4, 5, 6, 7]})

        q_result = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})

        self.assertEqual(q1.join_id(q2), q_result)
        self.assertEqual(q1.join_id(q3), q_result)
    ## join_id_outer
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
    ## join_id_outer_left
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
    ## join_id_outer_right
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

    ## (union)
    '''def test_union(self):
        q1 = Qfrom({'a': [1, 2], 'b': [5, 6]})
        q2 = Qfrom({'a': [3, 4], 'b': [7, 8]})
        q3 = Qfrom({'c': [9, 10]})
        q4 = Qfrom({'d': [11, 12]})
        q5 = Qfrom({'a': [2, 3, 4], 'b': [6, 7, 8]})

        q_result1 = Qfrom({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8]})
        q_result2 = Qfrom({'c': [9, 10, None, None], 'd': [None, None, 11, 12]})

        self.assertEqual(q1.union(q2), q_result1)
        self.assertEqual(q3.union(q4), q_result2)
        self.assertEqual(q1.union(q5), q_result1)
    ## (intersect)
    def test_intersect(self):
        q1 = Qfrom({'a': [1, 2, 3], 'b': [5, 6, 7]})
        q2 = Qfrom({'a': [2, 3, 4], 'b': [6, 7, 8]})

        q_result = Qfrom({'a': [2, 3], 'b': [6, 7]})

        self.assertEqual(q1.intersect(q2), q_result)
    ## (difference)
    def test_difference(self):
        q1 = Qfrom({'a': [1, 2, 3], 'b': [5, 6, 7]})
        q2 = Qfrom({'a': [2, 3, 4], 'b': [6, 7, 8]})

        q_result = Qfrom({'a': [1], 'b': [5]})

        self.assertEqual(q1.intersect(q2), q_result)
    ## (symmetric_difference)
    def test_symmetric_difference(self):
        q1 = Qfrom({'a': [1, 2, 3], 'b': [5, 6, 7]})
        q2 = Qfrom({'a': [2, 3, 4], 'b': [6, 7, 8]})

        q_result = Qfrom({'a': [1, 4], 'b': [5, 8]})

        self.assertEqual(q1.intersect(q2), q_result)
    ## (partition)
    '''
    ## concat
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
    ## concat_outer
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
    ## concat_outer_left
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
    ## concat_outer_right
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


    ## groupby
    def test_groupby(self):
        q1 = Qfrom({'a': [1, 2, 1, 2], 'b': [5, 6, 7, 8]})
        q2 = Qfrom({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8]})
        q3 = Qfrom({'a': [1, 1, 2, 2], 'b': [3, 3, 3, 4], 'c': [5, 6, 7, 8]})
        
        q_result1 = Qfrom({'key':[1, 2], 'group':list_to_array([Qfrom({'a':[1, 1], 'b':[5, 7]}), Qfrom({'a':[2, 2], 'b':[6, 8]})])})
        q_result2 = Qfrom({'key':[1, 0], 'group':list_to_array([Qfrom({'a':[1, 3], 'b':[5, 7]}), Qfrom({'a':[2, 4], 'b':[6, 8]})])})
        q_result3 = Qfrom({'key':[1, 2], 'group':list_to_array([Qfrom({'b':[5, 7]}), Qfrom({'b':[6, 8]})])})
        q_result5 = Qfrom({'key':[(1, 3), (2, 3), (2, 4)], 'group':list_to_array([Qfrom({'c':[5, 6]}), Qfrom({'c':[7]}), Qfrom({'c':[8]})])})

        self.assertEqual(q1.groupby('a'), q_result1)
        self.assertEqual(q1.groupby(('a',)), q_result1)
        self.assertEqual(q2.groupby(lambda a: a%2), q_result2)
        #self.assertEqual(q1.groupby('a', lambda b: b), q_result3)
        self.assertEqual(q1.groupby('a', 'b'), q_result3)
        self.assertEqual(q1.groupby('a', ('b',)), q_result3)
        self.assertEqual(q1.groupby('a', lambda x:x, 'b'), q_result3)
        self.assertEqual(q3.groupby('a, b', 'c'), q_result5)
        self.assertEqual(q3.groupby(('a', 'b'), 'c'), q_result5)
    ## groupby_pn
    def test_groupby_pn(self):
        q = Qfrom({'a': [1, 2, 3, 4, None], 'b': [5, 6, 7, 8, 9]})
        
        q_result = Qfrom({'key':[1, 0, None], 'group':list_to_array([Qfrom({'a':[1, 3], 'b':[5, 7]}), Qfrom({'a':[2, 4], 'b':[6, 8]}), Qfrom({'a':[None], 'b':[9]})])})
        
        self.assertEqual(q.groupby_pn(lambda a: a%2), q_result)
    ## flatten
    def test_flatten(self):
        q1 = Qfrom({'a': [1, 2], 'b': [[3, 4], [5, 6]]})
        q2 = Qfrom({'a': [1, 2], 'b': [[(3, 4), (5, 6)], [(7, 8), (9, 10)]]})
        q3 = Qfrom({'a': [1, 2]})

        q_result1 = Qfrom({0: [3, 4, 5, 6]})
        q_result2 = Qfrom({'c': [3, 4, 5, 6]})
        q_result3 = Qfrom({0: [3, 5, 7, 9], 1: [4, 6, 8, 10]})
        q_result4 = Qfrom({'c': [3, 5, 7, 9], 'd': [4, 6, 8, 10]})
        q_result5 = Qfrom({'c': [3, 4, 3, 4]})

        self.assertEqual(q1.flatten('b'), q_result1)
        self.assertEqual(q1.flatten(('b',)), q_result1)
        self.assertEqual(q1.flatten(lambda b: b), q_result1)
        self.assertEqual(q1.flatten('b', 'c'), q_result2)
        self.assertEqual(q1.flatten('b', ('c',)), q_result2)
        self.assertEqual(q1.flatten('b', lambda x: x, 'c'), q_result2)
        self.assertEqual(q2.flatten('b'), q_result3)
        self.assertEqual(q2.flatten('b', 'c, d'), q_result4)
        self.assertEqual(q2.flatten('b', ('c', 'd')), q_result4)
        self.assertEqual(q3.flatten(lambda a: [3, 4], 'c'), q_result5)
    ## flatten_pn
    ## flatten_join
    def test_flatten_join(self):
        q = Qfrom({'a':[1, 2], 'b':[[3, 4], [5, 6]]})

        q_result1 = Qfrom({'a': [1, 1, 2, 2], 'b': [[3, 4], [3, 4], [5, 6], [5, 6]], 0: [3, 4, 5, 6]})
        q_result2 = Qfrom({'a': [1, 1, 2, 2], 'b': [3, 4, 5, 6]})

        self.assertEqual(q.flatten_join('b'), q_result1)
        self.assertEqual(q.flatten_join(('b',)), q_result1)
        self.assertEqual(q.flatten_join(lambda b:b), q_result1)
        self.assertEqual(q.flatten_join('b', 'b'), q_result2)
        self.assertEqual(q.flatten_join('b', lambda x:x, 'b'), q_result2)
    ## flatten_join_pn
    ## unique
    def test_unique(self):
        q = Qfrom({'a': [1, 2, 2, 3, 3], 'b': [4, 5, 5, 6, 7]})

        q_result1 = Qfrom({'a': [1, 2, 3], 'b': [4, 5, 6]})
        q_result2 = Qfrom({'a': [1, 2, 3, 3], 'b': [4, 5, 6, 7]})

        self.assertEqual(q.unique('a'), q_result1)
        self.assertEqual(q.unique(('a',)), q_result1)
        self.assertEqual(q.unique('a, b'), q_result2)
        self.assertEqual(q.unique(('a', 'b')), q_result2)
        self.assertEqual(q.unique(lambda a,b: a+b), q_result2)
        self.assertEqual(q.unique('a,b', lambda x,y: x+y), q_result2)

    ## agg
    def test_agg(self):
        q1 = Qfrom({'a': [1, 2, 3], 'b': [5, 6, 7]})
        self.assertEqual(q1.agg('sum(a), sum(b)'), (6, 18))
        self.assertEqual(q1.agg('a.sum(), b.sum()'), (6, 18))
        self.assertEqual(q1.agg(lambda a,b: (sum(a), sum(b))), (6, 18))
        self.assertEqual(q1.agg(lambda a,b: {'a': sum(a), 'b':sum(b)}), {'a':6, 'b':18})
        self.assertEqual(q1.agg(('a', 'b'), lambda x,y: (sum(x), sum(y))), (6, 18))
        self.assertEqual(q1.agg('min(a), max(b)'), (1, 7))
        self.assertEqual(q1.agg('a.min(), b.max()'), (1, 7))
        #self.assertEqual(q1.agg('a.mean(), b.median()'), (2, 6))
        self.assertEqual(q1.agg('max(a), min(a), sum(a), max(b), min(b), sum(b)'), (3, 1, 6, 7, 5, 18))

    ## agg_pairs
    def test_agg_pairs(self):
        q1 = Qfrom({'a': [1, 2, 3], 'b': [5, 6, 7]})
        q2 = Qfrom({'a': ['a', 'b', 'c'], 'b': [5, 6, 7]})

        self.assertEqual(q1.agg_pairs('a[0]+a[1]'), 6)
        self.assertEqual(q1.agg_pairs(lambda a: a[0]+a[1]), 6)
        self.assertEqual(q1.agg_pairs('a', lambda x: x[0]+x[1]), 6)
        self.assertEqual(q1.agg_pairs(('a',), lambda x: x[0]+x[1]), 6)
        self.assertEqual(q1.agg_pairs(lambda a, b: (a[0]+a[1], b[0]+b[1])), (6, 18))
        self.assertEqual(q1.agg_pairs('a, b', lambda x, y: (x[0]+x[1], y[0]+y[1])), (6, 18))
        self.assertEqual(q1.agg_pairs(('a', 'b'), lambda x, y: (x[0]+x[1], y[0]+y[1])), (6, 18))
        self.assertEqual(q1.agg_pairs('a[0]+a[1], b[0]+b[1]'), (6, 18))
        self.assertEqual(q1.agg_pairs('sum(a), sum(b)'), (6, 18))
        self.assertEqual(q1.agg_pairs('min(a), max(b)'), (1, 7))
        #self.assertEqual(q1.agg_pairs('mean(a), median(b)'), (2, 6))
        #self.assertEqual(q1.agg_pairs('max(a), min(a), sum(a), max(b), min(b), sum(b)'), (3, 1, 6, 7, 5, 18))
        self.assertEqual(q2.agg_pairs(lambda a: a[0]+a[1]), 'abc')
    ## any
    '''def test_any(self):
        q1 = Qfrom({'a': [1, 2, 3], 'b': [5, 6, 7]})
        q2 = Qfrom({'a': [None, 2, 3], 'b': [5, 6, 7]})
        q3 = Qfrom({'a': [None, None, None], 'b': [5, 6, 7]})
        q4 = Qfrom({'a': [], 'b': []})
        q5 = Qfrom({})

        self.assertTrue(q1.any())
        self.assertTrue(q2.any('a'))
        self.assertTrue(q2.any(('a',)))
        self.assertTrue(q2.any(('a', 'b')))
        self.assertTrue(q2.any('a, b'))
        self.assertFalse(q3.any('a'))
        self.assertFalse(q3.any(('a',)))
        self.assertTrue(q3.any(('a', 'b')))
        self.assertTrue(q3.any('a, b'))
        self.assertFalse(q4.any())
        self.assertFalse(q4.any('a'))
        self.assertFalse(q5.any())
    def test_any_predicate(self):
        q = Qfrom({'a': [1, 2, 3], 'b': [5, 6, 7]})
        
        self.assertTrue(q.any(lambda a: a>2))
        self.assertTrue(q.any('a>2'))
        self.assertFalse(q.any(lambda a: a>3))
        self.assertFalse(q.any('a>3'))
    ## all
    def test_all(self):
        q1 = Qfrom({'a': [1, 2, 3], 'b': [5, 6, 7]})
        q2 = Qfrom({'a': [None, 2, 3], 'b': [5, 6, 7]})
        q3 = Qfrom({'a': [None, None, None], 'b': [5, 6, 7]})
        q4 = Qfrom({'a': [], 'b': []})
        q5 = Qfrom({})

        self.assertTrue(q1.all())
        self.assertFalse(q2.all('a'))
        self.assertTrue(q2.all('b'))
        self.assertFalse(q2.all(('a',)))
        self.assertFalse(q2.all(('a', 'b')))
        self.assertFalse(q2.all('a, b'))
        self.assertFalse(q3.all('a'))
        self.assertFalse(q3.all(('a',)))
        self.assertFalse(q3.all('a, b'))
        self.assertFalse(q4.all())
        self.assertFalse(q4.all('a'))
        self.assertFalse(q5.all())
    def test_all_predicate(self):
        q = Qfrom({'a': [1, 2, 3], 'b': [5, 6, 7]})
        
        self.assertTrue(q.all(lambda a: a>0))
        self.assertTrue(q.all('a>0'))
        self.assertFalse(q.all(lambda a: a>1))
        self.assertFalse(q.all('a>1'))
    '''
    ## min
    def test_min(self):
        q1 = Qfrom({'a': [1, 2, 3], 'b': [5, 6, 7]})

        self.assertEqual(q1.min(), (1, 5))
        self.assertEqual(q1.min('a'), 1)
        self.assertEqual(q1.min(lambda a, b, i: (a+b, i)), (6, 0))
    ## min_id
    def test_min_id(self):
        q1 = Qfrom({'a': [1, 2, 3], 'b': [5, 6, 7]})

        self.assertEqual(q1.min_id(), (0, 0))
        self.assertEqual(q1.min_id('a'), 0)
        self.assertEqual(q1.min_id(lambda a, b, i: (a+b, i)), (0, 0))
    ## min_item
    def test_min_item(self):
        q1 = Qfrom({'a': [1, 2, 3], 'b': [5, 6, 7]})

        self.assertEqual(q1.min_item(), ((1,5), (1,5)))
        self.assertEqual(q1.min_item('a'), (1,5))
        self.assertEqual(q1.min_item(lambda a, b, i: (a+b, i)), ((1,5), (1,5)))
    ## max
    def test_max(self):
        q1 = Qfrom({'a': [1, 2, 3], 'b': [5, 6, 7]})

        self.assertEqual(q1.max(), (3, 7))
        self.assertEqual(q1.max('a'), 3)
        self.assertEqual(q1.max(lambda a, b, i: (a+b, i)), (10, 2))
    ## max_id
    def test_max_id(self):
        q1 = Qfrom({'a': [1, 2, 3], 'b': [5, 6, 7]})

        self.assertEqual(q1.max_id(), (2, 2))
        self.assertEqual(q1.max_id('a'), 2)
        self.assertEqual(q1.max_id(lambda a, b, i: (a+b, i)), (2, 2))
    ## max_item
    ## sum
    def test_sum(self):
        q1 = Qfrom({'a': [1, 2, 3], 'b': [5, 6, 7]})

        self.assertEqual(q1.sum(), (6, 18))
        self.assertEqual(q1.sum('a'), 6)
        self.assertEqual(q1.sum(lambda a, b, i: (a+b, i)), (24, 3))
    ## mean
    def test_mean(self):
        q1 = Qfrom({'a': [1, 2, 3], 'b': [5, 6, 7]})

        self.assertEqual(q1.mean(), (2, 6))
        self.assertEqual(q1.mean('a'), 2)
        self.assertEqual(q1.mean(lambda a, b, i: (a+b, i)), (8, 1))
    ## median
    def test_median(self):
        q1 = Qfrom({'a': [1, 2, 3], 'b': [5, 6, 7]})

        self.assertEqual(q1.median(), (2, 6))
        self.assertEqual(q1.median('a'), 2)
        self.assertEqual(q1.median(lambda a, b, i: (a+b, i)), (8, 1))
    ## var
    ## len
    def test_len(self):
        q1 = Qfrom({'a': [1, 2, 3], 'b': [5, 6, 7]})
        q2 = Qfrom().concat_outer(q1)

        self.assertEqual(len(q1), 3)
        self.assertEqual(len(q2), 3)
    ## size
    def test_size(self):
        q1 = Qfrom({'a': [1, 2, 3], 'b': [5, 6, 7]})
        q2 = Qfrom().concat_outer(q1)

        self.assertEqual(len(q1), 3)
        self.assertEqual(len(q2), 3)
    
    ## calc
    ## call
    def test_call(self):
        q = Qfrom({'a': [1, 2, 3], 'b': [5, 6, 7]})
        q_result = Qfrom({'a': [1, 2, 3], 'm': [1, 4, 9]})
        
        f1 = lambda x: x\
            .select('a')\
            .select_join(lambda a: a**2, 'm')
        f2 = lambda x, pot: x\
            .select('a')\
            .select_join(lambda a: a**pot, 'm')
        
        self.assertEqual(q(f1), q_result)
        self.assertEqual(q(f2, 2), q_result)
        self.assertEqual(q(f2, pot=2), q_result)

    ## plot
    ## plot_bar
    ## plot_hist
    ## plot_box
    ## plot_scatter

    ## tolist
    def test_tolist(self):
        q = Qfrom({'a': [1, 2, 3], 'b': [5, 6, 7]})
        l = [(1, 5), (2, 6), (3, 7)]

        self.assertEqual(q.tolist(), l)
    ## (toset)
    ## todict
    def test_todict(self):
        d = {'a': np.array([1, 2, 3]), 'b': np.array([5, 6, 7])}
        q = Qfrom(d)

        self.assertEqual(q.todict(), d)
    ## (toarray)
    ## (tomtx)
    ## todf
    ## tocsv
    ## (tocsvfile)
    ## (tojson)
    ## (tojsonfile)
    ## __array__