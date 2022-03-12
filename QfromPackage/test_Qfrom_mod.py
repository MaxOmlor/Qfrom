import unittest
from Qfrom_mod import trans_func, trans_funcv_str, Qfrom
import pandas
import numpy as np

test_data_set = [
    {'name': 'Ann', 'age': 41, 'job_title': 'manager', 'manager': None},

    {'name': 'Steven', 'age': 42, 'job_title': 'manager', 'manager': 'Ann'},
    {'name': 'Max', 'age': 31, 'job_title': 'employee', 'manager': 'Steven'},
    {'name': 'Jack', 'age': 38, 'job_title': 'employee', 'manager': 'Steven'},
    {'name': 'Julia', 'age': 35, 'job_title': 'employee', 'manager': 'Steven'},
    {'name': 'Clara', 'age': 32, 'job_title': 'employee', 'manager': 'Steven'},

    {'name': 'Emma', 'age': 48, 'job_title': 'manager', 'manager': 'Ann'},
    {'name': 'Bob', 'age': 25, 'job_title': 'freelancer', 'manager': 'Emma'},
    {'name': 'Anna', 'age': 29, 'job_title': 'freelancer', 'manager': 'Emma'},
    {'name': 'Lena', 'age': 23, 'job_title': 'freelancer', 'manager': 'Emma'},
]
'''test_data_ordered_by_job_title_dict = {
    'manager': Qfrom([
        {'name': 'Ann', 'age': 41, 'job_title': 'manager', 'manager': None},
        {'name': 'Steven', 'age': 42, 'job_title': 'manager', 'manager': 'Ann'},
        {'name': 'Emma', 'age': 48, 'job_title': 'manager', 'manager': 'Ann'}]),
    'employee': Qfrom([
        {'name': 'Max', 'age': 31, 'job_title': 'employee', 'manager': 'Steven'},
        {'name': 'Jack', 'age': 38, 'job_title': 'employee', 'manager': 'Steven'},
        {'name': 'Julia', 'age': 35, 'job_title': 'employee', 'manager': 'Steven'},
        {'name': 'Clara', 'age': 32, 'job_title': 'employee', 'manager': 'Steven'}]),
    'freelancer': Qfrom([
        {'name': 'Bob', 'age': 25, 'job_title': 'freelancer', 'manager': 'Emma'},
        {'name': 'Anna', 'age': 29, 'job_title': 'freelancer', 'manager': 'Emma'},
        {'name': 'Lena', 'age': 23, 'job_title': 'freelancer', 'manager': 'Emma'}])}'''
test_data_ordered_by_job_title_dict = {
    'manager': [
        {'name': 'Ann', 'age': 41, 'job_title': 'manager', 'manager': None},
        {'name': 'Steven', 'age': 42, 'job_title': 'manager', 'manager': 'Ann'},
        {'name': 'Emma', 'age': 48, 'job_title': 'manager', 'manager': 'Ann'}],
    'employee': [
        {'name': 'Max', 'age': 31, 'job_title': 'employee', 'manager': 'Steven'},
        {'name': 'Jack', 'age': 38, 'job_title': 'employee', 'manager': 'Steven'},
        {'name': 'Julia', 'age': 35, 'job_title': 'employee', 'manager': 'Steven'},
        {'name': 'Clara', 'age': 32, 'job_title': 'employee', 'manager': 'Steven'}],
    'freelancer': [
        {'name': 'Bob', 'age': 25, 'job_title': 'freelancer', 'manager': 'Emma'},
        {'name': 'Anna', 'age': 29, 'job_title': 'freelancer', 'manager': 'Emma'},
        {'name': 'Lena', 'age': 23, 'job_title': 'freelancer', 'manager': 'Emma'}]}
        

def add(a, b):
    return a+b
def increase(x):
    return x+1

def infinite_sequence():
    i = 0
    while(True):
        yield i
        i += 1



class TestTransFuncStr(unittest.TestCase):
    def test_func(self):
        test_values = [
            #(*[in], out)
            (['x:x'], 'lambda x: x'),
            (['item:item'], 'lambda item: item'),
            (['item1 , item2:item1'], 'lambda item1 , item2: item1'),
            (['x:x[1]'], 'lambda x: x[1]'),
            (['x:x.a'], 'lambda x: x.a'),
            (['a, b:a+b'], 'lambda a, b: a+b'),

            (['item.abc'], 'lambda item: {"abc": item["item"]["abc"]}'),
            (['item.abc', False], 'lambda item: item["item"]["abc"]'),
            (['item1.ab, item2.cd'], 'lambda item: {"ab": item["item1"]["ab"], "cd": item["item2"]["cd"]}'),
            (['item1.ab,item2.cd'], 'lambda item: {"ab": item["item1"]["ab"], "cd": item["item2"]["cd"]}'),
            (['item1.ab as a, item2.cd as c'], 'lambda item: {"a": item["item1"]["ab"], "c": item["item2"]["cd"]}'),
            (['item1.ab as a, item2.cd'], 'lambda item: {"a": item["item1"]["ab"], "cd": item["item2"]["cd"]}'),
            (['i1.ab as a, i2.cd, i3.ef as e'], 'lambda item: {"a": item["i1"]["ab"], "cd": item["i2"]["cd"], "e": item["i3"]["ef"]}'),
            (['1.ab, 2.cd'], 'lambda item: {"ab": item[1]["ab"], "cd": item[2]["cd"]}'),

            (['-a'], 'lambda item: {key:item[key] for key in item if key != "a"}'),
            (['-a, -b'], 'lambda item: {key:item[key] for key in item if key not in ["a", "b"]}'),
            
            #predicates
            (['abc == "abc"', True, ['abc']], 'lambda item: item["abc"] == "abc"'),
            (['"abc" == abc'], 'lambda item: "abc" == item["abc"]'),
            (['abc != "abc"'], 'lambda item: item["abc"] != "abc"'),
            (['"abc" != abc'], 'lambda item: "abc" != item["abc"]'),
            (['abc < "abc"'], 'lambda item: item["abc"] < "abc"'),
            (['abc in ["abc"]'], 'lambda item: item["abc"] in ["abc"]'),
            (['abc is None'], 'lambda item: item["abc"] is None'),
            (['abc is not None'], 'lambda item: item["abc"] is not None'),
            (['a==1 or a==2'], 'lambda item: item["a"]==1 or item["a"]==2'),
            (['a==1 and a==2'], 'lambda item: item["a"]==1 and item["a"]==2'),
            (['2<a<5'], 'lambda item: 2<item["a"]<5'),
            (['2<=a<=5'], 'lambda item: 2<=item["a"]<=5'),
        ]
        for input, output in test_values:
            self.assertEqual(trans_funcv_str(*input), output)


        with self.assertRaises(SyntaxError):
            trans_funcv_str('')
        #with self.assertRaises(SyntaxError):
        #    trans_funcv_str('abc abc')
        #with self.assertRaises(SyntaxError):
        #    trans_funcv_str('item. abc')

class TestTransFunc(unittest.TestCase):
    def test_func(self):

        with self.assertRaises(SyntaxError):
            trans_func(None)
        

class TestQfromClass(unittest.TestCase):
# test all functions
    # __init__(self, *args) -> None:
    def test_init(self):
        self.assertEqual(Qfrom(range(5)).to_list(), [0, 1, 2, 3, 4])
        self.assertEqual(Qfrom(zip([1, 2, 3], [3, 2, 1])).to_list(), [(1, 3), (2, 2), (3, 1)])

        self.assertEqual(Qfrom().to_list(), [])

        with self.assertRaises(ValueError):
            Qfrom(None)
    def test_init_csv_str(self):
        csv = 'a,b,c\n1,2,3\n10, 20, 30\n'
        self.assertEqual(Qfrom(csv).to_list(), [{'a':1, 'b':2, 'c':3}, {'a':10, 'b':20, 'c':30}])
        csv = '''
        a,b,c
        1,2,3
        10,20,30
        '''
        self.assertEqual(Qfrom(csv).to_list(), [{'a':1, 'b':2, 'c':3}, {'a':10, 'b':20, 'c':30}])
        csv = 'a;b;c\n1;2;3\n10; 20; 30\n'
        self.assertEqual(Qfrom(csv, delimiter=';').to_list(), [{'a':1, 'b':2, 'c':3}, {'a':10, 'b':20, 'c':30}])
    def test_init_csv_file_ex_import(self):
        test_list = [{'a':1, 'b':2, 'c':3}, {'a':10, 'b':20, 'c':30}]
        file_path = "test csv.csv"

        Qfrom(test_list).to_csv_file(file_path)
        result = Qfrom(file_path).to_list()
        self.assertEqual(result, test_list)
    def test_init_csv_str(self):
        json = '[{"a":1, "b":2, "c":3}, {"a":10, "b":20, "c":30}]'
        self.assertEqual(Qfrom(json).to_list(), [{'a':1, 'b':2, 'c':3}, {'a':10, 'b':20, 'c':30}])

        json = "[{'a':1, 'b':2, 'c':3}, {'a':10, 'b':20, 'c':30}]"
        self.assertEqual(Qfrom(json).to_list(), [{'a':1, 'b':2, 'c':3}, {'a':10, 'b':20, 'c':30}])
    def test_init_json_file_ex_import(self):
        test_list = [{'a':1, 'b':2, 'c':3}, {'a':10, 'b':20, 'c':30}]
        file_path = "test json.json"

        Qfrom(test_list).to_csv_file(file_path)
        result = Qfrom(file_path).to_list()
        self.assertEqual(result, test_list)
    # __call__(self, *args):
    #def test_call(self):
        # Unpack Value
        #test_values_to_unpack = [
        #    #(*[input], output)
        #    ([[1, 2, 3]], np.array([1, 2, 3])), ([[]], []),
        #    ([np.array([1, 2, 3])], np.array([1, 2, 3])),
        #    ([set()], set()), ([{'a', 'b', 'c'}], {'a', 'b', 'c'}),
        #    ([{'a': 1, 'b': 2}], [{'key':'a', 'value':1}, {'key':'b', 'value':2}]), ([dict()], []),
        #]
        #for v_in, v_out in test_values_to_unpack:
        #    self.assertEqual(Qfrom(*v_in)(), v_out)
    # __len__(self):
    # __getitem__(self, key):
    def test_getitem(self):
        self.assertEqual(Qfrom([1, 2, 3])[1], 2)
        self.assertEqual(Qfrom([1, 2, 3])[1:], Qfrom([2, 3]))
        self.assertEqual(Qfrom([1, 2, 3])[:-1], Qfrom([1, 2]))
        self.assertEqual(Qfrom([1, 2, 3])[1], 2)
        self.assertEqual(Qfrom([[1, 2], [3, 4]])[1], [3, 4])
        self.assertEqual(Qfrom([1, 2, 3, 4])[[1, 3]], Qfrom([2, 4]))
    # __setitem__(self, key, item):
    def test_setitem(self):
        q = Qfrom([1, 2, 3])
        q[1] = 4
        self.assertEqual(q, Qfrom([1, 4, 3]))
    # __contains__(self, item):
    def test_contains(self):
        self.assertTrue(1 in Qfrom([1, 2, 3]))
        self.assertTrue(1 in Qfrom({1, 2, 3}))
        self.assertTrue('a' in Qfrom({'a', 'b', 'c'}))
        self.assertTrue([4, 5] in Qfrom([[1, 2, 3], [4, 5]]))
        self.assertTrue((1, 2) in Qfrom([(1, 2), (3, 4)]))
    # __iter__(self):
    def test_iter(self):
        self.assertEqual([item for item in Qfrom([1, 2, 3])], [1, 2, 3])
        self.assertEqual(next(iter(Qfrom([1, 2, 3]))), 1)
    # __eq__(self, other):
    def test_eq(self):
        test_values_eq = [
            [1, 2, 3],
            {1, 2},
            {1: 1, 2: 2},
        ]
        for v in test_values_eq:
            self.assertEqual(Qfrom(v), Qfrom(v))

        test_values_not_eq = [
            #(Qfrom(v1), v2)
            ([1, 2, 3], [1, 2, 3]),
            ({1, 2}, {1, 2}),
            ({1: 1, 2: 2}, {1: 1, 2: 2}),
        ]
        for v1, v2 in test_values_not_eq:
            self.assertNotEqual(Qfrom(v1), v2)
    # __str__(self) -> str:
    def test_str(self):
        self.assertEqual(str(Qfrom([1, 2, 3])), 'Qfrom([1 2 3])')
    # __repr__(self) -> str:
    # select(self, func):
    # s(self, func):
    def test_select(self):
        test_values_select = [
            #(data, *[select input], output)
            ([1, 2, 3], increase, [2, 3, 4]),
            ([1, 2, 3], lambda x:x+1, [2, 3, 4]),
            ([1, 2, 3], 'x : x+1', [2, 3, 4]),
            ([(1, 'a'), (2, 'b')], lambda x:x[0], [1, 2]),
            ([(1, 'a'), (2, 'b')], lambda x:x, [(1, 'a'), (2, 'b')]),
            (test_data_set, 'x:x["name"]', [
                'Ann', 'Steven', 'Max', 'Jack', 'Julia', 'Clara', 'Emma', 'Bob', 'Anna', 'Lena']),

            ([3, 2, 1], lambda x,i:x+i, [3, 3, 3]),
        ]
        for input_data, func, output in test_values_select:
            self.assertEqual(Qfrom(input_data).select(func).to_list(), output)
            self.assertEqual(Qfrom(input_data).s(func).to_list(), output)

        with self.assertRaises(SyntaxError):
            Qfrom([1, 2, 3]).select(None)
        with self.assertRaises(SyntaxError):
            Qfrom([1, 2, 3]).s(None)
        with self.assertRaises(TypeError):
            Qfrom([1, 2, 3]).select()
        with self.assertRaises(TypeError):
            Qfrom([1, 2, 3]).s()
    # edit_column(self, col_name, func):
    # edit(self, col_name, func):
    def test_edit_column(self):
        test_values = [
            #(data, col_name, func, output)
            ([1, 2, 3], 0, 'x:1', [{0:1}, {0:1}, {0:1}]),
            ([1, 2, 3], 1, 'x:1', [{0:1, 1:1}, {0:2, 1:1}, {0:3, 1:1}]),
            ([{'a':1, 'b':3}, {'a':2, 'b':4}], 'a', 'x:0', [{'a':0, 'b':3}, {'a':0, 'b':4}]),
            ([{'a':1}, {'a':2}], 'b', 'x:0', [{'a':1, 'b':0}, {'a':2, 'b':0}]),
            
            ([{'a':1}, {'a':2}], 'b', 'x, i:x["a"]-i', [{'a':1, 'b':1}, {'a':2, 'b':1}]),
        ]
        for input_data, col_name, func, output in test_values:
            self.assertEqual(Qfrom(input_data).edit_column(col_name, func).to_list(), output)
            self.assertEqual(Qfrom(input_data).edit(col_name, func).to_list(), output)

    # rename_column(self, col_name=None, new_name='Unnamed'):
    # rename(self, col_name=None, new_name='Unnamed'):
    def test_rename_column(self):
        test_values = [
            #(data, col_name, new_name, output)
            ([1, 2, 3, 4], None, 'x', [{'x':1}, {'x':2}, {'x':3}, {'x':4}]),
            ([[1, 2], [3, 4]], 0, 'x', [{'x':1, 1:2}, {'x':3, 1:4}]),
            ([{'a':1, 'b':3}, {'a':2, 'b':4}], 'a', 'c', [{'c':1, 'b':3}, {'c':2, 'b':4}]),
        ]
        for input_data, col_name, new_name, output in test_values:
            self.assertEqual(Qfrom(input_data).rename_column(col_name, new_name).to_list(), output)
            self.assertEqual(Qfrom(input_data).rename(col_name, new_name).to_list(), output)

        with self.assertRaises(ValueError):
            Qfrom([{'a':1, 'b':3}, {'a':2, 'b':4}]).rename('a')
    # where(self, func):
    # w(self, func):
    def test_where(self):
        test_values_where = [
            #(data, *[where input], output)
            ([('a', 1), ('b', 2), ('a', 3)], lambda x:x[0] == 'a', [('a', 1), ('a', 3)]),
            ([('a', 1), ('b', 2), ('a', 3)], "x:x[0] == 'a'", [('a', 1), ('a', 3)]),
            ([{'a': 1, }, {'a': 2}, {'a': 3}], 'x:x["a"] != 2', [{'a': 1}, {'a': 3}]),
            ([{'a': 1, }, {'a': 2}, {'a': 3}, {'a': 4}, {'a': 5}], '1<a<5', [{'a': 2}, {'a': 3}, {'a': 4}]),

            (['a', 'b', 'c', 'd'], lambda x,i:i%2==1, ['b', 'd']),
        ]
        for input_data, func, output in test_values_where:
            self.assertEqual(Qfrom(input_data).where(func).to_list(), output)
            self.assertEqual(Qfrom(input_data).w(func).to_list(), output)
            self.assertEqual(Qfrom(input_data).where(func).s('x:x').to_list(), output)
            self.assertEqual(Qfrom(input_data).w(func).s('x:x').to_list(), output)

        with self.assertRaises(SyntaxError):
            Qfrom([1, 2, 3]).where(None)
        with self.assertRaises(SyntaxError):
            Qfrom([1, 2, 3]).w(None)
        with self.assertRaises(TypeError):
            Qfrom([1, 2, 3]).where()
        with self.assertRaises(TypeError):
            Qfrom([1, 2, 3]).w()
    # flatten(self, select_collection=lambda item: item, select_result=lambda parent, child: child):
    # f(self, select_collection=lambda item: item, select_result=lambda parent, child: child):
    def test_flatten(self):
        test_values_flatten = [
            #(data, *[where input], output)
            ([[1, 2], [3, 4]], [], [1, 2, 3, 4]),
            ([1, 2], [lambda x:['a', 'b'], lambda p,c: (p, c)], [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b'),]),
            
            ([1, 2], [lambda x:['a', 'b'], lambda p,c,i: (p, c, i)], [(1, 'a', 0), (1, 'b', 1), (2, 'a', 2), (2, 'b', 3),])
        ]
        for input_data, paras, output in test_values_flatten:
            self.assertEqual(Qfrom(input_data).flatten(*paras).to_list(), output)
            self.assertEqual(Qfrom(input_data).f(*paras).to_list(), output)

        # test exceptions
    # aggregate(self, func):
    # agg(self, func):
    def test_aggregate(self):
        test_values_agg = [
            #(data, *[agg input], output)
            ([], add, None),
            ([1], add, 1),
            ([1, 2, 3, 4, 5], add, 15),
            ([1, 2, 3, 4, 5], lambda a,b: a+b, 15),
            ([1, 2, 3, 4, 5], 'a,b: a+b', 15),
            (['a', 'b', 'c'], lambda a,b: a+b, 'abc'),
            #([{1, 2}, {2, 3}, {3, 4}], lambda a,b: a.union(b), {1, 2, 3, 4}),
            ({1, 2, 3, 4, 5}, add, 15),
            ([True, False, True], lambda a,b: a or b, True),
            ([True, False, True], lambda a,b: a and b, False),
            ([{'a':1}, {'a':1}, {'a':1}], lambda a,b: {'a':a['a']+b['a']}, {'a':3}),
            ([{'a':'a'}, {'a':'b'}, {'a':'c'}], lambda a,b: {'a':a['a']+b['a']}, {'a':'abc'}),
        ]
        for input_data, func, output in test_values_agg:
            self.assertEqual(Qfrom(input_data).aggregate(func), output)
            self.assertEqual(Qfrom(input_data).agg(func), output)

        self.assertEqual(Qfrom([1, 2, 3, 4, 5]).aggregate(add), 15)
        self.assertEqual(Qfrom(['a', 'b', 'c']).aggregate(add), 'abc')
        self.assertEqual(Qfrom([{1, 2}, {2, 3}, {3, 4}]).aggregate('a,b: a.union(b)'), {1, 2, 3, 4})
        
        with self.assertRaises(SyntaxError):
            Qfrom([1, 2, 3]).aggregate(None)
        with self.assertRaises(SyntaxError):
            Qfrom([1, 2, 3]).agg(None)
            
    # group_by(self, get_key_func, get_value_func = lambda x:x):
    # g(self, get_key_func, get_value_func = lambda x:x):
    def test_group_by(self):
        '''test_values_order_by = [
            #(data, *[order by input], output)
            ([('a', 1), ('b', 2), ('a', 3)], [lambda x:x[0], lambda x:x[1]], {'a': Qfrom([1, 3]), 'b': Qfrom([2])}),
            ([('a', 1), ('b', 2), ('a', 3)], ['0', 'x:x[1]'], {'a': Qfrom([1, 3]), 'b': Qfrom([2])}),
            ([('a', 1), ('b', 2), ('a', 3)], ['x:x[0]', 'x:x[1]'], {'a': Qfrom([1, 3]), 'b': Qfrom([2])}),
            (test_data_set, [lambda x:x['job_title']], test_data_ordered_by_job_title_dict),
            (test_data_set, ['job_title'], test_data_ordered_by_job_title_dict),
            (test_data_set, ["x:x['job_title']"], test_data_ordered_by_job_title_dict),
            (['a', 'b', 'c', 'd'], [lambda x,i:i%2], [{'key':0, 'value':Qfrom(['a', 'c'])}, {'key':1, 'value':Qfrom(['b', 'd'])}]),
        ]'''
        test_values_order_by = [
            #(data, *[order by input], output)
            ([('a', 1), ('b', 2), ('a', 3)], [lambda x:x[0], lambda x:x[1]], {'a': [1, 3], 'b': [2]}),
            ([('a', 1), ('b', 2), ('a', 3)], ['0', 'x:x[1]'], {'a': [1, 3], 'b': [2]}),
            ([('a', 1), ('b', 2), ('a', 3)], ['x:x[0]', 'x:x[1]'], {'a': [1, 3], 'b': [2]}),
            (test_data_set, [lambda x:x['job_title']], test_data_ordered_by_job_title_dict),
            (test_data_set, ['job_title'], test_data_ordered_by_job_title_dict),
            (test_data_set, ["x:x['job_title']"], test_data_ordered_by_job_title_dict),
            (['a', 'b', 'c', 'd'], [lambda x,i:i%2], [{'key':0, 'value':['a', 'c']}, {'key':1, 'value':['b', 'd']}]),
        ]
        for input_data, paras, output in test_values_order_by:
            self.assertEqual(Qfrom(input_data).group_by(*paras), Qfrom(output))
            self.assertEqual(Qfrom(input_data).g(*paras), Qfrom(output))

        with self.assertRaises(SyntaxError):
            Qfrom([1, 2, 3]).group_by(None)
        with self.assertRaises(SyntaxError):
            Qfrom([1, 2, 3]).g(None)
        with self.assertRaises(SyntaxError):
            Qfrom([1, 2, 3]).group_by(None, None)
        with self.assertRaises(SyntaxError):
            Qfrom([1, 2, 3]).g(None, None)
        with self.assertRaises(SyntaxError):
            Qfrom([1, 2, 3]).group_by('x:x', None)
        with self.assertRaises(SyntaxError):
            Qfrom([1, 2, 3]).g('x:x', None)

    # order_by(self, get_key_func = lambda x:x, reverse=False):
    # o(self, get_key_func = lambda x:x, reverse=False):
    def test_order_by(self):
        test_values = [
            #(data, *[order by input], output)
            ([1, 4, 2, 3], [], [1, 2, 3, 4]),
            (test_data_set, [lambda x: x['name']], sorted(test_data_set, key=lambda x: x['name'])),
            (test_data_set, ["x: x['name']"], sorted(test_data_set, key=lambda x: x['name'])),
            (test_data_set, ["name"], sorted(test_data_set, key=lambda x: x['name'])),
            (test_data_set, ["x: x['name']", True], sorted(test_data_set, key=lambda x: x['name'], reverse=True)),
            (test_data_set, [lambda x: x['age']], sorted(test_data_set, key=lambda x: x['age'])),
            ([1, 2, 3, 4], [lambda x: 5-x, True], [1, 2, 3, 4]),
        ]
        for input_data, paras, output in test_values:
            self.assertEqual(Qfrom(input_data).order_by(*paras), Qfrom(output))
            self.assertEqual(Qfrom(input_data).o(*paras), Qfrom(output))
        
        #with self.assertRaises(ValueError):
        #    Qfrom([1, 2, 3]).order_by(None)
        #with self.assertRaises(ValueError):
        #    Qfrom([1, 2, 3]).o(None)
        #with self.assertRaises(ValueError):
        #    Qfrom([1, 2, 3]).order_by(None, None)
        #with self.assertRaises(ValueError):
        #    Qfrom([1, 2, 3]).o(None, None)
        with self.assertRaises(ValueError):
            Qfrom([1, 2, 3]).order_by('x:x', None)
        with self.assertRaises(ValueError):
            Qfrom([1, 2, 3]).o('x:x', None)

    # any(self, predicate=None):
    def test_any(self):
        test_values = [
            #(input, predicate, output)
            ([1, 2, 3], None, True),
            ([], None, False),
            ([1, 2, 3], lambda x:x>2, True),
            ([1, 2, 3], lambda x:x>3, False),
            ([(1, 'a'), (2, 'a'), (3, 'a')], lambda x:x[0]>2, True),
            ([(1, 'a'), (2, 'a'), (3, 'a')], lambda x:x[0]>3, False),
        ]
        for input_data, predicate, output in test_values:
            self.assertEqual(Qfrom(input_data).any(predicate), output)
    # all(self, predicate=None):
    # min(self, func=None):
    def test_min(self):
        test_values = [
            #(input, predicate, output)
            ([1, 4, 2, 3], None, 1),
            ([(1, 'a'), (4, 'a'), (2, 'a'), (3, 'a')], 'x:x[0]', (1, 'a')),
        ]
        for input_data, key, output in test_values:
            self.assertEqual(Qfrom(input_data).min(key), output)
    # max(self, func=None):
    def test_max(self):
        test_values = [
            #(input, predicate, output)
            ([1, 4, 2, 3], None, 4),
            ([(1, 'a'), (4, 'a'), (2, 'a'), (3, 'a')], 'x:x[0]', (4, 'a')),
        ]
        for input_data, key, output in test_values:
            self.assertEqual(Qfrom(input_data).max(key), output)
    # sum(self, key):
    def test_sum(self):
        test_values = [
            #(input, predicate, output)
            ([1, 4, 2, 3], None, 10),
            ([(1, 'a'), (4, 'a'), (2, 'a'), (3, 'a')], 'x:x[0]', 10),
        ]
        for input_data, key, output in test_values:
            self.assertEqual(Qfrom(input_data).sum(key), output)
    # mean(self, key):
    def test_sum(self):
        test_values = [
            #(input, predicate, output)
            ([1, 4, 2, 3], None, 2.5),
            ([(1, 'a'), (4, 'a'), (2, 'a'), (3, 'a')], 'x:x[0]', 2.5),
        ]
        for input_data, key, output in test_values:
            self.assertEqual(Qfrom(input_data).mean(key), output)
    # median(self, key):
    def test_median(self):
        test_values = [
            #(input, predicate, output)
            ([1, 15, 2, 7, 3], None, 3),
            ([(1, 'a'), (15, 'a'), (2, 'a'), (7, 'a'), (3, 'a')], 'x:x[0]', 3),
        ]
        for input_data, key, output in test_values:
            self.assertEqual(Qfrom(input_data).median(key), output)
    # var(self, key):
    # normalize(self, key=None):
    # norm(self, key=None):
    def test_normalize(self):
        self.assertEqual(Qfrom([1, 2, 3, 4]).normalize().to_list(), [0.25, 0.5, 0.75, 1])
        self.assertEqual(Qfrom([1, 2, 3, 4]).norm().to_list(), [0.25, 0.5, 0.75, 1])

        self.assertEqual(Qfrom([{'a': 1}, {'a': 2}, {'a': 3}, {'a': 4}]).normalize('a').to_list(), [{'a': 0.25}, {'a': 0.5}, {'a': 0.75}, {'a': 1}])
        self.assertEqual(Qfrom([{'a': 1}, {'a': 2}, {'a': 3}, {'a': 4}]).norm('a').to_list(), [{'a': 0.25}, {'a': 0.5}, {'a': 0.75}, {'a': 1}])
    # concat(self, other):
    def test_concat(self):
        self.assertEqual(Qfrom([1, 2]).concat([3, 4]).to_list(), [1, 2, 3, 4])
        self.assertEqual(Qfrom([1, 2]).concat(Qfrom([3, 4])).to_list(), [1, 2, 3, 4])
    # add(self, item):
    def test_add(self):
        new_row =  {'name': 'Peter', 'age': 29, 'job_title': 'freelancer', 'manager': 'Emma'}

        test_values = [
            #(data, input, output)
            ([1, 2, 3, 4], 5, [1, 2, 3, 4, 5]),
            (test_data_set, new_row, test_data_set+[new_row]),
            #([1, 2, 3, 4], (5, 6), [1, 2, 3, 4, (5, 6)]),
        ]
        for input_data, paras, output in test_values:
            q = Qfrom(input_data)
            q.add(paras)
            self.assertEqual(q, Qfrom(output))
    # columns(self):
    def test_columns(self):
        self.assertEqual(Qfrom([{'a':1}, {'a':1}, {'a':1}]).columns(), ['a'])
        self.assertEqual(Qfrom([1, 2, 3]).columns(), None)
        self.assertEqual(Qfrom([(1, 2, 3), (4, 5, 6)]).columns(), [0, 1, 2])
    # foreach(self, action):
    # to_dict(self, key=lambda x:x[0], value=lambda x:x[1]):
    def test_to_dict(self):
        self.assertEqual(Qfrom(test_data_ordered_by_job_title_dict).to_dict(), test_data_ordered_by_job_title_dict)
        self.assertEqual(Qfrom([]).to_dict(), dict())
    # to_dataframe(self):
    def test_to_dataframe(self):
        test_df = pandas.DataFrame(test_data_set)
        output = Qfrom(test_df).to_dataframe()
        for (i1, row1), (i2, row2) in zip(test_df.iterrows(), output.iterrows()):
            self.assertEqual(i1, i2)
            self.assertEqual(dict(row1), dict(row2))
    # to_csv_file(self, path, encoding='UTF8'):
    def test_to_csv_file(self):
        output_path = 'test csv export.csv'
        Qfrom(test_data_set).to_csv_file(output_path)
        result = Qfrom('test csv export.csv').to_list()
        self.assertEqual(result, test_data_set)
    # to_json_file(self):
    # calculate(self):
    # as_parallel(self):
    def test_as_parallel(self):
        data = list(range(100))
        q = Qfrom(data)\
            .select('x:x+1')\
            .where('x:x<=50')\
            .select('x:x-1')\
            .order_by(reverse=True)\
            .select('x:x+1')\
            .where('x:x<=25')\
            .select('x:x-1')
        self.assertEqual(q.as_parallel().to_list(), list(range(24, -1, -1)))
            

# test what appens, if first qfrom of chained qfroms is chanching

class TestQfromIterClass(unittest.TestCase):
    # test all functions
        # __init__(self, qfrom):
        # __next__(self):
    pass