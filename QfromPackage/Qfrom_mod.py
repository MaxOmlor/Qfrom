from collections.abc import Iterable
import pandas as pd
import csv
import numpy as np
import enum
from collections.abc import Iterable
#import time
import multiprocessing
from multiprocessing import Pool, Process
from pathos.multiprocessing import ProcessingPool
import matplotlib.pyplot as plt
import inspect

#from translate_func_mod import trans_func
#from parse_mod import parse_str_to_collection, pandas_row_dict_iter

import re
import json
import os
import requests
import chardet
from io import StringIO



#def output_correct_type(item):
#    if isinstance(item, Iterable) and type(item) is not str:
#        return Qfrom(item)
#    return item
    

#def dict_kvp_iter(input_dict):
#    for t in input_dict.items():
#        yield t

#def flatten(iterator, select_collection_func, select_result_func):
    # optimize !!!
#    return np.array([select_result_func(row, item) for row in iterator for item in select_collection_func(row)])

#def concat_iterables_generator(iterable1, iterable2):
#    for item in iterable1:
#        yield item
#    for item in iterable2:
#        yield item



def split_predicate_by_var(predicate_str):
    var_pattern = '[a-zA-Z\_\.0-9]'
    escape_pattern = '[\'\"]'
    var_state = False
    escape_state = False
    str_part_list = []
    new_word = ''
    for c in predicate_str:
        #print(c, re.search(var_pattern, c) is not None, re.search(escape_pattern, c) is not None)
        old_var_state = var_state
        var_state = re.search(var_pattern, c) is not None and not escape_state
        is_escape_char = re.search(escape_pattern, c) is not None
        escape_state = is_escape_char ^ escape_state

        if old_var_state is not var_state:
            str_part_list.append(new_word)
            new_word = ''
        new_word += c
    str_part_list.append(new_word)
    return str_part_list

def trans_item_str(item_str, pefix='item'):
    word_list = item_str.strip().split('.')
    key_list = ['"'+word+'"'if type(try_parse_str(word)) is str else word for word in word_list]
    return pefix+'['+']['.join(key_list)+']'

def trans_funcv_str(func_str, to_dict=True, keys=None):
        match = re.search('^(\w+(\s*\,\s*\w+)*)\s*\:\s*(.*)$', func_str)
        if match is not None:
            return 'lambda ' + match[1] + ': ' + match[3]
        
        var_str = '(\w+(\.\w+)*'
        match = re.search('^'+var_str+'(\s*as\s*\w+)?)(\s*\,\s*'+var_str+'(\s*as\s*\w+)?))*$', func_str)
        if match is not None:
            column_list = []
            for col in func_str.split(','):
                col_str = col.strip()

                key = None
                value = None
                match = re.search('^(\w+(\.\w+)*)\s*as\s*(\w+)$', col_str)
                if match is not None:
                    parts = col_str.split('as')
                    key = parts[1].strip()
                    value = parts[0].strip()
                else:
                    key = col_str.split('.')[-1]
                    value = col_str
                column_list.append((key, trans_item_str(value)))
                
            item_str_list = ['"'+str(key)+'": '+value for key, value in column_list]

            if to_dict and len(item_str_list) == 1:
                return 'lambda item: {' + item_str_list[0] + '}'
            elif len(item_str_list) == 1:
                return 'lambda item: ' + [value for key, value in column_list][0]
            else:
                return 'lambda item: {' + ', '.join(item_str_list) + '}'
        
        #remove
        match = re.search('^(-\w+)(\s*\,\s*(-\w+))*', func_str)
        if match is not None:
            cols_to_remove = ['"'+re.sub('^-', '', col.strip())+'"' for col in func_str.split(',')]
            if len(cols_to_remove) == 1:
                return 'lambda item: {key:item[key] for key in item if key != '+cols_to_remove[0]+'}'
            return 'lambda item: {key:item[key] for key in item if key not in ['+', '.join(cols_to_remove)+']}'


        #predicates
        result = Qfrom(split_predicate_by_var(func_str))
        if len(result) > 1:
            var_list = result.s(lambda x,i:(x, i))\
                                [1::2]\
                                .w(lambda t:t[0] not in ['and', 'or', 'not', 'in', 'is'])\
                                .s(lambda t:(t[0], t[1], t[0].split('.')))\
                                .w(lambda t:t[2][0] in keys if keys is not None else type(try_parse_str(t[2][0])) is str)\
                                .s(lambda t:(trans_item_str(t[0]), t[1]))
            for var, i in var_list:
                result[i] = var
            return 'lambda item: ' + ''.join(result)
            
        '''result = split_predicate_by_var(func_str)
        if len(result) > 1:
            item_id_list = [(result[i], i) for i in np.arange(len(result))]
            item_id_list = item_id_list[1::2]
            item_id_list = [item for item in item_id_list if item[0] not in ['and', 'or', 'not', 'in', 'is']]
            item_id_attributes_list = [(item[0], item[1], item[0].split('.')) for item in item_id_list]
            item_id_attributes_list = [item for item in item_id_attributes_list if (keys is not None and item[2][0] in keys) or (keys is None and type(try_parse_str(item[2][0])) is str)]
            var_list = [(trans_item_str(item[0]), item[1]) for item in item_id_attributes_list]
            for var, i in var_list:
                result[i] = var
            return 'lambda item: ' + ''.join(result)'''

            
        #print('typo in: ' + func)
        #return None
        raise SyntaxError(func_str + ' cant be interpreted as a function')

        #return eval('lambda ' + func)

def trans_func(func, to_dict=True):
    if func is None:
        raise SyntaxError(str(func) + ' cant be interpreted as a function')
    if callable(func):
        return func
    if type(func) is str:
        return eval(trans_funcv_str(func, to_dict))

    raise SyntaxError(str(func) + ' cant be interpreted as a function')








def try_parse_str(text):
    if text == 'None':
        return None

    try:
        num = int(text)
        return num
    except BaseException:
        pass

    try:
        num = float(text)
        return num
    except BaseException:
        pass
    
    return text

def try_parse_str_to_collection(text, delimiter=',', headers=True):
    #try json
    try:
        mod_json_str = re.sub("\'", '\"', text)
        #mod_json_str = text.replace("\'", '\"')
        result = json.loads(mod_json_str)
        return result
    except BaseException:
        pass

    #try csv
    try:
        reader = csv.reader(StringIO(text), delimiter=delimiter)
        file_data = np.array([[try_parse_str(item.strip()) for item in row] for row in reader if ''.join(row).strip() != ''])
        if headers:
            header_list = file_data[0]
            data = file_data[1:]
            result = np.array([{h: try_parse_str(v) for h, v in zip(header_list, row)} for row in data])
            return result
        else:
            return file_data
    except BaseException:
        pass
    
    return None

def parse_str_to_collection(text, delimiter=',', headers=True):
    if re.search('^(https://|http://)', text) is not None:
        return requests.get(text).json()

    col_text = text
    if os.path.exists(text):
        f_rawdata = open(text, 'rb')
        rawdata = f_rawdata.read()
        f_rawdata.close()
        encoding = chardet.detect(rawdata)['encoding']

        f_col_text = open(text, newline='', encoding=encoding)
        col_text = f_col_text.read()
        f_col_text.close()
    
    return try_parse_str_to_collection(col_text, delimiter, headers)

def pandas_row_dict_iter(dataframe):
    #iterator = dataframe.iterrows()
    #func = lambda i: dict(next(iterator)[1])
    #func = np.frompyfunc(func, 1, 1)
    #return func(np.arange(len(dataframe)))

    #return np.array([dict(row) for index, row in dataframe.iterrows()])
    #return np.array([row.to_dict() for index, row in dataframe.iterrows()])

    #col_list = dataframe.columns
    #return np.array([{col_list[i]:row[i] for i in np.arange(len(col_list))} for row in dataframe.values])
    
    col_dict = {col: dataframe[col].values for col in dataframe.columns}
    return np.array([{col:values[i] for col, values in col_dict.items()} for i in np.arange(len(dataframe))])


handle_collection_specialcases = {
        dict: {
            #'unpack': lambda input_dict: dict_kvp_iter(input_dict),
            'unpack': lambda input_dict: np.array([{'key': key, 'value': value} for key, value in input_dict.items()]),
            'pack': lambda dict_list: dict((row['key'], row['value']) for row in dict_list),
        },
        pd.core.frame.DataFrame: {
            'unpack': lambda df: pandas_row_dict_iter(df),
            'pack': lambda row_dict_list: pd.DataFrame(row_dict_list),
        },
    }

def parse_iterable_to_array(iterable):
    iter_type = type(iterable)
    if iter_type is np.ndarray:
        if len(iterable) > 0 and type(iterable[0]) is str:
            return iterable.astype('object')
        return iterable
    if iter_type is Qfrom:
        return iterable()
        
    if iter_type in handle_collection_specialcases:
        return handle_collection_specialcases[iter_type]['unpack'](iterable)
    
    if isinstance(iterable, list):
        if len(iterable) > 0 and type(iterable[0]) in [list, np.array, tuple, str]:
            arr = np.empty(len(iterable), dtype=object)
            arr[:] = iterable
            return arr
        return np.array(iterable)
    # numpy from iter
    iter_list = list(iterable)
    if len(iter_list) > 0 and type(iter_list[0]) in [list, np.array, tuple, str]:
        arr = np.empty(len(iter_list), dtype=object)
        arr[:] = iter_list
        return arr
    return np.array(iter_list)

def apply_func(func, data_array):
    parameter_count = len(inspect.signature(func).parameters)
    if parameter_count == 1:
        func = np.frompyfunc(func, 1, 1)
        return func(data_array)
    if parameter_count == 2:
        func = np.frompyfunc(func, 2, 1)
        return func(data_array, np.arange(data_array.size))
    raise ValueError('Function needs ' + parameter_count+ ' parameters but only 2 are geiven.')
def worker_calc_operations(t):
        item, op_pipeline = t
        return calc_operations(item, op_pipeline)
def calc_operations(data_array, operation_list):
    object_type_items = [dict, list, tuple, set]
    result_array = data_array

    for op in operation_list:
        if not any(result_array):
            return result_array

        if op['Operation'] == Operation.SELECT:
            func = op['func']
            result_array = apply_func(func, result_array)
            continue
        if op['Operation'] == Operation.EDIT_COLUMN:
            if not any(result_array):
                return result_array

            col_name = op['col_name']
            func = None
            parameter_count = len(inspect.signature(op['func']).parameters)
            if parameter_count == 1:
                if type(result_array[0]) is dict:
                    func = lambda item: {**item, col_name:op['func'](item)}
                elif type(result_array[0]) in [list, np.ndarray]:
                    func = lambda item: {**{i:item[i] for i in np.arange(len(item))}, col_name:op['func'](item)}
                else:
                    func = lambda item: {0:item, col_name:op['func'](item)}
            if parameter_count == 2:
                if type(result_array[0]) is dict:
                    func = lambda item, i: {**item, col_name:op['func'](item, i)}
                elif type(result_array[0]) in [list, np.ndarray]:
                    func = lambda item, index: {**{i:item[i] for i in np.arange(len(item))}, col_name:op['func'](item, index)}
                else:
                    func = lambda item, i: {0:item, col_name:op['func'](item, i)}

            result_array = apply_func(func, result_array)
            continue
        if op['Operation'] == Operation.RENAME_COLUMN:
            if not any(result_array):
                return result_array

            col_name = op['col_name']
            new_name = op['new_name']
            func = None
            if type(result_array[0]) is dict and col_name is not None:
                func = lambda item: {key if key != col_name else new_name:value for key, value in item.items()}
            elif type(result_array[0]) in [list, np.ndarray] and col_name is not None:
                func = lambda item: {i if i != col_name else new_name:item[i] for i in np.arange(len(item))}
            else:
                if col_name is not None:
                    raise ValueError('there is no column: ' + col_name)
                func = lambda item: {new_name:item}

            func = np.frompyfunc(func, 1, 1)
            result_array = func(result_array)
            continue
        if op['Operation'] == Operation.WHERE:
            func = op['func']
            #mask = func(result_array)
            #masked_ids = np.where(mask, )
            #result_array = result_array[masked_ids]
            where_filter = apply_func(func, result_array)
            where_filter = where_filter.astype('bool')
            result_array = result_array[where_filter]
            continue
        if op['Operation'] == Operation.FLATTEN:
            select_collection_func = op['col_func']
            select_result_func = op['result_func']

            col_array = apply_func(select_collection_func, result_array)

            parameter_count = len(inspect.signature(select_result_func).parameters)
            result_list = []
            if parameter_count == 2:
                result_list = [select_result_func(result_array[i], item) for i in np.arange(result_array.size) for item in col_array[i]]
            if parameter_count == 3:
                i = 0
                for id_parent in np.arange(result_array.size):
                    parent = result_array[id_parent]
                    for child in col_array[id_parent]:
                        result_list.append(select_result_func(parent, child, i))
                        i += 1

            if any(result_list) and type(result_list[0]) in object_type_items:
                result_array = np.empty(len(result_list), dtype=object)
                result_array[:] = result_list
            else:
                result_array = np.array(result_list)
            continue
        if op['Operation'] == Operation.GROUP_BY:
            key_func = op['key_func']
            value_func = op['value_func']

            #group_dict = dict()
            #for item in result_array:
            #    key = key_func(item)
            #    value = value_func(item)

            #    if key not in group_dict:
            #        group_dict[key] = []

            #    group_dict[key].append(value)
            
            #result_array = unpack_iterable(group_dict)
            #continue

            #key_arr = apply_func(key_func, result_array)
            #value_arr = apply_func(value_func, result_array)
            group_dict = dict()
            #for key, value in zip(key_arr, value_arr):
            #for key, value in np.column_stack((key_arr, value_arr)):
            key_parameter_count = len(inspect.signature(key_func).parameters)
            value_parameter_count = len(inspect.signature(value_func).parameters)
            if key_parameter_count > 1 and value_parameter_count > 1:
                for i in np.arange(result_array.size):
                    item = result_array[i]
                    key = key_func(item, i)
                    value = value_func(item, i)
                    if key not in group_dict:
                        group_dict[key] = []
                    group_dict[key].append(value)
            elif key_parameter_count == 1 and value_parameter_count > 1:
                for i in np.arange(result_array.size):
                    item = result_array[i]
                    key = key_func(item)
                    value = value_func(item, i)
                    if key not in group_dict:
                        group_dict[key] = []
                    group_dict[key].append(value)
            elif key_parameter_count > 1 and value_parameter_count == 1:
                for i in np.arange(result_array.size):
                    item = result_array[i]
                    key = key_func(item, i)
                    value = value_func(item)
                    if key not in group_dict:
                        group_dict[key] = []
                    group_dict[key].append(value)
            else:
                for item in result_array:
                    key = key_func(item)
                    value = value_func(item)
                    if key not in group_dict:
                        group_dict[key] = []
                    group_dict[key].append(value)
            #group_dict = {key: Qfrom(value) for key, value in group_dict.items()}
            
            result_array = parse_iterable_to_array(group_dict)
            continue
        if op['Operation'] == Operation.ORDER_BY:
            key_func = op['key_func']
            reverse = op['reverse']

            if result_array.dtype == object:
                #print('py sort')
                #key_func = np.frompyfunc(key_func, 1, 1)
                #key_array = key_func(result_array)
                #sorted_ids = np.argsort(key_array)
                #sorted_array = result_array[sorted_ids]

                #result_array = np.array(sorted(result_array, key=key_func, reverse=reverse))
                if key_func is None:
                    result_array[:] = sorted(result_array, reverse=reverse)
                else:
                    result_array[:] = sorted(result_array, key=key_func, reverse=reverse)
            else:
                #print('np sort')
                if key_func is None:
                    sorted_array = np.sort(result_array)
                else:
                    key_array = apply_func(key_func, result_array)
                    sorted_ids = np.argsort(key_array)
                    sorted_array = result_array[sorted_ids]
                result_array = np.flip(sorted_array) if reverse else sorted_array
            continue
    
    return result_array

def add_item_to_value_holder_func(func, value_holder, item):
    value_holder.value = func(value_holder.value, item)






class Operation(enum.Enum):
    SELECT = 1
    EDIT_COLUMN = 2
    RENAME_COLUMN = 3
    WHERE = 4
    FLATTEN = 5
    GROUP_BY = 6
    ORDER_BY = 7



class Qfrom():
    def __init__(self, iterable=[], operation_list=[], delimiter=',' , headers=True) -> None:
        if type(iterable) is str:
            self.__iterable = parse_iterable_to_array(parse_str_to_collection(iterable, delimiter=delimiter, headers=headers))
        elif isinstance(iterable, Iterable):
            self.__iterable = parse_iterable_to_array(iterable)
        else:
            raise ValueError(str(iterable) + ' is not iterable or a known string')

        self.__operation_list = operation_list

    #-- standart list func --------------------------------------#
    def __len__(self) -> int:
        if any(self.__operation_list):
            self.calculate()
        return self.__iterable.size

    def __getitem__(self, key):
        if any(self.__operation_list):
            self.calculate()
        result = self.__iterable[key]
        
        if type(result) is np.ndarray:
            return Qfrom(result)
        return result

    def __setitem__(self, key, item):
        if any(self.__operation_list):
            self.calculate()
        self.__iterable[key] = item

    def __contains__(self, item) -> bool:
        if any(self.__operation_list):
            self.calculate()
        if type(item) in [list, tuple]:
            return item in self.__iterable.tolist()
        return item in self.__iterable
    
    def __iter__(self):
        self.calculate()
        return iter(self.__iterable)

    def add(self, item) -> None:
        self.__iterable = self.concat([item])()

    def __str__(self) -> str:
        self.calculate()
        if self.any() and type(self.__iterable[0]) in [dict, tuple, list, np.ndarray]:
            return 'Qfrom(\t'+'\n\t'.join([str(row) for row in self.__iterable])+')'
        return 'Qfrom('+str(self.__iterable)+')'

    def __repr__(self) -> str:
        return str(self)


    #-- expanded list func --------------------------------------#
    def select(self, func):
        select_func = trans_func(func)

        operation = {
            'Operation': Operation.SELECT,
            'func': select_func
        }
        return Qfrom(self.__iterable, operation_list=self.__operation_list+[operation])
    def s(self, func):
        return self.select(func)

    def where(self, func):
        where_func = trans_func(func)

        operation = {
            'Operation': Operation.WHERE,
            'func': where_func
        }
        return Qfrom(self.__iterable, operation_list=self.__operation_list+[operation])
    def w(self, func):
        return self.where(func)

    def flatten(self, select_collection=lambda item: item, select_result=lambda parent, child: child):
        select_collection_func = trans_func(select_collection, False)
        select_result_func = trans_func(select_result)

        operation = {
            'Operation': Operation.FLATTEN,
            'col_func': select_collection_func,
            'result_func': select_result_func
            }
        return Qfrom(self.__iterable, operation_list=self.__operation_list+[operation])
    def f(self, select_collection=lambda item: item, select_result=lambda parent, child: child):
        return self.flatten(select_collection, select_result)

    def group_by(self, key, value = lambda x:x):
        get_key_func = trans_func(key, False)
        get_value_func = trans_func(value)
        
        operation = {
            'Operation': Operation.GROUP_BY,
            'key_func': get_key_func,
            'value_func': get_value_func
            }
        return Qfrom(self.__iterable, operation_list=self.__operation_list+[operation])
    def g(self, get_key_func, get_value_func = lambda x:x):
        return self.group_by(get_key_func, get_value_func)

    def order_by(self, key = None, reverse=False):
        if reverse is None:
            raise ValueError('reverse should be a boolean not: ' + str(reverse))
        
        operation = {
            'Operation': Operation.ORDER_BY,
            'key_func': None,
            'reverse': reverse
        }
        if key is not None:
            get_key_func = trans_func(key, False)
            operation['key_func'] = get_key_func

        return Qfrom(self.__iterable, operation_list=self.__operation_list+[operation])
    def o(self, get_key_func = lambda x:x, reverse=False):
        return self.order_by(get_key_func, reverse)

    def first(self, key=None):
        self.calculate()
        if key == None:
            return self.__iterable[0] if any(self.__iterable) else None
        q = self.where(key)
        if q.any():
            return q[0]
        return None

    def any(self, predicate=None) -> bool:
        self.calculate()
        if predicate == None:
            #if len(self.__iterable)>0:# and type(self.__iterable[0]) in [int, float]:
            return np.any(self.__iterable)
            #for item in self.__iterable:
            #    if item:
            #        return True
            #return False

        predicate_func = trans_func(predicate)

        for item in self.__iterable:
            if predicate_func(item):
                return True
        return False

    def all(self, predicate=None) -> bool:
        self.calculate()
        if predicate == None:
            return np.all(self.__iterable)

        predicate_func = trans_func(predicate)

        for item in self.__iterable:
            if not predicate_func(item):
                return False
        return True

    def concat(self, other):
        self.calculate()
        return Qfrom(np.concatenate((self.__iterable, parse_iterable_to_array(other))))

    #def foreach(self, action):
    #    func = trans_func(action)
    #    if func is None:
    #        raise ValueError(str(action) + ' cant be interpreted as a function')
    #    
    #    for item in self.__iterable:
    #        func(item)



    #-- table func ----------------------------------------------#
    def edit_column(self, col_name, func):
        edit_func = trans_func(func)

        operation = {
            'Operation': Operation.EDIT_COLUMN,
            'col_name': col_name,
            'func': edit_func
        }
        return Qfrom(self.__iterable, operation_list=self.__operation_list+[operation])
    def edit(self, col_name, func):
        return self.edit_column(col_name, func)

    def rename_column(self, col_name=None, new_name=None):
        if new_name is None:
            raise ValueError('new_name must not be None')

        operation = {
            'Operation': Operation.RENAME_COLUMN,
            'col_name': col_name,
            'new_name': new_name
        }
        return Qfrom(self.__iterable, operation_list=self.__operation_list+[operation])
    def rename(self, col_name=None, new_name=None):
        return self.rename_column(col_name, new_name)

    def columns(self):
        self.calculate()
        if not any(self.__iterable):
            raise ValueError('There are no entries, therefore there are no columns')
        if type(self.__iterable[0]) == dict:
            return [key for key in self.__iterable[0]]
        if type(self.__iterable[0]) in [list, tuple]:
            return [i for i in range(len(self.__iterable[0]))]
        #raise ValueError('There are no columns')
        return None


    #-- math func -----------------------------------------------#
    def __add__(self, other):
        self.calculate()
        return Qfrom(self.__iterable + other)
    def __sub__(self, other):
        self.calculate()
        return Qfrom(self.__iterable - other)
    def __mul__(self, other):
        self.calculate()
        return Qfrom(self.__iterable * other)
    def __truediv__(self, other):
        self.calculate()
        return Qfrom(self.__iterable / other)
    def __pow__(self, other):
        self.calculate()
        return Qfrom(self.__iterable ** other)
    def __eq__(self, other) -> bool:
        #if any(self.__operation_list):
        self.calculate()
        if isinstance(other, Qfrom):
            return np.array_equal(self.__iterable, other())
        return False
    #def __lt__(self, other):
    #def __le__(self, other):
    #def __gt__(self, other):
    #def __ge__(self, other):

    def normalize(self, key=None):
        if key is None:
            min_value = abs(self.min())
            max_value = abs(self.max())
            div = max(min_value, max_value)
            if div != 0:
                return self.select(lambda x:x/div)
            raise ValueError('all item are 0. cant divide by 0')
        
        min_value = abs(self.select(lambda x:x[key]).min())
        max_value = abs(self.select(lambda x:x[key]).max())
        div = max(min_value, max_value)
        if div != 0:
            return self.select(lambda x: {**x, key:x[key]/div})
        raise ValueError(f'all item[{key}] are 0. cant divide by 0')
    def norm(self, key=None):
        return self.normalize(key)


    #-- random func ---------------------------------------------#
    def shuffle(self):
        self.calculate()
        shuffled_arr = np.copy(self.__iterable)
        np.random.shuffle(shuffled_arr)
        return Qfrom(shuffled_arr)
         
    


    #-- aggregation func ----------------------------------------#
    def aggregate(self, func):
        if len(self.__iterable) == 0:
            return None

        agg_func = trans_func(func)

        self.calculate()

        #if self.__iterable.dtype.type is np.str_:
        agg = self.__iterable[0]
        if len(self.__iterable) <= 1:
            return agg

        for item in self.__iterable[1:]:
            agg = agg_func(agg, item)
        return agg
        #else:
        #    agg_func_arr = np.frompyfunc(agg_func, 2, 1)
        #    agg = self.__iterable
        #    while len(agg) > 1:
        #        b = agg[1::2]
        #        if (len(agg) % 2) == 0:
        #            a = agg[::2]
        #        else:
        #            a = agg[:-1:2]
        #            b[-1] = agg_func(b[-1], agg[-1])
        #        agg = agg_func_arr(a,b)
        #    return agg[0]
    def agg(self, func):
        return self.aggregate(func)

    def min(self, func=None):
        self.calculate()
        if func == None:
            return np.min(self.__iterable)

        select_func = trans_func(func, False)
        return min(self.__iterable, key=select_func)
    def max(self, func=None):
        self.calculate()
        if func == None:
            return np.max(self.__iterable)

        select_func = trans_func(func, False)
        return max(self.__iterable, key=select_func)

    def sum(self, key=None):
        self.calculate()
        if key == None:
            return np.sum(self.__iterable)
        func = trans_func(key, False)
        col_arr = apply_func(func, self.__iterable)
        return np.sum(col_arr)
    def mean(self, key=None):
        self.calculate()
        if key == None:
            return np.mean(self.__iterable)
        func = trans_func(key, False)
        col_arr = apply_func(func, self.__iterable)
        return np.mean(col_arr)
    def median(self, key=None):
        self.calculate()
        if key == None:
            return np.median(self.__iterable)
        func = trans_func(key, False)
        col_arr = apply_func(func, self.__iterable)
        return np.median(col_arr)
    def var(self, key=None):
        self.calculate()
        if key == None:
            return np.var(self.__iterable)
        func = trans_func(key, False)
        col_arr = apply_func(func, self.__iterable)
        return np.var(col_arr)



    #-- extend func ---------------------------------------------#
    def __call__(self, *args, use_iterable=False):
        if not any(args):
            self.calculate()
            return self.__iterable

        func = trans_func(args[0])

        if not use_iterable and len(args) > 1:
            return func(self, *args[1:])
        elif not use_iterable and any(args):
            return func(self)
        
        self.calculate()

        if len(args) > 1:
            return func(self.__iterable, *args[1:])
        return func(self.__iterable)


    #-- special func --------------------------------------------#
    def calculate(self):
        if any(self.__operation_list):
            self.__iterable = calc_operations(np.copy(self.__iterable), self.__operation_list)
            self.__operation_list = []
        return self.__iterable
    
    def as_parallel(self):
        core_count = multiprocessing.cpu_count()
        
        result_array = np.copy(self.__iterable)
        pipe = []
        pool = ProcessingPool(nodes=core_count)
        for op in self.__operation_list:
            if op['Operation'] in [Operation.ORDER_BY, Operation.GROUP_BY]:
                data_fragment_array = np.array_split(result_array, core_count)

                calc_fragment_list = pool.map(worker_calc_operations, [(item, pipe) for item in data_fragment_array])
                #calc_fragment_list = None
                #with Pool(core_count) as p:
                #    calc_fragment_list = p.map(lambda x: calc_operations(x, pipe), data_fragment_array)

                result_array = np.concatenate(calc_fragment_list)

                result_array = calc_operations(result_array, [op])
                pipe = []
            else:
                pipe.append(op)
        if any(pipe):
            data_fragment_array = np.array_split(result_array, core_count)

            #pool = ProcessingPool(nodes=core_count)
            #calc_fragment_list = pool.map(lambda x: calc_operations(x, pipe), data_fragment_array)
            calc_fragment_list = pool.map(worker_calc_operations, [(item, pipe) for item in data_fragment_array])
            #calc_fragment_list = None
            #with Pool(core_count) as p:
            #    calc_fragment_list = p.map(lambda x: calc_operations(x, pipe), data_fragment_array)

            result_array = np.concatenate(calc_fragment_list)

        if pool is not None:
            pool.close()
            pool.join()
            pool.terminate()

        self.__iterable = result_array
        return self
         


    #-- plot func -----------------------------------------------#
    def plot(self, x=None, show_legend=True, title=None, x_scale_log=False, y_scale_log=False, axis=None) -> None:
        self.calculate()

        ax = axis
        if axis==None:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)

        if x is None:
            x = self.columns()[0]
        col_list = [col for col in self.columns() if col != x]

        x_list = self.select(lambda item:item[x])()
        ax.set_xlabel(x)
        for c in col_list:
            c_list = self.select(lambda item:item[c])()
            ax.plot(x_list, c_list, label=c)
        
        if x_scale_log:
            ax.set_xscale('log')
        if y_scale_log:
            ax.set_yscale('log')
        if show_legend:
            ax.legend()
        if title is not None:
            ax.set_title(title)
        
        if axis== None:
            plt.show()
        
    def plot_bar(self, x=None, show_legend=True, title='Bar plot', space=0.1, y_scale_log=False, axis=None) -> None:
        self.calculate()
        
        ax = axis
        if axis==None:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)

        col_list = self.columns()
        if x is None and col_list is None:
            x_list = np.arange(len(self.__iterable))
            ax.bar(x_list, self.__iterable)
        elif x is not None and col_list is None:
            raise ValueError('There is no column ' + str(x))
        elif x is None and col_list is not None:
            x_list = np.arange(len(self.__iterable))
            bar_size = (1-space)/len(col_list)
            for i in np.arange(len(col_list)):
                a = col_list[i]
                a_list = self.select(lambda item:item[a])()
                ax.bar(x_list + bar_size*i - (bar_size/2.0)*(len(col_list)-1), a_list, label=a, width=bar_size)
                
                # ! replace
                plt.xticks(x_list, x_list)
        else:
            col_list = [col for col in col_list if col != x]
            x_list = self.select(lambda item:item[x])()
            x_pos = np.arange(len(x_list))
            bar_size = (1-space)/len(col_list)
            for i in np.arange(len(col_list)):
                a = col_list[i]
                a_list = self.select(lambda item:item[a])()
                ax.bar(x_pos + bar_size*i - (bar_size/2.0)*(len(col_list)-1), a_list, label=a, width=bar_size)
                
                # ! replace
                plt.xticks(x_pos, x_list)
        
        if y_scale_log:
            ax.set_yscale('log')
        if show_legend and col_list is not None:
            ax.legend()
        if title is not None:
            ax.set_title(title)

        if axis==None:
            plt.show()

    def plot_histogram(self, density=False, bins=None, stacked=False, show_legend=True, title='Histogram plot', y_scale_log=False, ylabel='Frequency', axis=None) -> None:
        self.calculate()
        
        ax = axis
        if axis==None:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)

        if bins is None:
            bins = len(self.__iterable)
        
        col_list = self.columns()
        #if col_list is None:
        #    ax.hist(self.__iterable, density=density, bins=bins, stacked=stacked)
        #else:
        #    for c in col_list:
        #        c_list = self.select(lambda item:item[c])()
        #        ax.hist(c_list, label=c, density=density, bins=bins, stacked=stacked)
        if col_list is None:
            ax.hist(self.__iterable, density=density, bins=bins, stacked=stacked)
        #elif len(col_list) == 1:
        #    x = self.select(lambda item:item[col_list[0]])()
        #    ax.hist(x, label=col_list[0], density=density, bins=bins, stacked=stacked)
        else:
            x_multi = [self.select(lambda item:item[c])() for c in col_list]
            ax.hist(x_multi, label=col_list, density=density, bins=bins, stacked=stacked)

        #ax.set_xlabel(column)
        ax.set_ylabel(ylabel)
        
        if y_scale_log:
            ax.set_yscale('log')
        if show_legend and col_list is not None:
            ax.legend()
        if title is not None:
            ax.set_title(title)

        if axis==None:
            plt.show()

    def plot_boxplot(self, title='Boxplot', y_scale_log=False, axis=None) -> None:
        self.calculate()
        
        ax = axis
        if axis==None:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
        
        col_list = self.columns()
        if col_list is None:
            ax.boxplot(self.__iterable)
        else:
            data = [self.select(lambda item:item[a])() for a in col_list]
            ax.boxplot(data)
            plt.xticks(range(1, 1+len(col_list)), col_list)

        #ax.set_xlabel(column)
        #ax.set_ylabel('Frequency')
        
        if y_scale_log:
            ax.set_yscale('log')
        if title is not None:
            ax.set_title(title)

        if axis==None:
            plt.show()

    def plot_scatter(self, show_legend=True, title='Scatter plot', x_scale_log=False, y_scale_log=False, axis=None) -> None:
        self.calculate()
        
        ax = axis
        if axis==None:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)

        #x_list = self.select(lambda item:item[x])()
        #y_list = self.select(lambda item:item[y])()
        #ax.scatter(x_list, y_list)
        col_list = self.columns()
        for i in np.arange(0,len(col_list),2):
            a = col_list[i]
            b = col_list[i+1]
            a_list = self.select(lambda item:item[a])()
            b_list = self.select(lambda item:item[b])()
            ax.scatter(a_list, b_list, label=str(a)+', '+str(b))
        #ax.set_xlabel(x)
        #ax.set_ylabel(y)
        if x_scale_log:
            ax.set_xscale('log')
        if y_scale_log:
            ax.set_yscale('log')
        if show_legend:
            ax.legend()
        if title is not None:
            ax.set_title(title)

        if axis==None:
            plt.show()



    #-- export func ---------------------------------------------#
    def to_array(self):
        self.calculate()
        return self.__iterable

    def to_list(self) -> list:
        self.calculate()

        if len(self.__iterable)>0 and type(self.__iterable[0]) in [dict, list, tuple, set]:
            return list(self.__iterable)
        return list(self.__iterable.tolist())
    
    def to_set(self) -> set:
        self.calculate()
        return set(self.__iterable)

    def to_dict(self, key=None, value=None) -> dict:
        self.calculate()
        if not self.any():
            return dict()
        if key is None:
            if type(self.__iterable[0]) == dict:
                key = lambda x:x['key']
            if type(self.__iterable[0]) in [tuple, list]:
                key = lambda x:x[0]
        if value is None:
            if type(self.__iterable[0]) == dict:
                value = lambda x:x['value']
            if type(self.__iterable[0]) in [tuple, list]:
                value = lambda x:x[1]
        
        get_key_func = trans_func(key)
        get_value_func = trans_func(value)
        return dict((get_key_func(row), get_value_func(row)) for row in self.__iterable)

    def to_dataframe(self) -> pd.DataFrame:
        self.calculate()
        #if any(self.__iterable):
        #    data = dict()
        #    for key in self.__iterable[0]:
        #        data[key] = self.select(lambda x:x[key])()
        #    return pandas.DataFrame(data)
        #else:
        return pd.DataFrame(self.__iterable.tolist())

    def to_csv_str(self, delimiter=',') -> str:
        self.calculate()
        if not any(self.__iterable):
            return ''

        header = None
        data = []
        if type(self.__iterable[0]) == dict:
            header = [key for key in self.__iterable[0]]
            for row in self.__iterable:
                if type(row) != dict:
                    raise ValueError('cant identify data pattern for csv conversion')
                data.append([str(row[key]) for key in header])
        
        elif type(self.__iterable[0]) == list:
            for row in self.__iterable:
                if type(row) != list:
                    raise ValueError('cant identify data pattern for csv conversion')
                data.append([str(item) for item in row])
        else:
            data = [[str(row)] for row in self.__iterable]

        csv_str = '\n'.join([delimiter.join(line) for line in [header]+data])
        return csv_str

    def to_csv_file(self, path, encoding='UTF8', delimiter=',') -> None:
        self.calculate()
        if not any(self.__iterable):
            return
        header = None
        data = []
        if type(self.__iterable[0]) == dict:
            header = [key for key in self.__iterable[0]]
            for row in self.__iterable:
                if type(row) != dict:
                    raise ValueError('cant identify data pattern for csv conversion')
                data.append([str(row[key]) for key in header])
        
        elif type(self.__iterable[0]) == list:
            for row in self.__iterable:
                if type(row) != list:
                    raise ValueError('cant identify data pattern for csv conversion')
                data.append([str(item) for item in row])
        else:
            data = [[str(row)] for row in self.__iterable]

        with open(path, 'w', encoding=encoding, newline='') as f:
            writer = csv.writer(f, delimiter=delimiter)
            if header is not None:
                writer.writerow(header)
            writer.writerows(data)

    #def to_json_file(self):
