from collections.abc import Iterable
from typing import Any
import pandas as pd
import csv
import numpy as np
import enum
from collections.abc import Iterable
#import time
import matplotlib.pyplot as plt
import inspect
import itertools
from queue import LifoQueue

import re
import json
import os
import requests
import chardet
from io import StringIO



def split_func_str_by_var(func_str):
    #var_pattern = '[a-zA-Z\_\.0-9]'
    var_pattern = '[a-zA-Z\_0-9]'
    escape_pattern = '[\'\"]'
    var_state = False
    escape_state = False
    str_part_list = []
    new_word = ''
    for c in func_str:
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

def split_func_str_coma(func_str):
    coma_pattern = ','
    escape_patters = [
            ('\'', '\''),
            ('\"', '\"'),
            ('(', ')'),
            ('[', ']'),
            ('{', '}'),
        ]
    escape_start_to_id = {start:i for i, (start, end) in enumerate(escape_patters)}
    escape_end_to_id = {end:i for i, (start, end) in enumerate(escape_patters)}
    escape_stack = LifoQueue()
    str_part_list = []
    new_word = ''

    for c in func_str:
        if c in escape_start_to_id:
            escape_stack.put(escape_start_to_id[c])
        if c in escape_end_to_id:
            i = escape_end_to_id[c]
            top_item = escape_stack.get()
            if i != top_item:
                escape_stack.put(top_item)

        if escape_stack.empty() and c == coma_pattern:
            str_part_list.append(new_word.strip())
            new_word = ''
        else:
            new_word += c
    str_part_list.append(new_word.strip())
    return str_part_list
def split_col_in_source_target(col_str):
    parts = col_str.split(' as ')
    if len(parts) == 1:
        if re.search('^\w+$', parts[0]):
            return (parts[0], parts[0])
        else:
            return (parts[0], None)
    return (parts[0], parts[1])

def get_used_vars_from_func_str(predicate_str, keys):
    result = split_func_str_by_var(predicate_str)
    if len(result) > 1:
        word_list = result[1::2]
        py_keywords = [
            'and', 'or', 'not', 'in', 'is',
            'len', 'min', 'max', 'sum', 'mean', 'median', 
            'print', 'type',
            'for', 'if', 'while'
        ]
        word_list_no_bool_ops = [word for word in word_list if word not in py_keywords]
        word_list_no_values = [word for word in word_list_no_bool_ops if word not in ['None'] and re.search('^[0-9]+$', word) is None]
        filter_by_keys = [word for word in word_list_no_values if word in keys or re.search('^\w+$', word) is not None]
        
        return set(filter_by_keys)

    raise SyntaxError(predicate_str + ' cant be interpreted as a function')
def predicatestr_to_funcstr(predicate_str, keys):
    column_list = split_func_str_coma(predicate_str) if type(predicate_str) is str else predicate_str
    used_vars = set([var for col in column_list for var in get_used_vars_from_func_str(col, keys)])
    args_str = ", ".join(set(used_vars))
    func_body_str = None
    if len(column_list) == 1:
        func_body_str = column_list[0]
    else:
        func_body_str = ') and ('.join(column_list)
        func_body_str = f'({func_body_str})'
    return f'lambda {args_str}: {func_body_str}'

def selectarg_to_funcstr(select_arg, keys):
    #remove
    match = re.search('^(-\w+)(\s*\,\s*(-\w+))*', select_arg if type(select_arg) is str else ', '.join(select_arg))
    if match:
        cols_to_remove = ['"'+re.sub('^-', '', col.strip())+'"' for col in select_arg.split(',')] if type(select_arg) is str else ['"'+re.sub('^-', '', col)+'"' for col in select_arg]
        if len(cols_to_remove) == 1:
            return (None, None, 'lambda x: {key:value for key, value in x.items() if key != '+cols_to_remove[0]+'}')
        return (None, None, 'lambda x: {key:value for key, value in x.items() if key not in ['+', '.join(cols_to_remove)+']}')

    #select
    column_list = split_func_str_coma(select_arg) if type(select_arg) is str else select_arg
    column_list = [split_col_in_source_target(col) for col in column_list]
    id = 0
    for i in range(len(column_list)):
        if column_list[i][1] is None:
            column_list[i] = (column_list[i][0], id)
            id += 1

    select_join_parts = [(source, target) for source, target in column_list if (type(source) is str and re.search('^\w+$', source) is None) or (keys is not None and source not in keys)]
    select_join_func_used_vars = [var for source, target in select_join_parts for var in get_used_vars_from_func_str(source, keys)]
    
    # set() changes the order of vars
    select_join_func_used_vars_unique = []
    for var in select_join_func_used_vars:
        if var not in select_join_func_used_vars_unique:
            select_join_func_used_vars_unique.append(var)
    
    args_str = ", ".join(select_join_func_used_vars_unique)
    select_join_func_str = None
    if len(select_join_parts) == 1:
        select_join_func_str = f'lambda {args_str}: {select_join_parts[0][0]}'
    elif len(select_join_parts) > 1:
        inner_tuple_str = ', '.join([source for source, target in select_join_parts])
        select_join_func_str = f'lambda {args_str}: ({inner_tuple_str})'
    select_join_func_col_names = tuple(target for source, target in select_join_parts) if len(select_join_parts) else None

    source_target_pairs = [(source, target) if type(source) is not str or re.search('^\w+$', source) else (target, target) for source, target in column_list]
    source_target_pairs = [(f'"{source}"' if type(source) is str else source, f'"{target}"' if type(target) is str else target) for source, target in source_target_pairs]
    col_select_str = ", ".join([f'{target}: x[{source}]' for source, target in source_target_pairs])
    select_func_str = 'lambda x: {'+col_select_str+'}'

    return (select_join_func_str, select_join_func_col_names, select_func_str)

def trans_select_func(func, keys):
    if func is None:
        raise SyntaxError(str(func) + ' cant be interpreted as a function')
    if callable(func):
        return (None, None, None, func)
    if type(func) in [str, tuple, list, np.str_]:
        select_join_func_str, select_join_func_col_names, select_func_str = selectarg_to_funcstr(func, keys)
        #print(f'{func} -> {select_join_func_str}, {select_join_func_col_names}, {select_func_str}')
        return (eval(select_join_func_str) if select_join_func_str else None, select_join_func_col_names, eval(select_func_str), None)

    raise SyntaxError(str(func) + ' cant be interpreted as a function')
def trans_select_func_args(args, keys):
    selected_col_names = None
    map_func = None
    new_col_names = None
    select_join_func = None
    select_join_func_col_names = None
    select_func = None

    if len(args) == 3:
        selected_col_names, map_func, new_col_names = args
        selected_col_names = [col.strip() for col in selected_col_names.split(',')] if type(selected_col_names) is str else selected_col_names
        new_col_names = [name.strip() for name in new_col_names.split(',')] if type(new_col_names) is str else new_col_names
    elif len(args) == 2 and callable(args[1]):
        selected_col_names, map_func = args
        selected_col_names = [col.strip() for col in selected_col_names.split(',')] if type(selected_col_names) is str else selected_col_names
    elif len(args) == 2 and callable(args[0]):
        map_func, new_col_names = args
        new_col_names = [name.strip() for name in new_col_names.split(',')] if type(new_col_names) is str else new_col_names
    elif len(args) == 1:
        select_join_func, select_join_func_col_names, select_func, map_func = trans_select_func(args[0], keys)

    return (
        selected_col_names,
        map_func,
        new_col_names,
        select_join_func,
        select_join_func_col_names,
        select_func,
    )
def trans_predicate_func(func, keys):
    if func is None:
        raise SyntaxError(str(func) + ' cant be interpreted as a function')
    if callable(func):
        return func
    if type(func) in [str, tuple, list]:
        translated_func_str = predicatestr_to_funcstr(func, keys)
        #print(f'{func} -> {translated_func_str}')
        return eval(translated_func_str)

    raise SyntaxError(str(func) + ' cant be interpreted as a function')
def trans_predicate_func_args(args, keys):
    selected_col_names = None
    predicate_func = None

    if len(args) == 2:
        selected_col_names = [col.strip() for col in args[0].split(',')] if type(args[0]) is str else args[0]
        predicate_func = trans_predicate_func(args[1], keys)
    elif len(args) == 1:
        predicate_func = trans_predicate_func(args[0], keys)

    return selected_col_names, predicate_func

def trans_groupby_func_args(args, keys):
    selected_col_names = None
    map_func = None
    select_join_func = None
    select_join_func_col_names = None
    select_key_func = None
    select_func = None

    if len(args) == 3:
        selected_col_names, map_func, new_col_names = args
        selected_col_names = [col.strip() for col in selected_col_names.split(',')] if type(selected_col_names) is str else selected_col_names
        new_col_names = [name.strip() for name in new_col_names.split(',')] if type(new_col_names) is str else new_col_names
        select_func = trans_select_func(new_col_names, keys)[2]
    elif len(args) == 2 and callable(args[1]):
        selected_col_names, map_func = args
        selected_col_names = [col.strip() for col in selected_col_names.split(',')] if type(selected_col_names) is str else selected_col_names
    elif len(args) == 2 and callable(args[0]):
        map_func, new_col_names = args
        new_col_names = [name.strip() for name in new_col_names.split(',')] if type(new_col_names) is str else new_col_names
        select_func = trans_select_func(new_col_names, keys)[2]
    elif len(args) == 2:
        select_join_func, select_join_func_col_names, select_key_func, map_func = trans_select_func(args[0], keys)

        new_col_names = args[1]
        new_col_names = [name.strip() for name in new_col_names.split(',')] if type(new_col_names) is str else new_col_names
        select_func = trans_select_func(new_col_names, keys)[2]
    elif len(args) == 1:
        select_join_func, select_join_func_col_names, select_key_func, map_func = trans_select_func(args[0], keys)

    return (
        selected_col_names,
        map_func,
        select_join_func,
        select_join_func_col_names,
        select_key_func,
        select_func,
    )
def trans_flatten_func_args(args, keys):
    selected_col_names = None
    map_func = None
    select_join_func = None
    select_join_func_col_names = None
    select_key_func = None
    new_col_names = None

    if len(args) == 3:
        selected_col_names, map_func, new_col_names = args
        selected_col_names = [col.strip() for col in selected_col_names.split(',')] if type(selected_col_names) is str else selected_col_names
        new_col_names = [name.strip() for name in new_col_names.split(',')] if type(new_col_names) is str else new_col_names
    elif len(args) == 2 and callable(args[1]):
        selected_col_names, map_func = args
        selected_col_names = [col.strip() for col in selected_col_names.split(',')] if type(selected_col_names) is str else selected_col_names
    elif len(args) == 2 and callable(args[0]):
        map_func, new_col_names = args
        new_col_names = [name.strip() for name in new_col_names.split(',')] if type(new_col_names) is str else new_col_names
    elif len(args) == 2:
        select_join_func, select_join_func_col_names, select_key_func, map_func = trans_select_func(args[0], keys)

        new_col_names = args[1]
        new_col_names = [name.strip() for name in new_col_names.split(',')] if type(new_col_names) is str else new_col_names
    elif len(args) == 1:
        select_join_func, select_join_func_col_names, select_key_func, map_func = trans_select_func(args[0], keys)

    return (
        selected_col_names,
        map_func,
        select_join_func,
        select_join_func_col_names,
        select_key_func,
        new_col_names,
    )
def trans_agg_func_args(args, keys):
    selected_col_names = None
    map_func = None

    if len(args) == 2 and callable(args[1]):
        selected_col_names, map_func = args
        selected_col_names = [col.strip() for col in selected_col_names.split(',')] if type(selected_col_names) is str else selected_col_names
    elif len(args) == 1 and callable(args[0]):
        map_func = args[0]
    elif len(args) == 1:
        map_func = trans_select_func(args[0], keys)[0]

    return (
        selected_col_names,
        map_func,
    )

def try_parse_str(text):
    if text == 'None' or text.strip() == '':
        #return None
        return np.nan

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

HANDLE_COLLECTION_SPECIALCLASSES = {
        dict: {
            #'unpack': lambda input_dict: dict_kvp_iter(input_dict),
            #'unpack': lambda input_dict: np.array([{'key': key, 'value': value} for key, value in input_dict.items()]),
            #'pack': lambda dict_list: dict((row['key'], row['value']) for row in dict_list),
        },
        pd.core.frame.DataFrame: {
            #'unpack': lambda df: pandas_row_dict_iter(df),
            #'pack': lambda row_dict_list: pd.DataFrame(row_dict_list),
        },
    }
HANLDE_TABLE_SPECIALCLASSES = {
    pd.core.frame.DataFrame: {
            #'unpack': lambda df: pandas_row_dict_iter(df),
            #'pack': lambda row_dict_list: pd.DataFrame(row_dict_list),
        },
}
def parse_iterable_to_array(iterable):
    global HANDLE_COLLECTION_SPECIALCLASSES
    iter_type = type(iterable)
    if iter_type is np.ndarray:
        if len(iterable) > 0 and type(iterable[0]) is str:
            return iterable.astype('object')
        return iterable

    if iter_type in HANDLE_COLLECTION_SPECIALCLASSES:
        raise NotImplementedError('not implemented')
    
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
def parse_iterables_to_arrays(table_dict):
    global HANLDE_TABLE_SPECIALCLASSES

    table_dict_type = type(table_dict)
    if table_dict_type in HANLDE_TABLE_SPECIALCLASSES:
        raise NotImplementedError('not implemented')

    if table_dict_type is Qfrom:
        return table_dict.table_dict
    if table_dict_type is dict:
        return {col_name: parse_iterable_to_array(col) for col_name, col in table_dict.items()}
    raise ValueError(f'Parameter "table_dict" is no dict or Qfrom_tab. {table_dict=}')

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
            result = {col_name: np.array([try_parse_str(row[index]) for row in data]) for index, col_name in enumerate(header_list)}
            #result = np.array([{h: try_parse_str(v) for h, v in zip(header_list, row)} for row in data])
            return result
        else:
            return file_data
    except BaseException:
        pass
    
    raise ValueError('Cant interpret str-arg.')
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



def pass_none(func, *args, **kwrgs):
    '''
    returns wrapped func.\n
    the wrapper checks if an argument is None.\n
    in this case the wrapper returns None.
    '''
    return lambda *_args, **_kwrgs:\
        None\
            if any(a is None for a in _args) or\
                any(v is None for k,v in _kwrgs.items())\
            else\
                func(*_args, *args, **kwrgs, **_kwrgs)
    '''return lambda *_args, **_kwrgs:\
        np.nan\
            if any(a is None or a == np.nan for a in _args) or\
                any(v is None or v == np.nan for k,v in _kwrgs.items())\
            else\
                func(*_args, *args, **kwrgs, **_kwrgs)'''
def first(iterable, predicate_func=None):
    if predicate_func:
        for item in iterable:
            if predicate_func(item):
                return item
        return None
    return next(iter(iterable))
def list_to_array(l):
    a = np.empty(len(l), dtype=object)
    
    for i, item in enumerate(l):
        a[i] = item
    
    return a
def iter_table_dict(table_dict):
    return (
        tuple(col[i] for col in table_dict.values())\
            if len(table_dict.keys()) > 1\
            else\
                first(table_dict.values())[i]\
        for i in range(len(first(table_dict.values())))\
    )


def get_func_output_col_count(table_dict, func, do_pass_none=False):
    #if not list(table_dict.values())[0].any():
    if list(table_dict.values())[0].size == 0:
        return None
    var_names = func.__code__.co_varnames
    args = tuple(table_dict[var][0] if var in table_dict else 0 for var in var_names)
    output_first_row = pass_none(func)(*args) if do_pass_none else func(*args)
    output_col_count = 1
    if type(output_first_row) is tuple:
        output_col_count = len(output_first_row)
    return output_col_count
def get_func_args(table_dict, func):
    func_type = type(func)
    var_names = func.__code__.co_varnames
    #if 'i' in var_names and 'i' not in result_dict.keys():
    #    pass
    #if 'index' in var_names and 'index' not in result_dict.keys():
    #    pass
    args = tuple(table_dict[var] if var in table_dict else np.arange(len(list(table_dict.values())[0])) for var in var_names)
    
    return args
def get_key_array(table_dict, key_func):
    if key_func is None:
        raise ValueError('key_func cant be None')
    args = get_func_args(table_dict, key_func)
    key_func = np.frompyfunc(key_func, len(args), 1)
    return key_func(*args)
def group_by_dict(key_array):
    group_ids_dict = {}
    for i, key in enumerate(key_array):
        if key not in group_ids_dict:
            group_ids_dict[key] = []
        group_ids_dict[key].append(i)
    
    return group_ids_dict

def append_table_dict(table_dict, item):
    if type(item) is tuple:
        return {key:np.append(table_dict[key], [item[i]]) for i, key in enumerate(table_dict.keys())}
    if type(item) is dict:
        return {key:np.append(table_dict[key], [item[key]]) for key in table_dict.keys()}
    raise ValueError('item must be of type tuple or dict')
def order_by_table_dict(table_dict, key_dict, reverse, select_func):
    ids = np.lexsort(list(key_dict.values())[::-1])
    result_dict = None
    if reverse:
        result_dict = {col_name:col[ids][::-1] for col_name, col in table_dict.items()}
    else:
        result_dict = {col_name:col[ids] for col_name, col in table_dict.items()}
    if select_func:
        result_dict = select_func(result_dict)
    return result_dict
def map_table_dict(table_dict, selected_col_names, func, do_pass_none, out_col_names=None):
    args = tuple(table_dict[col] for col in selected_col_names) if selected_col_names else get_func_args(table_dict, func)
    output_col_count = get_func_output_col_count(table_dict, func, do_pass_none)
    func = np.frompyfunc(pass_none(func), len(args), output_col_count)\
        if do_pass_none\
        else\
            np.frompyfunc(func, len(args), output_col_count)

    result_array = func(*args)
    
    #print(f'{result_array}')
    #print(f'{out_col_names=}, {output_col_count=}')
    if out_col_names and output_col_count == len(out_col_names):
        if output_col_count == 1:
            return {out_col_names[0]:result_array}
        return {out_col_names[i]:col for i, col in enumerate(result_array)}
    elif out_col_names and output_col_count > len(out_col_names) and len(out_col_names) == 1:
        return {out_col_names[0]:result_array}

    if output_col_count == 1 and len(result_array) > 0 and type(first(result_array)) is dict:
        return {key:np.array([row[key] for row in result_array]) for key in first(result_array).keys()}
    if output_col_count == 1:
        return {0:result_array}
    return {i:col for i, col in enumerate(result_array)}
def where_table_dict(table_dict, selected_col_names, func):#, invert):
    args = tuple(table_dict[col] for col in selected_col_names) if selected_col_names else get_func_args(table_dict, func)
    output_col_count = get_func_output_col_count(table_dict, func)
    if output_col_count != 1:
        raise ValueError('predicate must return one bool not a tuple')
    #if invert:
    #    func = np.frompyfunc(lambda *t: not func(*t), len(args), output_col_count)
    #else:
    func = np.frompyfunc(func, len(args), output_col_count)
    where_filter = func(*args)
    where_filter = where_filter.astype('bool')
    result = {col_name:col[where_filter] for col_name, col in table_dict.items()}
    return result
def join_table_dict(table_dict, other, key_dict, join_left_outer=False, join_right_outer=False):
    if key_dict is None:
        key_dict = {key:key for key in set(table_dict.keys()) & set(other.keys())}
    
    table_dict_keys = table_dict[first(key_dict.keys())] if len(key_dict.keys()) == 1 else\
        [tuple(table_dict[key][i] for key in key_dict.keys()) for i in range(len(first(table_dict.values())))]
    other_keys = other[first(key_dict.values())] if len(key_dict.values()) == 1 else\
        [tuple(other[key][i] for key in key_dict.values()) for i in range(len(first(other.values())))]
    this_group_ids_dict = group_by_dict(table_dict_keys)
    other_group_ids_dict = group_by_dict(other_keys)

    none_id_table_dict = len(table_dict_keys)
    none_id_other = len(other_keys)
    table_dict_none_row = {key:np.append(col, [None]) for key, col in table_dict.items()}
    other_none_row = {key:np.append(col, [None]) for key, col in other.items()}

    result_pair_ids = []
    for key, group_ids in this_group_ids_dict.items():
        if key in other_group_ids_dict:
            result_pair_ids += list(itertools.product(group_ids, other_group_ids_dict[key]))
        elif join_left_outer:
            result_pair_ids += [(i, none_id_other) for i in group_ids]
    if join_right_outer:
        for key in set(other_group_ids_dict.keys()) - set(this_group_ids_dict.keys()):
            result_pair_ids += [(none_id_table_dict, i) for i in other_group_ids_dict[key]]

    this_ids = [pair[0] for pair in result_pair_ids]
    other_ids = [pair[1] for pair in result_pair_ids]

    this_result_dict = {key:col[this_ids] for key, col in table_dict_none_row.items()}
    other_result_dict = {key:col[other_ids] for key, col in other_none_row.items()}
    
    combine_func = np.frompyfunc(lambda a,b: b if a is None else a, 2, 1)
    result_dict = this_result_dict
    for key, col in other_result_dict.items():
        if key in result_dict:
            result_dict[key] = combine_func(result_dict[key], col)
        else:
            result_dict = {**result_dict, key:col}
    return result_dict
def join_cross_table_dict(table_dict, other):
    table_dict_ids = range(len(first(table_dict.values())))
    other_ids = range(len(first(other.values())))
    result_pair_ids = list(itertools.product(table_dict_ids, other_ids))

    this_result_ids = [pair[0] for pair in result_pair_ids]
    other_result_ids = [pair[1] for pair in result_pair_ids]

    this_result_dict = {col_name:col[this_result_ids] for col_name, col in table_dict.items()}
    other_result_dict = {col_name:col[other_result_ids] for col_name, col in other.items()}

    return {**this_result_dict, **other_result_dict}
def join_id_table_dict(table_dict, other, join_left_outer=False, join_right_outer=False):
    len_table_dict = len(first(table_dict.values()))
    len_other = len(first(other.values()))
    if len_table_dict == len_other:
        return {**table_dict, **other}
    elif join_left_outer and len_table_dict > len_other:
        dif = len_table_dict - len_other
        none_list = np.full(dif, None)
        return {**table_dict, **{key:np.append(col, none_list) for key, col in other.items()}}
    elif join_right_outer and len_table_dict < len_other:
        dif = len_other - len_table_dict
        none_list = np.full(dif, None)
        return {**{key:np.append(col, none_list) for key, col in table_dict.items()}, **other}
    elif len_table_dict > len_other:
        return {**{key:col[0:len_other] for key, col in table_dict.items()}, **other}
    else:
        return {**table_dict, **{key:col[0:len_table_dict] for key, col in other.items()}}
def concat_table_dict(table_dict, other, join_outer_left, join_outer_right):
    if join_outer_left and join_outer_right:
        if len(table_dict) == 0:
            return other
        if len(other) == 0:
            return table_dict
        len_table_dict = len(first(table_dict.values()))
        len_other = len(first(other.values()))
        return {key:np.append(
            table_dict[key] if key in table_dict else np.full(len_table_dict, None),
            other[key] if key in other else np.full(len_other, None)
            ) for key in set(table_dict.keys()) | set(other.keys())}
    elif join_outer_left:
        if len(table_dict) == 0:
            return table_dict
        len_table_dict = len(first(table_dict.values()))
        len_other = len(first(other.values()))
        return {key:np.append(
            col,
            other[key] if key in other else np.full(len_other, None)
            ) for key, col in table_dict.items()}
    elif join_outer_right:
        if len(other) == 0:
            return other
        len_table_dict = len(first(table_dict.values()))
        len_other = len(first(other.values()))
        return {key:np.append(
            table_dict[key] if key in table_dict else np.full(len_table_dict, None),
            col) for key, col in other.items()}
    else:
        if len(table_dict) == 0:
            return table_dict
        if len(other) == 0:
            return other
        len_table_dict = len(first(table_dict.values()))
        len_other = len(first(other.values()))
        return {key:np.append(table_dict[key], other[key]) for key in set(table_dict.keys()) & set(other.keys())}
def group_by_table_dict(table_dict, key_array, select_func):
    group_ids_dict = group_by_dict(key_array)

    group_array = np.empty(len(group_ids_dict), dtype=object)
    if select_func is None:
        for i, group in enumerate(group_ids_dict.values()):
            group_array[i] = Qfrom({key: col[group] for key, col in table_dict.items()})
    else:
        for i, group in enumerate(group_ids_dict.values()):
            group_array[i] = Qfrom(select_func({key: col[group] for key, col in table_dict.items()}))
    
    result_dict = {
        'key': np.array(list(group_ids_dict.keys())),
        'group': group_array}

    return result_dict
def flatten_table_dict(collection_list, new_col_names):
    #item_ids = [i for i, col in enumerate(collection_list) for item in col]
    result_array = np.array([item for col in collection_list for item in col])
    if len(result_array.shape) > 1:
        result_array = np.rot90(result_array)
        if new_col_names:
            return {new_col_names[i]:col for i, col in enumerate(result_array[::-1])}
        return {i:col for i, col in enumerate(result_array[::-1])}
    else:
        if new_col_names:
            return {new_col_names[0]:result_array}
        return {0:result_array}
def flattenjoin_table_dict(table_dict, collection_list, new_col_names):
    item_ids = [i for i, col in enumerate(collection_list) for item in col]
    expanded_dict = {key:col[item_ids] for key, col in table_dict.items()}

    result_array = np.array([item for col in collection_list for item in col])
    join_dict = None
    if len(result_array.shape) > 1:
        result_array = np.rot90(result_array)
        if new_col_names:
            join_dict = {new_col_names[i]:col for i, col in enumerate(result_array[::-1])}
        else:
            join_dict = {i:col for i, col in enumerate(result_array[::-1])}
    else:
        if new_col_names:
            join_dict = {new_col_names[0]:result_array}
        else:
            join_dict = {0:result_array}
    
    return {**expanded_dict, **join_dict}


def calc_operations(table_dict, operation_list):
    result_dict = table_dict

    for op in operation_list:
        #if not any(list(result_dict.values())[0]):
        #    return result_dict
        
        match op['Operation']:
            case Operation.APPEND:
                item = op['item']
                result_dict = append_table_dict(result_dict, item)
                continue
            case Operation.RENAME:
                if len(result_dict) == 0:
                    continue
                func = op['func']
                result_dict = func(result_dict)
            case Operation.ORDERBY:
                if len(result_dict) == 0:
                    continue
                selected_col_names = op['selected_col_names']
                map_func = op['map_func']
                select_join_func = op['select_join_func']
                select_join_func_col_names = op['select_join_func_col_names']
                select_key_func = op['select_key_func']
                select_func = op['select_func']
                pass_none = op['pass_none']
                reverse = op['reverse']

                key_dict = result_dict
                if map_func:
                    key_dict = map_table_dict(result_dict, selected_col_names, map_func, pass_none)
                else:
                    if select_join_func:
                        key_dict = {**result_dict, **map_table_dict(result_dict, selected_col_names, select_join_func, pass_none, select_join_func_col_names)}
                    key_dict = select_key_func(key_dict)
                result_dict = order_by_table_dict(result_dict, key_dict, reverse, select_func)
                continue
            case Operation.WHERE:
                if len(result_dict) == 0:
                    continue
                selected_col_names = op['selected_col_names']
                func = op['func']
                #invert = op['invert']
                result_dict = where_table_dict(result_dict, selected_col_names, func)#, invert)
                continue
            case Operation.SELECT:
                if len(result_dict) == 0:
                    continue
                selected_col_names = op['selected_col_names']
                map_func = op['map_func']
                new_col_names = op['new_col_names']
                select_join_func = op['select_join_func']
                select_join_func_col_names = op['select_join_func_col_names']
                select_func = op['select_func']
                pass_none = op['pass_none']

                if map_func:
                    result_dict = map_table_dict(result_dict, selected_col_names, map_func, pass_none, new_col_names)
                else:
                    if select_join_func:
                        result_dict = {**result_dict, **map_table_dict(result_dict, selected_col_names, select_join_func, pass_none, select_join_func_col_names)}
                    result_dict = select_func(result_dict)
                continue
            case Operation.SELECTJOIN:
                if len(result_dict) == 0:
                    continue
                selected_col_names = op['selected_col_names']
                map_func = op['map_func']
                new_col_names = op['new_col_names']
                select_join_func = op['select_join_func']
                select_join_func_col_names = op['select_join_func_col_names']
                select_func = op['select_func']
                pass_none = op['pass_none']

                if map_func:
                    result_dict = {**result_dict, **map_table_dict(result_dict, selected_col_names, map_func, pass_none, new_col_names)}
                else:
                    if select_join_func:
                        result_dict = {**result_dict, **map_table_dict(result_dict, selected_col_names, select_join_func, pass_none, select_join_func_col_names)}
                    result_dict = {**result_dict, **select_func(result_dict)}
                continue
            case Operation.JOIN:
                if len(result_dict) == 0:
                    continue
                other = op['other']
                key_dict = op['key_dict']
                join_outer_left = op['join_outer_left']
                join_outer_right = op['join_outer_right']
                result_dict = join_table_dict(result_dict, other, key_dict, join_outer_left, join_outer_right)
                continue
            case Operation.JOINCROSS:
                if len(result_dict) == 0:
                    continue
                other = op['other']
                result_dict = join_cross_table_dict(result_dict, other)
                continue
            case Operation.JOINID:
                if len(result_dict) == 0:
                    continue
                other = op['other']
                join_outer_left = op['join_outer_left']
                join_outer_right = op['join_outer_right']
                result_dict = join_id_table_dict(result_dict, other, join_outer_left, join_outer_right)
                continue
            case Operation.CONCAT:
                others = op['others']
                join_outer_left = op['join_outer_left']
                join_outer_right = op['join_outer_right']
                for table_dict in others:
                    result_dict = concat_table_dict(result_dict, table_dict, join_outer_left, join_outer_right)
                continue
            case Operation.GROUPBY:
                if len(result_dict) == 0:
                    continue
                selected_col_names = op['selected_col_names']
                map_func = op['map_func']
                select_join_func = op['select_join_func']
                select_join_func_col_names = op['select_join_func_col_names']
                select_key_func = op['select_key_func']
                select_func = op['select_func']
                pass_none = op['pass_none']

                key_dict = result_dict
                if map_func:
                    key_dict = map_table_dict(result_dict, selected_col_names, map_func, pass_none)
                else:
                    if select_join_func:
                        key_dict = {**result_dict, **map_table_dict(result_dict, selected_col_names, select_join_func, pass_none, select_join_func_col_names)}
                    key_dict = select_key_func(key_dict)
                key_list = list(iter_table_dict(key_dict))
                result_dict = group_by_table_dict(result_dict, key_list, select_func)
                continue
            case Operation.FLATTEN:
                if len(result_dict) == 0:
                    continue
                selected_col_names = op['selected_col_names']
                map_func = op['map_func']
                select_join_func = op['select_join_func']
                select_join_func_col_names = op['select_join_func_col_names']
                select_key_func = op['select_key_func']
                new_col_names = op['new_col_names']
                pass_none = op['pass_none']

                key_dict = result_dict
                if map_func:
                    key_dict = map_table_dict(result_dict, selected_col_names, map_func, pass_none)
                else:
                    if select_join_func:
                        key_dict = {**result_dict, **map_table_dict(result_dict, selected_col_names, select_join_func, pass_none, select_join_func_col_names)}
                    key_dict = select_key_func(key_dict)
                key_list = list(iter_table_dict(key_dict))
                result_dict = flatten_table_dict(key_list, new_col_names)
                continue
            case Operation.FLATTENJOIN:
                if len(result_dict) == 0:
                    continue
                selected_col_names = op['selected_col_names']
                map_func = op['map_func']
                select_join_func = op['select_join_func']
                select_join_func_col_names = op['select_join_func_col_names']
                select_key_func = op['select_key_func']
                new_col_names = op['new_col_names']
                pass_none = op['pass_none']

                key_dict = result_dict
                if map_func:
                    key_dict = map_table_dict(result_dict, selected_col_names, map_func, pass_none)
                else:
                    if select_join_func:
                        key_dict = {**result_dict, **map_table_dict(result_dict, selected_col_names, select_join_func, pass_none, select_join_func_col_names)}
                    key_dict = select_key_func(key_dict)
                key_list = list(iter_table_dict(key_dict))
                result_dict = flattenjoin_table_dict(result_dict, key_list, new_col_names)
                continue
    
    return result_dict


class Operation(enum.Enum):
    APPEND          = 1
    SELECT          = 2
    SELECTJOIN      = 3
    RENAME          = 4
    WHERE           = 5
    ORDERBY         = 6
    JOIN            = 7
    JOINCROSS       = 8
    JOINID          = 9
    CONCAT          = 10
    GROUPBY         = 11
    FLATTEN         = 12
    FLATTENJOIN     = 13

class Qfrom():
    def __init__(self, collection=None, operation_list=[]) -> None:
        #if type(table_dict) is str:
        #    self.table_dict =  parse_iterables_to_arrays(parse_str_to_collection(table_dict))
        #elif type(table_dict) in [list, np.ndarray]:
        #    self.table_dict =  parse_iterables_to_arrays({'x':table_dict})
        #else:
        #    self.table_dict =  parse_iterables_to_arrays(table_dict)
        self.table_dict = dict()
        if isinstance(collection, str):
            self.table_dict =  parse_iterables_to_arrays(parse_str_to_collection(collection))
        elif isinstance(collection, dict):
            if len(collection) == 0:
                self.table_dict = collection
            else:
                first_item = first(collection)
                if type(collection[first_item]) is not np.ndarray:
                    self.table_dict = {key:np.array(value) for key, value in collection.items()}
                else:
                    self.table_dict = collection
        elif isinstance(collection, Qfrom):
            collection.calculate()
            self.table_dict = {key:np.copy(value) for key, value in collection.table_dict.items()}
        elif isinstance(collection, Iterable):
            collection_list = list(collection)
            if len(collection_list) > 0:
                first_item = first(collection_list)
                if isinstance(first_item, dict):
                    self.table_dict = {key:np.array([item[key] for item in collection_list]) for key in first_item.keys()}
                elif isinstance(first_item, tuple):
                    self.table_dict = {i:np.array([item[i] for item in collection_list]) for i in range(len(first_item))}
                else:
                    self.table_dict = {0:np.array(collection_list)}
        
        self.__operation_list = operation_list

    #-- standart list func --------------------------------------#
    def __len__(self) -> int:
        if any(self.__operation_list):
            self.calculate()
        if len(self.table_dict) == 0:
            return 0
        return len(list(self.table_dict.values())[0])
    def size(self) -> int:
        return len(self)

    def __getitem__(self, *args):
        if any(self.__operation_list):
            self.calculate()
        
        if len(args) == 1:
            key = args[0]
            if type(key) is int:
                return tuple(col[key] for col in self.table_dict.values()) if len(list(self.table_dict.values())) != 1 else first(self.table_dict.values())[key]
            
            if type(key) is slice:
                result = {col_name: col[key] for col_name, col in self.table_dict.items()}
                return Qfrom(result)
        
        return self.select(*args)
        '''columns = key
        if type(key) is str:
            columns = tuple(col.strip() for col in columns.split(','))
        if type(columns) in [tuple, list]:
            if len(columns) == 1:
                return self.table_dict[columns[0]]
            else:
                return self.select(columns)'''
    
    def __setitem__(self, key, newvalue):
        if any(self.__operation_list):
            self.calculate()

        if type(key) is int:
            if type(newvalue) is tuple:
                for i, k in enumerate(self.table_dict.keys()):
                    self.table_dict[k][key] = newvalue[i]
                return
            if type(newvalue) is dict:
                for k, v in newvalue.items():
                    self.table_dict[k][key] = newvalue[k]
                return

            raise ValueError('if key is of type int, newvalue must be of type tuple or dict')

        columns = key
        if type(columns) is str:
            columns = tuple(col.strip() for col in columns.split(','))
        if type(columns) in [tuple, list]:
            newvaluedict = newvalue
            if type(newvaluedict) is list:
                if len(columns) == 1:
                    newvaluedict = {columns[0]: [row[0] for row in newvaluedict] if type(first(newvaluedict)) is tuple and len(first(newvaluedict)) == 1 else newvaluedict}
                else:
                    newvaluedict = {col:[row[i] for row in newvaluedict] for i, col in enumerate(columns)}
            if type(newvaluedict) is dict:
                for col in columns:
                    self.table_dict[col] = np.array(newvaluedict[col])
                return
            if type(newvalue) is Qfrom:
                if all(col in columns for col in newvalue.columns()):
                    for col in columns:
                        self.table_dict[col] = np.array(newvalue[col])
                    return
                else:
                    newvalue_cols = newvalue.columns()
                    for i, col in enumerate(columns):
                        self.table_dict[col] = np.array(newvalue[newvalue_cols[i]])
                    return

        raise ValueError('key must be of type int, str, tuple or list')

    def __iter__(self):
        self.calculate()
        if len(self.table_dict) == 0:
            return None
        return (
            tuple(col[i] for col in self.table_dict.values())\
                if len(self.table_dict.keys()) > 1\
                else\
                    first(self.table_dict.values())[i]\
            for i in range(len(first(self.table_dict.values())))
        )
    
    def append(self, item):
        operation = {
            'Operation': Operation.APPEND,
            'item': item
        }
        return Qfrom(
            self.table_dict,
            operation_list=self.__operation_list+[operation])

    def copy(self):
        #self.calculate()
        #new_table_dict = {key:np.copy(value) for key, value in self.table_dict.items()}
        return Qfrom(self)


    def __str__(self) -> str:
        self.calculate()
        
        if self.any() and len(self.columns()) == 1:
            return 'Qfrom\n'+\
                '\t'.join(str(key) for key in self.table_dict.keys())+'\n'+\
                '\n'.join(str(row) for row in self)
        elif self.any() and len(self.columns()) > 1:
            return 'Qfrom\n'+\
                '\t'.join(str(key) for key in self.table_dict.keys())+'\n'+\
                '\n'.join('\t'.join([str(v) for v in row]) for row in self)
        return 'Qfrom\nempty'
        
        #return 'Qfrom(' + str(self.table_dict) + ')'

    def __repr__(self) -> str:
        #return 'Qfrom(' + str(self.table_dict) + ')'
        return str(self)
    
    def __eq__(self, other) -> bool:
        #if any(self.__operation_list):
        self.calculate()
        if isinstance(other, Qfrom):
            return all(col_name in other.table_dict and np.array_equal(col, other.table_dict[col_name]) for col_name, col in self.table_dict.items())\
                and all(col_name in self.table_dict.keys() for col_name in other.table_dict.keys())
        return False

    def __contains__(self, item):
        self.calculate()

        if type(item) is tuple:
            candidate_ids = np.arange(len(self))
            for i, col in enumerate(self.table_dict.keys()):
                candidate_ids = np.where(self.table_dict[col][candidate_ids]==item[i])[0]
                if candidate_ids.size == 0:
                    return False
            return True
        if type(item) is dict:
            candidate_ids = np.arange(len(self))
            for col in self.table_dict.keys():
                candidate_ids = np.where(self.table_dict[col][candidate_ids]==item[col])[0]
                if candidate_ids.size == 0:
                    return False
            return True

        raise ValueError('item must be of type tuple or dict')
        


    #-- table func ----------------------------------------------#
    def rename(self, arg):
        rename_func = None
        arg_dict = arg

        if type(arg_dict) is str:
            column_list = split_func_str_coma(arg_dict)
            column_list = [split_col_in_source_target(col) for col in column_list]
            arg_dict = dict(column_list)
        if type(arg_dict) is dict:
            rename_func = lambda x: {arg_dict[col_name] if col_name in arg_dict else col_name : col for col_name, col in self.table_dict.items()}

        operation = {
            'Operation': Operation.RENAME,
            'func': rename_func
        }
        return Qfrom(
            self.table_dict,
            operation_list=self.__operation_list+[operation])

    def orderby(self, *args, pass_none=False, reverse=False):
        if type(reverse) is not bool:
            raise ValueError('reverse should be a boolean not a ' + str(type(reverse)))
        selected_col_names,\
        map_func,\
        select_join_func,\
        select_join_func_col_names,\
        select_key_func,\
        select_func\
            = trans_groupby_func_args(args, keys=self.columns())

        operation = {
            'Operation': Operation.ORDERBY,
            'selected_col_names': selected_col_names,
            'map_func': map_func,
            'select_join_func': select_join_func,
            'select_join_func_col_names': select_join_func_col_names,
            'select_key_func': select_key_func,
            'select_func': select_func,
            'pass_none': pass_none,
            'reverse': reverse
        }
        return Qfrom(
            self.table_dict,
            operation_list=self.__operation_list+[operation])
    def orderby_pn(self, *args, reverse=False):
        return self.orderby(*args, pass_none=True, reverse=reverse)

    def first(self, *args):
        if len(args) > 0:
            selected_col_names, mod_predicate = trans_predicate_func_args(args, keys=self.columns())
            var_names = mod_predicate.__code__.co_varnames
            col_id_dict = {col:i for i, col in enumerate(self.columns())}
            predicate_func = None
            if selected_col_names:
                predicate_func = lambda x: mod_predicate(*(x[col_id_dict[selected_col_names[i]]] for i, var in enumerate(var_names)))
            else:
                predicate_func = lambda x: mod_predicate(*(x[col_id_dict[var]] for var in var_names))
            #predicate_func = lambda x: mod_predicate(*(x[col_id_dict[selected_col_names[i]]] for i, var in enumerate(var_names))) if selected_col_names else lambda x: mod_predicate(*(x[col_id_dict[var]] for var in var_names))
            return first(self, predicate_func)
        return first(self)

    def columns(self):
        return tuple(self.table_dict.keys())

    def where(self, *args):
        selected_col_names, where_predicate = trans_predicate_func_args(args, keys=self.columns())

        operation = {
            'Operation': Operation.WHERE,
            'selected_col_names': selected_col_names,
            'func': where_predicate,
            #'invert': False,
        }
        return Qfrom(
            self.table_dict,
            operation_list=self.__operation_list+[operation])

    '''def split(self, *args):
        selected_col_names, where_predicate = trans_predicate_func_args(args, keys=self.columns())

        operation1 = {
            'Operation': Operation.WHERE,
            'selected_col_names': selected_col_names,
            'func': where_predicate,
            'invert': False,
        }
        operation2 = {
            'Operation': Operation.WHERE,
            'selected_col_names': selected_col_names,
            'func': where_predicate,
            'invert': True,
        }
        result = (
            Qfrom(
                self.table_dict,
                operation_list=self.__operation_list+[operation1]),
            Qfrom(
                self.table_dict,
                operation_list=self.__operation_list+[operation2]),
        )
        print(result)
        return result'''

    def select(self, *args, pass_none=False):
        selected_col_names,\
        map_func,\
        new_col_names,\
        select_join_func,\
        select_join_func_col_names,\
        select_func\
            = trans_select_func_args(args, keys=self.columns())

        operation = {
            'Operation': Operation.SELECT,
            'selected_col_names': selected_col_names,
            'map_func': map_func,
            'new_col_names': new_col_names,
            'select_join_func': select_join_func,
            'select_join_func_col_names': select_join_func_col_names,
            'select_func': select_func,
            'pass_none': pass_none,
        }
        return Qfrom(
            self.table_dict,
            operation_list=self.__operation_list+[operation])
    def select_pn(self, *args):
        return self.select(*args, pass_none=True)
    def select_join(self, *args, pass_none=False):
        selected_col_names,\
        map_func,\
        new_col_names,\
        select_join_func,\
        select_join_func_col_names,\
        select_func\
            = trans_select_func_args(args, keys=self.columns())

        operation = {
            'Operation': Operation.SELECTJOIN,
            'selected_col_names': selected_col_names,
            'map_func': map_func,
            'new_col_names': new_col_names,
            'select_join_func': select_join_func,
            'select_join_func_col_names': select_join_func_col_names,
            'select_func': select_func,
            'pass_none': pass_none,
        }
        return Qfrom(
            self.table_dict,
            operation_list=self.__operation_list+[operation])
    def select_join_pn(self, *args):
        return self.select_join(*args, pass_none=True)

    def shuffle(self):
        self.calculate()
        dict_shuffled = {key: np.random.shuffle(np.copy(col)) for key, col in self.table_dict.items()}
        return Qfrom(dict_shuffled)

    def join(self, other, key_dict=None, join_outer_left=False, join_outer_right=False):
        operation = {
            'Operation': Operation.JOIN,
            'other': other.table_dict,
            'key_dict': key_dict,
            'join_outer_left': join_outer_left,
            'join_outer_right': join_outer_right,
        }
        return Qfrom(
            self.table_dict,
            operation_list=self.__operation_list+[operation])
    def join_cross(self, other):
        operation = {
            'Operation': Operation.JOINCROSS,
            'other': other.table_dict,
        }
        return Qfrom(
            self.table_dict,
            operation_list=self.__operation_list+[operation])
    def join_outer(self, other, key_dict=None):
        return self.join(other, key_dict, join_outer_left=True, join_outer_right=True)
    def join_outer_left(self, other, key_dict=None):
        return self.join(other, key_dict, join_outer_left=True, join_outer_right=False)
    def join_outer_right(self, other, key_dict=None):
        return self.join(other, key_dict, join_outer_left=False, join_outer_right=True)
    def join_id(self, other, join_outer_left=False, join_outer_right=False):
        operation = {
            'Operation': Operation.JOINID,
            'other': other.table_dict,
            'join_outer_left': join_outer_left,
            'join_outer_right': join_outer_right,
        }
        return Qfrom(
            self.table_dict,
            operation_list=self.__operation_list+[operation])
    def join_id_outer(self, other):
        return self.join_id(other, join_outer_left=True, join_outer_right=True)
    def join_id_outer_left(self, other):
        return self.join_id(other, join_outer_left=True, join_outer_right=False)
    def join_id_outer_right(self, other):
        return self.join_id(other, join_outer_left=False, join_outer_right=True)

    def concat(self, other, join_outer_left=False, join_outer_right=False):
        operation = {
            'Operation': Operation.CONCAT,
            'others': [q.table_dict for q in other] if type(other) is list else [other.table_dict],
            'join_outer_left': join_outer_left,
            'join_outer_right': join_outer_right,
        }
        return Qfrom(
            self.table_dict,
            operation_list=self.__operation_list+[operation])
    def concat_outer(self, other):
        return self.concat(other, join_outer_left=True, join_outer_right=True)
    def concat_outer_left(self, other):
        return self.concat(other, join_outer_left=True, join_outer_right=False)
    def concat_outer_right(self, other):
        return self.concat(other, join_outer_left=False, join_outer_right=True)

    def groupby(self, *args, pass_none=False):
        selected_col_names,\
        map_func,\
        select_join_func,\
        select_join_func_col_names,\
        select_key_func,\
        select_func\
            = trans_groupby_func_args(args, keys=self.columns())

        operation = {
            'Operation': Operation.GROUPBY,
            'selected_col_names': selected_col_names,
            'map_func': map_func,
            'select_join_func': select_join_func,
            'select_join_func_col_names': select_join_func_col_names,
            'select_key_func': select_key_func,
            'select_func': select_func,
            'pass_none': pass_none,
        }
        return Qfrom(
            self.table_dict,
            operation_list=self.__operation_list+[operation])
    def groupby_pn(self, *args):
        return self.groupby(*args, pass_none=True)

    def flatten(self, *args, pass_none=False):
        selected_col_names,\
        map_func,\
        select_join_func,\
        select_join_func_col_names,\
        select_key_func,\
        new_col_names\
            = trans_flatten_func_args(args, keys=self.columns())

        operation = {
            'Operation': Operation.FLATTEN,
            'selected_col_names': selected_col_names,
            'map_func': map_func,
            'select_join_func': select_join_func,
            'select_join_func_col_names': select_join_func_col_names,
            'select_key_func': select_key_func,
            'new_col_names': new_col_names,
            'pass_none': pass_none,
        }
        return Qfrom(
            self.table_dict,
            operation_list=self.__operation_list+[operation])
    def flatten_join(self, *args, pass_none=False):
        selected_col_names,\
        map_func,\
        select_join_func,\
        select_join_func_col_names,\
        select_key_func,\
        new_col_names\
            = trans_flatten_func_args(args, keys=self.columns())

        operation = {
            'Operation': Operation.FLATTENJOIN,
            'selected_col_names': selected_col_names,
            'map_func': map_func,
            'select_join_func': select_join_func,
            'select_join_func_col_names': select_join_func_col_names,
            'select_key_func': select_key_func,
            'new_col_names': new_col_names,
            'pass_none': pass_none,
        }
        return Qfrom(
            self.table_dict,
            operation_list=self.__operation_list+[operation])

    def unique(self, *args, pass_none=False):
        return self\
            .groupby(*args, pass_none=pass_none)\
            .select(lambda group:group[0], self.columns(), pass_none=pass_none)

    def agg(self, *args):
        self.calculate()

        selected_col_names,\
        map_func\
            = trans_agg_func_args(args, keys=self.columns())

        var_names = map_func.__code__.co_varnames
        if selected_col_names:
            var_names = selected_col_names
        args = tuple(self.table_dict[var] for var in var_names)

        return map_func(*args)
        
    def agg_pairs(self, *args):
        self.calculate()

        if len(self) == 0:
            return None
        if len(self) == 1:
            return self[0]

        selected_col_names,\
        map_func\
            = trans_agg_func_args(args, keys=self.columns())

        var_names = map_func.__code__.co_varnames
        if selected_col_names:
            var_names = selected_col_names

        q_args = self[var_names]
        agg = q_args[0]

        if len(var_names) > 1:
            for item in q_args[1:]:
                map_func_args = tuple(zip(agg, item))
                agg = map_func(*map_func_args)
            return agg
        else:
            for item in q_args[1:]:
                agg = map_func((agg, item))
            return agg




        


    #-- expanded list func --------------------------------------#
    def any(self):
        self.calculate()
        if len(self.table_dict) == 0:
            return False
        if  len(list(self.table_dict.values())[0]) == 0:
            return False
        return True
    '''def any(self, predicate=None) -> bool:
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
        return False'''
    
    def min(self, *args):
        self.calculate()
        q_result = self.select(*args) if len(args) > 0 else self
        q_result.calculate()
        if len(q_result.columns()) == 1:
            return first(q_result.table_dict.values()).min()
        return tuple(col.min() for col in q_result.table_dict.values())
    
    def min_id(self, *args):
        self.calculate()
        q_result = self.select(*args) if len(args) > 0 else self
        q_result.calculate()
        if len(q_result.columns()) == 1:
            return int(np.argmin(first(q_result.table_dict.values())))
        return tuple(int(np.argmin(col)) for col in q_result.table_dict.values())
    
    def min_item(self, *args):
        self.calculate()
        q_result = self.select(*args) if len(args) > 0 else self
        q_result.calculate()
        if len(q_result.columns()) == 1:
            id = int(np.argmin(first(q_result.table_dict.values())))
            return self[id]
        ids = tuple(int(np.argmin(col)) for col in q_result.table_dict.values())
        return tuple(self[i] for i in ids)

    def max(self, *args):
        self.calculate()
        q_result = self.select(*args) if len(args) > 0 else self
        q_result.calculate()
        if len(q_result.columns()) == 1:
            return first(q_result.table_dict.values()).max()
        return tuple(col.max() for col in q_result.table_dict.values())
    
    def max_id(self, *args):
        self.calculate()
        q_result = self.select(*args) if len(args) > 0 else self
        q_result.calculate()
        if len(q_result.columns()) == 1:
            return int(np.argmax(first(q_result.table_dict.values())))
        return tuple(int(np.argmax(col)) for col in q_result.table_dict.values())
    
    def max_item(self, *args):
        self.calculate()
        q_result = self.select(*args) if len(args) > 0 else self
        q_result.calculate()
        if len(q_result.columns()) == 1:
            id = int(np.argmax(first(q_result.table_dict.values())))
            return self[id]
        ids = tuple(int(np.argmax(col)) for col in q_result.table_dict.values())
        return tuple(self[i] for i in ids)

    def sum(self, *args):
        self.calculate()
        q_result = self.select(*args) if len(args) > 0 else self
        q_result.calculate()
        if len(q_result.columns()) == 1:
            return first(q_result.table_dict.values()).sum()
        return tuple(col.sum() for col in q_result.table_dict.values())

    def mean(self, *args):
        self.calculate()
        q_result = self.select(*args) if len(args) > 0 else self
        q_result.calculate()
        if len(q_result.columns()) == 1:
            return np.mean(first(q_result.table_dict.values()))
        return tuple(np.mean(col) for col in q_result.table_dict.values())

    def median(self, *args):
        self.calculate()
        q_result = self.select(*args) if len(args) > 0 else self
        q_result.calculate()
        if len(q_result.columns()) == 1:
            return np.median(first(q_result.table_dict.values()))
        return tuple(np.median(col) for col in q_result.table_dict.values())

    def var(self, *args):
        self.calculate()
        q_result = self.select(*args) if len(args) > 0 else self
        q_result.calculate()
        if len(q_result.columns()) == 1:
            return np.var(first(q_result.table_dict.values()))
        return tuple(np.var(col) for col in q_result.table_dict.values())

    #-- special func --------------------------------------------#
    def calculate(self):
        if any(self.__operation_list):
            self.table_dict = calc_operations(dict(self.table_dict), self.__operation_list)
            self.__operation_list = []
        return self.table_dict
    
    def __call__(self, func: callable, *args: Any, **kwds: Any) -> Any:
        return func(self, *args, **kwds)

    
    #-- export func ---------------------------------------------#
    def tolist(self) -> list:
        self.calculate()

        if len(self.table_dict.keys()) == 1:
            return list(first(self.table_dict.values()))
        if len(self.table_dict.keys()) > 1:
            return [self[i] for i in range(len(first(self.table_dict.values())))]
        return []

    def todict(self):
        self.calculate()
        return self.table_dict
    
    def toarray(self) -> np.array:
        self.calculate()

        if len(self.table_dict) == 0:
            return np.empty()
        if len(self.table_dict) == 1:
            return first(self.table_dict.values())

        result = np.array(list(self.table_dict.values())[::-1])
        result = np.rot90(result, 3)
        return result

    def tocsv(self, delimiter=',', header=True) -> str:
        self.calculate()
        if len(self) == 0:
            return ''
        
        header_str = delimiter.join([str(key) for key in self.columns()])
        data = [delimiter.join([str(item) for item in row]) for row in self]
        data_str = '\n'.join(data)

        return f'{header_str}\n{data_str}' if header else data_str


    #-- plot func -----------------------------------------------#
    def plot(self, x=None, show_legend=True, title=None, x_scale_log=False, y_scale_log=False, axis=None, figsize=None, order_by_x=True) -> None:
        self.calculate()

        ax = axis
        if axis==None:
            fig = plt.figure()
            if figsize:
                fig.set_figwidth(figsize[0])
                fig.set_figheight(figsize[1])
            ax = fig.add_subplot(1,1,1)

        if x is None:
            x = self.columns()[0]
        
        q_data = self.orderby(x) if order_by_x else self
        q_data.calculate()
        col_list = [col for col in q_data.columns() if col != x]

        x_list = q_data[x].toarray()
        ax.set_xlabel(x)
        for c in col_list:
            c_list = q_data[c].toarray()
            ax.plot(x_list, c_list, label=c)
        
        if x_scale_log:
            ax.set_xscale('log')
        if y_scale_log:
            ax.set_yscale('log')
        if show_legend:
            ax.legend()
        if title is not None:
            ax.set_title(title)
        
        if axis == None:
            plt.show()