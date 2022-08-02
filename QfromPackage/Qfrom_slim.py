from __future__ import annotations
from email.policy import default
from pickle import FALSE
import numpy as np
import pandas as pd
import re
import os
import json
import requests
import chardet
import inspect
import enum
import itertools
from typing import Any, overload
from collections import deque
from collections.abc import Iterable, Callable
import csv
from io import StringIO
import matplotlib.pyplot as plt
import time




# custom types
Table = dict[str, np.ndarray]
Selection = str|tuple[str]|list[str]
ColFunc = Callable[[np.ndarray], np.ndarray | tuple[np.ndarray] | Table]
AggFunc = Callable[[np.ndarray],Any]
TableFunc = Callable[[Table], Table]


def first(iterable, predicate_func=None):
    if predicate_func:
        return next((item for item in iterable if predicate_func(item)), None)
    return next(iter(iterable))

def split_iterable(list_to_split: Iterable, predicate: callable) -> tuple[list, list]:
    true_list, false_list = [], []
    for item in list_to_split:
        (false_list, true_list)[predicate(item)].append(item)
    return true_list, false_list
def split_list_by_value(list_to_split: list, value: Any) -> tuple:
    value_ids = [i for i, item in enumerate(list_to_split) if item == value]
    value_ids.append(len(list_to_split))

    start_id = 0
    splited_lists = []
    for end_id in value_ids:
        splited_lists.append(list_to_split[start_id:end_id])
        start_id = end_id+1
    return splited_lists
   
#def iter_table_dict(table_dict: dict):
#    return iter_array_list(list(table_dict.values()))
def iter_array_list(array_list: list[np.ndarray]) -> Iterable|None:
    # sourcery skip: assign-if-exp
    if not array_list:
        return None
    if len(array_list) == 1:
        return iter(array_list[0])
        #return np.nditer(array_list[0], flags=['refs_ok'])
    else:
        return zip(*array_list)
        #return np.nditer(array_list, flags=['refs_ok'])

def arr_set_value(arr: np.array, key: int, value):
    value_type = np.array([value]).dtype
    if arr.dtype < value_type:
        arr = arr.astype(value_type) 
    arr[key] = value
    return arr

def array_tuple_to_tuple_array(array_list: tuple[np.ndarray]|list[np.ndarray]) -> np.ndarray[tuple]:
        l = list(zip(*array_list))
        #it = np.nditer(array_list, flags=['refs_ok'])
        a = np.empty(len(array_list[0]), dtype=object)
        a[:] = l
        return a

def optimize_array_dtype(array):
    return parse.list_to_array(list(array))



def get_keys_from_func(func, kwrgs=True):
    #names = tuple(func.__code__.co_varnames)
    sig = inspect.signature(func)
    names = sig.parameters.keys()
    paras = [str(v) for v in sig.parameters.values()]
    #return tuple('...' if '*' in para else n for n, para in zip(names, paras))
    if kwrgs:
        return tuple('**' if '**' in para else '*' if '*' in para else n for n, para in zip(names, paras))
    return tuple('*' if '*' in para else n for n, para in zip(names, paras) if '=' not in para and '**' not in para)




class parse():
    @classmethod
    def try_str(cls, text: str) -> np.nan|int|float|str:
        if text == 'None' or not text.strip():
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

    @classmethod
    def iterables_to_arrays(cls, table_dict: Qfrom|Table) -> Table:
        table_dict_type = type(table_dict)

        if table_dict_type is Qfrom:
            return {key: np.copy(col) for key, col in table_dict.table_dict}
        if table_dict_type is dict:
            return {col_name: cls.iter_to_array(col) for col_name, col in table_dict.items()}
        raise ValueError(f'Parameter "table_dict" is no dict or Qfrom_tab. {table_dict=}')

    @classmethod
    def try_str_to_collection(
        cls,
        text: str,
        delimiter: str = ',',
        headers: bool = True
        ) -> Table:
        #try json
        try:
            mod_json_str = re.sub("\'", '\"', text)
            #mod_json_str = text.replace("\'", '\"')
            result = json.loads(mod_json_str)
            return cls.iterables_to_arrays(result)
        except BaseException:
            pass

        #try csv
        try:
            reader = csv.reader(StringIO(text), delimiter=delimiter)
            file_data = np.array([[cls.try_str(item.strip()) for item in row] for row in reader if ''.join(row).strip() != ''])
            if headers:
                header_list = file_data[0]
                data = file_data[1:]
                result = {col_name: np.array([cls.try_str(row[index]) for row in data]) for index, col_name in enumerate(header_list)}
                #result = np.array([{h: try_parse_str(v) for h, v in zip(header_list, row)} for row in data])
                return result
            else:
                return file_data
        except BaseException:
            pass
        
        raise ValueError('Cant interpret str-arg.')

    @classmethod
    def str_to_collection(
        cls,
        text: str,
        delimiter: str = ',',
        headers: bool = True
        ) -> Table:
        if re.search('^(https://|http://)', text) is not None:
            return cls.iterables_to_arrays(requests.get(text).json())

        col_text = text
        if os.path.exists(text):
            with open(text, 'rb') as f_rawdata:
                rawdata = f_rawdata.read()
            encoding = chardet.detect(rawdata)['encoding']

            with open(text, newline='', encoding=encoding) as f_col_text:
                col_text = f_col_text.read()
        return cls.try_str_to_collection(col_text, delimiter, headers)

    @classmethod
    def __will_be_transformed_to_array(cls, data) -> bool:
        class_dir = dir(type(data))
        return '__len__' in class_dir and '__getitem__' in class_dir
    @classmethod
    def list_to_array(cls, l: list) -> np.ndarray:
        if not l or not cls.__will_be_transformed_to_array(l[0]):
            return np.array(l)
        a = np.empty(len(l), dtype=object)
        a[:] = l
        return a
    @classmethod
    def iter_to_array(cls, iterable: Iterable) -> np.ndarray:
        iter_type = type(iterable)
        if iter_type is np.ndarray:
            return np.copy(iterable)
        
        if isinstance(iterable, list):
            return cls.list_to_array(iterable)
            
        first_item = next(iter(iterable))
        if cls.__will_be_transformed_to_array(first_item):
            l = list(iterable)
            a = np.empty(len(l), dtype=object)
            a[:] = l
            return a
        else:
            t = str(np.array(first_item).dtype)
            return np.fromiter(iterable, dtype=t)

    @classmethod
    def __seperate_keys_from_rest(cls, select_t: list[str]):
        def is_key(item):
            return item not in ['*', '...', '.'] and not callable(item)

        sep_select_t = []
        current_list = []
        last_item_is_key = None
        last_item_is_key = False

        for item in select_t:
            item_is_key = is_key(item)
            #if last_item_is_key is None or item_is_key == last_item_is_key:
            if item_is_key == last_item_is_key:
                current_list.append(item)
            else:
                sep_select_t.append(current_list)
                current_list = [item]
            
            last_item_is_key = item_is_key

        sep_select_t.append(current_list)
        return sep_select_t
    @classmethod
    def __get_key_intervals(cls, sep_select_t: list[list[str]], keys: tuple[str]):
        key_items_list = sep_select_t[1::2]
        key_items_list.append(None)
        #print(f'{no_key_items=}, {key_items=}')

        start_id = 0
        intervals = []
        for key_items in key_items_list:
            end_id = len(keys)
            if key_items:
                end_id = keys.index(key_items[0])
            intervals.append(keys[start_id:end_id])
            if key_items:
                start_id = keys.index(key_items[-1])+1

        return intervals
    @classmethod
    def __seperate_not_key_part(cls, no_key_part) -> tuple:
        prefix_group = []
        point_group = []
        suffix_group = []
        points_occured = False

        for item in no_key_part:
            if item == '...':
                points_occured = True
                point_group.append(item)
            elif points_occured:
                suffix_group.append(item)
            else:
                prefix_group.append(item)

        return prefix_group, point_group, suffix_group
    @classmethod
    def __partition_keys(cls, sep_no_key_parts, intervals, keys):
        keys = list(keys)
        #print(f'{sep_no_key_parts=}, {intervals=}')
        no_key_parts = []
        for (prefix_group, point_group, suffix_group), interval\
            in zip(sep_no_key_parts, intervals):
            if len(prefix_group) == 1 and prefix_group[0] == '*':
                no_key_parts.append(keys)
            else:
                mod_interval = list(interval)
                mod_prefix_group = [mod_interval.pop(0) for keys in prefix_group for _ in keys]
                mod_suffix_group = [key for keys in suffix_group[::-1] for key in [mod_interval.pop() for _ in keys][::-1]][::-1]
                mod_point_group = mod_interval if len(point_group) > 0 else []

                no_key_parts.append(mod_prefix_group+mod_point_group+mod_suffix_group)

        return no_key_parts
    @classmethod
    def selection(cls, selection: Selection, keys: tuple[str]) -> list[str]:
        if selection is None:
            return None
        if type(selection) is str:
            selection = tuple(item.strip() for item in selection.split(','))
        if type(keys) is not tuple:
            keys = tuple(keys)
        if all(s in keys for s in selection):
            return selection

        # check mapping element types and names
        special_str = ('*','...','.')
        possible_keys = keys+special_str
        not_valid_keys = [key not in possible_keys for key in selection]
        if all(not_valid_keys):
            raise ValueError(f'{not_valid_keys} are no valid keys')

        sep_select_t = cls.__seperate_keys_from_rest(selection)
        intervals = cls.__get_key_intervals(sep_select_t, keys)

        #check for not_key_parts conditions
        no_key_parts = sep_select_t[::2]
        key_parts = sep_select_t[1::2]
        if any('*' in part and len(part) > 1 for part in no_key_parts):
            raise ValueError("'*' cant be the direct neighbor of ['...', '.', func]")
        if any(len([item for item in part if item == '...']) > 1 for part in no_key_parts):
            raise ValueError("there can not be more than one '...' between two keys.")

        sep_no_key_parts = [cls.__seperate_not_key_part(part) for part in no_key_parts]
        no_key_parts = cls.__partition_keys(sep_no_key_parts, intervals, keys)

        #key_parts = list(key_parts)
        sep_select_t[::2] = no_key_parts
        sep_select_t[1::2] = key_parts

        return tuple(item for part in sep_select_t for item in part)

    @classmethod
    def dict_to_table(cls, d: dict) -> Table:
        result_table = d
        if any(type(key) is not str for key in result_table):
            result_table = {str(key): value for key, value in result_table.items()}
        if any(type(value) is not np.ndarray for value in result_table.values()):
            result_table = {key: cls.iter_to_array(value) for key, value in result_table.items()}
        
        if result_table:
            first_col_size = first(result_table.values()).size
            if any(value.size != first_col_size for value in result_table.values()):
                raise ValueError('All Iterables must have the same length')

        return result_table


def get_keys_from_func_args(func: callable, keys):
    func_args = get_keys_from_func(func)
    kwrgs = False

    unused_keys = list(keys)
    select_args = []
    for arg in func_args:
        if arg == '**':
            select_args += unused_keys
            kwrgs = True
            break
        if arg == '*':
            select_args += unused_keys
            break
        elif arg not in unused_keys:
            raise ValueError(f"callable argument '{arg}' cant be found in keys {keys}")
        else:
            unused_keys.remove(arg)
            select_args.append(arg)
    return select_args, kwrgs





def group_by_dict(key_iter: Iterable):
    group_ids_dict = {}
    for i, key in enumerate(key_iter):
        #group_ids_dict.get(key, deque()).append(i)
        if key not in group_ids_dict:
            #group_ids_dict[key] = []
            group_ids_dict[key] = deque()
        group_ids_dict[key].append(i)
    
    return group_ids_dict
def key_array_to_value_counts(key_iter: Iterable) -> Table:
    unique_id_dict = {}
    for key in key_iter:
        #print(f'{key=}')
        if key not in unique_id_dict:
            unique_id_dict[key] = 0
        unique_id_dict[key] += 1
    
    return {
        #'value': parse.list_to_array(list(unique_id_dict.keys())),
        'value': parse.iter_to_array(unique_id_dict.keys()),
        #'count': np.array(list(unique_id_dict.values()))
        'count': parse.iter_to_array(unique_id_dict.values())
        }



class table():
    @classmethod
    def __get_keys_array_list(
        cls,
        table_dict: Table,
        selection: Selection,
        func: ColFunc = None
        ) -> list[np.ndarray]:

        selection = parse.selection(selection, table_dict.keys())
        if not func:
            return [table_dict[key] for key in selection]
        key_dict = table.map(selection, func)(table_dict)
        return list(key_dict.values())

    @classmethod
    def append(cls, item: tuple|dict[str,Any]) -> TableFunc:
        def table_func(table_dict: Table) -> Table:
            table_dict = parse.dict_to_table(table_dict)
            if type(item) is tuple:
                if not table_dict and len(item) == 1:
                    return {'y': parse.list_to_array([item[0]])}
                if not table_dict:
                    return {f'y{i}': parse.list_to_array([value]) for i, value in enumerate(item)}
                return {key:np.append(col, [item[i]]) for i, (key, col) in enumerate(table_dict.items())}

            if type(item) is dict:
                return {key: np.append(col, [item[key]]) for key, col in table_dict.items()} if table_dict else {key: parse.list_to_array([value]) for key, value in item.items()}

            if not table_dict:
                return {'y': parse.list_to_array([item])}
            key, col = first(table_dict.items())
            return {key:np.append(col, [item])}
        return table_func

    @classmethod
    def select(cls, selection: Selection) -> TableFunc:
        def table_func(table_dict: Table) -> Table:
            table_dict = parse.dict_to_table(table_dict)
            if not table_dict:
                return table_dict
            selected_keys = parse.selection(selection, keys=table_dict.keys())
            #return {key: table_dict[key] for key in selection}
            return {key: col for key, col in table_dict.items() if key in selected_keys}
        return table_func
    
    @classmethod
    def map(
        cls,
        args: Selection = None,
        func: ColFunc = None,
        out: Selection = None
        ) -> TableFunc:
        def table_func(table_dict: Table) -> Table:
            table_dict = parse.dict_to_table(table_dict)
            if not table_dict:
                return table_dict

            selected_keys = parse.selection(args, keys=table_dict.keys())

            out_keys = out
            if type(out_keys) is str:
                out_keys = [key.strip() for key in out_keys.split(',')]

            kwrgs = False
            func_result = None
            if selected_keys is None:
                selected_keys, as_kwrgs = get_keys_from_func_args(func, table_dict.keys())
            
                if as_kwrgs:
                    kwrgs_cols = {key:table_dict[key] for key in selected_keys}
                    func_result = func(**kwrgs_cols)
                else:
                    arg_cols = tuple(table_dict[key] for key in selected_keys)
                    func_result = func(*arg_cols)
            elif func:
                keys = get_keys_from_func(func)
                if '**' in keys:
                    kwrgs = selected_keys[len(keys)-1:] 
                    selected_keys = selected_keys[:len(keys)-1]
                    kwrgs_cols = {key:table_dict[key] for key in kwrgs}
                    arg_cols = tuple(table_dict[key] for key in selected_keys)
                    func_result = func(*arg_cols, **kwrgs_cols)
                else:
                    arg_cols = tuple(table_dict[key] for key in selected_keys)
                    func_result = func(*arg_cols)
            elif len(selected_keys) > 1:
                arg_cols = tuple(table_dict[key] for key in selected_keys)
                func_result = arg_cols
            else:
                func_result = table_dict[selected_keys[0]]

            #func_result = optimize_array_dtype(func_result)

            if type(func_result) is tuple and out_keys is not None:
                return {out_keys[i]: col for i, col in enumerate(func_result)}
            if type(func_result) is tuple and len(selected_keys) > 0 and len(func_result) == 1:
                return {selected_keys[0]: func_result[0]}
            if type(func_result) is tuple and len(selected_keys) > 0:
                return {f'{selected_keys[0]}{i}': col for i, col in enumerate(func_result)}
            if type(func_result) is tuple:
                return {f'y{i}': col for i, col in enumerate(func_result)}
            if type(func_result) is dict:
                return func_result

            if inspect.isgenerator(func_result) and len(table_dict) > 0 and out_keys is not None:
                first_col = first(table_dict.values())
                return {out_keys[0]: parse.list_to_array([next(func_result) for _ in first_col])}
            if inspect.isgenerator(func_result) and len(table_dict) > 0 and len(selected_keys) > 0:
                first_col = first(table_dict.values())
                return {selected_keys[0]: parse.list_to_array([next(func_result) for _ in first_col])}
            if inspect.isgenerator(func_result) and len(table_dict) > 0:
                first_col = first(table_dict.values())
                return {'y': parse.list_to_array([next(func_result) for _ in first_col])}

            #if type(func_result) is not np.ndarray and isinstance(func_result, Iterable):
            if type(func_result) is not np.ndarray and type(func_result) is list:
                func_result = parse.list_to_array(func_result)
            if type(func_result) is not np.ndarray and type(func_result) is Iterable:
                func_result = parse.iter_to_array(func_result)
            if type(func_result) is np.ndarray and out_keys is not None:
                return {out_keys[0]: func_result}
            if type(func_result) is np.ndarray and len(selected_keys) > 0:
                return {selected_keys[0]: func_result}
            if type(func_result) is np.ndarray:
                return {'y': func_result}

            if table_dict and out_keys is not None:
                first_col = first(table_dict.values())
                return {out_keys[0]: parse.list_to_array([func_result for _ in first_col])}
            if table_dict and len(selected_keys) > 0:
                first_col = first(table_dict.values())
                return {selected_keys[0]: parse.list_to_array([func_result for _ in first_col])}
            if table_dict:
                first_col = first(table_dict.values())
                return {'y': parse.list_to_array([func_result for _ in first_col])}

            raise ValueError('cant interpret func result')    
        return table_func
    
    @classmethod
    def map_join(
        cls,
        args: Selection = None,
        func: ColFunc = None,
        out: Selection = None
        ) -> TableFunc:
        map_func = table.map(args, func, out)
        def table_func(table_dict: Table) -> Table:
            table_dict = parse.dict_to_table(table_dict)
            mapped_dict = map_func(table_dict)
            return table_dict | mapped_dict
        return table_func

    @classmethod
    def join(
        cls, 
        other: Table, 
        key_dict: dict[str, str] = None, 
        join_left_outer: bool = False, 
        join_right_outer: bool = False
        ) -> TableFunc:
        def table_func(table_dict: Table) -> Table:
            table_dict = parse.dict_to_table(table_dict)
            if not table_dict:
                return table_dict

            _key_dict = key_dict
            if _key_dict is None:
                _key_dict = {key:key for key in set(table_dict.keys()) & set(other.keys())}
            
            table_dict_keys = table_dict[first(_key_dict.keys())] if len(_key_dict.keys()) == 1 else\
                [tuple(table_dict[key][i] for key in _key_dict.keys()) for i in range(len(first(table_dict.values())))]
            other_keys = other[first(_key_dict.values())] if len(_key_dict.values()) == 1 else\
                [tuple(other[key][i] for key in _key_dict.values()) for i in range(len(first(other.values())))]
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
        return table_func
    
    @classmethod
    def join_cross(cls, other: Table) -> TableFunc:
        def table_func(table_dict: Table) -> Table:
            table_dict = parse.dict_to_table(table_dict)
            if not table_dict:
                return table_dict

            table_dict_ids = range(len(first(table_dict.values())))
            other_ids = range(len(first(other.values())))
            result_pair_ids = list(itertools.product(table_dict_ids, other_ids))

            this_result_ids = [pair[0] for pair in result_pair_ids]
            other_result_ids = [pair[1] for pair in result_pair_ids]

            this_result_dict = {col_name:col[this_result_ids] for col_name, col in table_dict.items()}
            other_result_dict = {col_name:col[other_result_ids] for col_name, col in other.items()}

            return this_result_dict | other_result_dict
        return table_func

    @classmethod
    def join_id(
        cls,
        other: Table,
        join_left_outer: bool = False,
        join_right_outer: bool = False
        ) -> TableFunc:
        def table_func(table_dict: Table) -> Table:
            table_dict = parse.dict_to_table(table_dict)
            if not table_dict:
                return table_dict

            len_table_dict = len(first(table_dict.values()))
            len_other = len(first(other.values()))
            if len_table_dict == len_other:
                return table_dict | other
            elif join_left_outer and len_table_dict > len_other:
                dif = len_table_dict - len_other
                none_list = np.full(dif, None)
                return table_dict | {key:np.append(col, none_list) for key, col in other.items()}
            elif join_right_outer and len_table_dict < len_other:
                dif = len_other - len_table_dict
                none_list = np.full(dif, None)
                return {key:np.append(col, none_list) for key, col in table_dict.items()} | other
            elif len_table_dict > len_other:
                return {key: col[:len_other] for key, col in table_dict.items()} | other
            else:
                return table_dict | {key: col[:len_table_dict] for key, col in other.items()}
        return table_func

    @classmethod
    def concat(
        cls,
        other: Table,
        join_left_outer: bool = False,
        join_right_outer: bool = False
        ) -> TableFunc:
        def table_func(table_dict: Table) -> Table:
            table_dict = parse.dict_to_table(table_dict)
            if join_left_outer and join_right_outer:
                if not table_dict:
                    return other
                if not other:
                    return table_dict
                len_table_dict = len(first(table_dict.values()))
                len_other = len(first(other.values()))
                return {key: np.append(
                    table_dict[key] if key in table_dict else np.full(len_table_dict, None),
                    other[key] if key in other else np.full(len_other, None)
                    ) for key in set(table_dict.keys()) | set(other.keys())}
            elif join_left_outer:
                if not table_dict:
                    return table_dict
                len_table_dict = len(first(table_dict.values()))
                len_other = len(first(other.values()))
                return {key: np.append(
                    col,
                    other[key] if key in other else np.full(len_other, None)
                    ) for key, col in table_dict.items()}
            elif join_right_outer:
                if not other:
                    return other
                len_table_dict = len(first(table_dict.values()))
                len_other = len(first(other.values()))
                return {key: np.append(
                    table_dict[key] if key in table_dict else np.full(len_table_dict, None),
                    col) for key, col in other.items()}
            else:
                if not table_dict:
                    return table_dict
                if not other:
                    return other
                len_table_dict = len(first(table_dict.values()))
                len_other = len(first(other.values()))
                return {key: np.append(table_dict[key], other[key]) for key in set(table_dict.keys()) & set(other.keys())}
        return table_func

    @classmethod
    def concat_multiple(
        cls,
        others: Iterable[Table],
        join_left_outer: bool = False,
        join_right_outer: bool = False,
        ) -> TableFunc:
        concat_func_list = [table.concat(other, join_left_outer, join_right_outer) for other in others]
        def table_func(table_dict: Table) -> Table:
            table_dict = parse.dict_to_table(table_dict)
            result_dict = table_dict
            for f in concat_func_list:
                result_dict = f(result_dict)
            return result_dict
        return table_func

    @classmethod
    def orderby(
        cls,
        selection: Selection,
        func: ColFunc,
        reverse: bool
        ) -> TableFunc:
        def table_func(table_dict: Table) -> Table:
            table_dict = parse.dict_to_table(table_dict)
            if not table_dict:
                return table_dict

            keys_array_list = cls.__get_keys_array_list(table_dict, selection, func)

            ids = None
            if len(keys_array_list) == 1:
                ids = np.argsort(keys_array_list[0])
            else:
                ids = np.lexsort(keys_array_list[::-1])

            if reverse:
                return {key: col[ids][::-1] for key, col in table_dict.items()}
            return {key: col[ids] for key, col in table_dict.items()}
        return table_func
    
    @classmethod
    def where(
        cls,
        selection: Selection = None,
        func: ColFunc = None
        ) -> TableFunc:
        def table_func(table_dict: Table) -> Table:
            table_dict = parse.dict_to_table(table_dict)
            if not table_dict:
                return table_dict
            keys_array_list = cls.__get_keys_array_list(table_dict, selection, func)
            
            filter_array = keys_array_list[0]
            if len(keys_array_list) > 1:
                filter_array = np.logical_and.reduce(keys_array_list)
            filter_array = np.where(filter_array)
            return {key: col[filter_array] for key, col in table_dict.items()}
        return table_func

    @classmethod
    def groupby(
        cls,
        selection: Selection,
        func: ColFunc
        ) -> Table:
        def table_func(table_dict: Table) -> Table:
            table_dict = parse.dict_to_table(table_dict)
            if not table_dict:
                return table_dict

            keys_array_list = cls.__get_keys_array_list(table_dict, selection, func)
            key_iter = keys_array_list[0]
            if len(keys_array_list) > 1:
                #key_array = parse.list_to_array(list(iter_array_list(keys_array_list)))
                key_iter = iter_array_list(keys_array_list)

            group_ids_dict = group_by_dict(key_iter)

            group_array = np.empty(len(group_ids_dict), dtype=object)
            #for i, group in enumerate(group_ids_dict.values()):
            #    group_array[i] = Qfrom({key: col[group] for key, col in table_dict.items()})
            group_array[:] = [Qfrom({key: col[group] for key, col in table_dict.items()}) for group in group_ids_dict.values()]

            result_dict = {
                #'key': parse.list_to_array(list(group_ids_dict.keys())),
                'key': parse.iter_to_array(group_ids_dict.keys()),
                'group': group_array}

            return result_dict
        return table_func

    @classmethod
    def flatten(cls, selection: Selection, out: str = None) -> TableFunc:
        def table_func(table_dict: Table) -> Table:
            table_dict = parse.dict_to_table(table_dict)
            if not table_dict:
                return table_dict

            selected_keys = parse.selection(selection, keys=table_dict.keys())

            out_key = out
            if not out_key:
                out_key = selected_keys[0]

            key_array = table_dict[selected_keys[0]] if len(selected_keys) == 1 else list(iter_array_list(cls.__get_keys_array_list(table_dict, selected_keys)))
            item_ids = np.array([i for i, col in enumerate(key_array) for _ in col])
            result_array = parse.list_to_array([item for row in key_array for item in row])
            scaled_table = {k: col[item_ids] for k, col in table_dict.items() if k!=out_key}
            new_row = {out_key: result_array}
            return scaled_table | new_row
        return table_func

    @classmethod
    def unique(cls, selection: Selection) -> TableFunc:
        def table_func(table_dict: Table) -> Table:
            table_dict = parse.dict_to_table(table_dict)
            if not table_dict:
                return table_dict

            selected_keys = parse.selection(selection, keys=table_dict.keys())
            key_dict = {key: table_dict[key] for key in selected_keys}
            keys_array_list = list(key_dict.values())
            key_array = array_tuple_to_tuple_array(keys_array_list)

            _, unique_ids = np.unique(key_array, return_index=True)
            return {key: col[unique_ids] for key, col in table_dict.items()}
        return table_func
    
    @classmethod
    def value_counts(cls, selection: Selection) -> TableFunc:
        def table_func(table_dict: Table) -> Table:
            table_dict = parse.dict_to_table(table_dict)
            if not table_dict:
                return table_dict

            selected_keys = parse.selection(selection, keys=table_dict.keys())
            key_dict = {key: table_dict[key] for key in selected_keys}
            keys_array_list = list(key_dict.values())
            key_iter = iter_array_list(keys_array_list)
            return key_array_to_value_counts(key_iter)
        return table_func

    @classmethod
    def remove(cls, selection: Selection) -> TableFunc:
        def table_func(table_dict: Table) -> Table:
            table_dict = parse.dict_to_table(table_dict)
            if not table_dict:
                return table_dict

            selected_keys = parse.selection(selection, keys=table_dict.keys())
            return {key: col for key, col in table_dict.items() if key not in selected_keys}
        return table_func

    @classmethod
    def rename(cls, map: dict[str, str]) -> TableFunc:
        def table_func(table_dict: Table) -> Table:
            table_dict = parse.dict_to_table(table_dict)
            if not table_dict:
                return table_dict
            return {map[key] if key in map else key: col for key, col in table_dict.items()}
        return table_func
        






class Qfrom():
    # - import_list
    # - import_dict
    # - (import_set)
    # - (import_array)
    # - (import_mtx)
    # - import_dataframe
    # - import_csv
    # - (import_json)
    # - import_generator

    # - eq
    # - str
    # - repr
    # - append
    # - setitem -> more dim slice support
    # - getitem
    # - contains
    # - iter
    # - len

    # - keys
    # - values
    # - items
    # - (stats)

    # - remove(selection: Selection)
    # - rename(map: dict[str, str])
    # - select(selection: Selection)
    # - map(args: Selection, func: ColFunc, out=Selection)
    # - orderby(selection: Selection, func: ColFunc, reverse: bool)

    # - where(selection: Selection, predicate: ColFunc)
    # - groupby(selection: Selection, func: ColFunc)
    # - flatten
    # - unique
    # - value_counts

    # - agg

    # - join
    # - join_cross
    # - join_outer
    # - join_outer_left
    # - join_outer_right
    # - join_id
    # - join_id_outer
    # - join_id_outer_left
    # - join_id_outer_right

    # - concat
    # - concat_outer
    # - concat_outer_left
    # - concat_outer_right

    # - calc
    # - call



    
    # - import_list
    # - import_dict
    # - (import_set)
    # - (import_array)
    # - (import_mtx)
    # - import_dataframe
    # - import_csv
    # - (import_json)
    # - import_generator
    def __init__(self, collection=None, operation_list=None, table_dict=None) -> Qfrom:
        operation_list = operation_list or []
        self.__operation_list = operation_list

        if table_dict:
            self.table_dict = table_dict
        else:
            self.table_dict = {}
            if isinstance(collection, str):
                #self.table_dict = parse.iterables_to_arrays(parse.str_to_collection(collection))
                self.table_dict = parse.str_to_collection(collection)
            elif isinstance(collection, dict):
                self.table_dict = parse.iterables_to_arrays(collection)
            elif isinstance(collection, Qfrom):
                collection.calculate()
                self.table_dict = {key:np.copy(value) for key, value in collection.table_dict.items()}
            elif isinstance(collection, np.ndarray) and len(collection.shape) == 1:
                self.table_dict = {'y': np.copy(collection)}
            elif isinstance(collection, np.ndarray) and len(collection.shape) > 1:
                collection_rot = np.rot90(collection)
                self.table_dict = {f'y{i}':col for i, col in enumerate(collection_rot[::-1])}
            elif isinstance(collection, pd.DataFrame):
                self.table_dict = {key: collection[key].values for key in collection.columns}
            elif isinstance(collection, Iterable):
                collection_list = list(collection)
                if len(collection_list) > 0:
                    first_item = first(collection_list)
                    if isinstance(first_item, dict):
                        self.table_dict = {key: parse.list_to_array([item[key] for item in collection_list]) for key in first_item.keys()}
                    elif (isinstance(first_item, tuple) or isinstance(first_item, list)) and len(first_item) == 1:
                        self.table_dict = {f'y': parse.list_to_array([item[0] for item in collection_list])}
                    elif (isinstance(first_item, tuple) or isinstance(first_item, list)) and len(first_item) > 1:
                        self.table_dict = {f'y{i}': parse.list_to_array([item[i] for item in collection_list]) for i in range(len(first_item))}
                    else:
                        self.table_dict = {'y': parse.list_to_array(collection_list)}
        
        #test if all cols have the same length
        #first_len = len(first(self.table_dict.values()))
        

    # - eq
    def __eq__(self, other: Qfrom) -> bool:
        self.calculate()
        if isinstance(other, Qfrom):
            return all(key in other.table_dict for key in self.table_dict)\
                and all(key in self.table_dict for key in other.table_dict)\
                and all(np.array_equal(col, other.table_dict[key]) for key, col in self.table_dict.items())
        return False
    # - str
    def __str__(self) -> str:
        self.calculate()
        
        if len(self.keys()) == 1:
            return 'Qfrom\n'+\
                '\t'.join(str(key) for key in self.table_dict.keys())+'\n'+\
                '\n'.join(str(row) for row in self)
        elif len(self.keys()) > 1:
            return 'Qfrom\n'+\
                '\t'.join(str(key) for key in self.table_dict.keys())+'\n'+\
                '\n'.join('\t'.join([str(v) for v in row]) for row in self)
        return 'Qfrom\nempty'
        
        #return 'Qfrom({'+ ', '.join(f'{key}:{col}' for key, col in self.table_dict.items()) + '})'
        #return f'Qfrom({str(self.table_dict)})'
    # - repr
    def __repr__(self) -> str:
        #return 'Qfrom(' + str(self.table_dict) + ')'
        return str(self)
    # - append
    def append(self, item: Any|tuple|dict) -> None:
        operation = (table.append, {'item': item})
        self.__operation_list.append(operation)
    # - setitem -> more dim slice support
    def __setitem__(self, key, newvalue):
        if any(self.__operation_list):
            self.calculate()

        if type(key) is int:
            if type(newvalue) is tuple:
                for i, k in enumerate(self.table_dict.keys()):
                    self.table_dict[k] = arr_set_value(self.table_dict[k], key, newvalue[i])
                return
            if type(newvalue) is dict:
                for k, v in newvalue.items():
                    self.table_dict[k] = arr_set_value(self.table_dict[k], key, v)
                return

            raise ValueError('if key is of type int, newvalue must be of type tuple or dict')

        columns = key
        if type(columns) is str:
            columns = tuple(col.strip() for col in columns.split(','))
        if type(columns) in [tuple, list]:
            if len(columns) == 2 and type(columns[0]) is str and type(columns[1]) is int:
                self.table_dict[columns[0]] = arr_set_value(self.table_dict[columns[0]], columns[1], newvalue)
                return
            newvaluedict = newvalue
            if type(newvaluedict) is list and len(columns) == 1:
                newvaluedict = {columns[0]: [row[0] for row in newvaluedict] if type(first(newvaluedict)) is tuple and len(first(newvaluedict)) == 1 else newvaluedict}
            elif type(newvaluedict) is list:
                newvaluedict = {col: [row[i] for row in newvaluedict] for i, col in enumerate(columns)}
            if type(newvaluedict) is dict:
                for col in columns:
                    self.table_dict[col] = np.array(newvaluedict[col])
                return
            if type(newvalue) is Qfrom:
                if all(col in columns for col in newvalue.keys()):
                    for col in columns:
                        self.table_dict[col] = np.array(newvalue[col])
                    return
                else:
                    newvalue_cols = newvalue.keys()
                    for i, col in enumerate(columns):
                        self.table_dict[col] = np.array(newvalue[newvalue_cols[i]])
                    return

        raise ValueError('key must be of type int, str, tuple or list')
    # - getitem
    def __getitem__(self, *args) -> Any|Qfrom:
        if any(self.__operation_list):
            self.calculate()
        
        if len(args) == 1:
            #key = args[0]
            if isinstance(args[0], int) or np.issubdtype(type(args[0]), np.integer):
                return tuple(col[args[0]] for col in self.table_dict.values()) if len(list(self.table_dict.values())) != 1 else first(self.table_dict.values())[args[0]]
            
            if isinstance(args[0], slice):
                result = {col_name: col[args[0]] for col_name, col in self.table_dict.items()}
                return Qfrom(result)
        
            if isinstance(args[0], tuple) and len(args[0]) == 2 and type(args[0][0]) is str and type(args[0][1]) is int:
                return self.table_dict[args[0][0]][args[0][1]]
            
        
        #if len(args[0]) > 0:
        #    return self.select(*args[0])
        return self.select(*args)
    # - contains
    def __contains__(self, item: Any|tuple|dict) -> bool:
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
    # - iter
    def __iter__(self) -> Iterable:
        self.calculate()
        return iter_array_list(list(self.table_dict.values()))
    # - len
    def __len__(self) -> int:
        if any(self.__operation_list):
            self.calculate()
        if len(self.table_dict) == 0:
            return 0
        return len(list(self.values())[0])

    # - keys
    def keys(self) -> Iterable[str]:
        self.calculate()
        return self.table_dict.keys()
    # - values
    def values(self) -> Iterable[np.ndarray]:
        self.calculate()
        return self.table_dict.values()
    # - items
    def items(self) -> Iterable[str,np.ndarray]:
        self.calculate()
        return self.table_dict.items()
    # - (stats)

    # - remove(selection: Selection)
    def remove(self, selection: Selection):
        operation = (table.remove, {'selection': selection})
        return Qfrom(
            operation_list=self.__operation_list+[operation],
            table_dict=self.table_dict,
            )
    # - rename(map: dict[str, str])
    def rename(self, map: dict[str, str]) -> Qfrom:
        operation = (table.rename, {'map': map})
        return Qfrom(
            operation_list=self.__operation_list+[operation],
            table_dict=self.table_dict,
            )
    # - select(selection: Selection)
    def select(self, selection: Selection) -> Qfrom:
        operation = (table.select, {'selection': selection})
        return Qfrom(
            operation_list=self.__operation_list+[operation],
            table_dict=self.table_dict,
            )
    # - map(args: Selection, func: ColFunc, out=Selection)
    def map(
        self,
        args: Selection=None,
        func: ColFunc=None,
        out: Selection=None
        ) -> Qfrom:
        operation = (table.map_join, {
            'args': args,
            'func': func,
            'out': out,
        })
        return Qfrom(
            operation_list=self.__operation_list+[operation],
            table_dict=self.table_dict,
            )
    # - orderby(selection: Selection, func: ColFunc, reverse: bool)
    def orderby(
        self,
        selection: Selection = None,
        func: ColFunc = None,
        reverse: bool = False
        ) -> Qfrom:
        operation = (table.orderby, {
            'selection': selection,
            'func': func,
            'reverse': reverse,
        })
        return Qfrom(
            operation_list=self.__operation_list+[operation],
            table_dict=self.table_dict,
            )

    # - where(selection: Selection, predicate: ColFunc)
    def where(
        self,
        selection: Selection = None,
        func: ColFunc = None
        ) -> Qfrom:
        operation = (table.where, {
            'selection': selection,
            'func': func,
        })
        return Qfrom(
            operation_list=self.__operation_list+[operation],
            table_dict=self.table_dict,
            )
    # - groupby(selection: Selection, func: ColFunc)
    def groupby(
        self,
        selection: Selection = None,
        func: ColFunc = None
        )-> Qfrom:
        operation = (table.groupby, {
            'selection': selection,
            'func': func,
        })
        return Qfrom(
            operation_list=self.__operation_list+[operation],
            table_dict=self.table_dict,
            )
    # - flatten
    def flatten(self, selection: Selection, out: str = None) -> Qfrom:
        operation = (table.flatten, {
            'selection': selection,
            'out': out,
        })
        return Qfrom(
            operation_list=self.__operation_list+[operation],
            table_dict=self.table_dict,
            )
    # - unique
    def unique(self, selection: Selection) -> Qfrom:
        operation = (table.unique, {'selection': selection})
        return Qfrom(
            operation_list=self.__operation_list+[operation],
            table_dict=self.table_dict,
            )
    # - value_counts
    def value_counts(self, selection: Selection) -> Qfrom:
        operation = (table.value_counts, {'selection': selection})
        return Qfrom(
            operation_list=self.__operation_list+[operation],
            table_dict=self.table_dict,
            )

    # - agg
    def agg(self, func_tuple: AggFunc|tuple[AggFunc]) -> Any|tuple[Any]:
        if any(self.__operation_list):
            self.calculate()

        cols = list(self.values())
        if callable(func_tuple) and len(cols) > 1:
            result = tuple(func_tuple(col) for col in cols)
            return result
        if callable(func_tuple):
            result = func_tuple(cols[0])
            return result

        agg_result = []
        not_used_columns = list(self.table_dict.values())
        for func in func_tuple:
            func_keys = get_keys_from_func(func, False)
            if len(func_keys) > len(not_used_columns):
                raise ValueError('To many funcs for columns')
            if '*' in func_keys:
                agg_result.append(func(*not_used_columns))
                not_used_columns = []
            else:
                agg_result.append(func(*not_used_columns[:len(func_keys)]))
                not_used_columns = not_used_columns[len(func_keys):]

        if len(agg_result) == 1:
            return agg_result[0]
        return tuple(agg_result)


    # - join
    def join(
        self,
        other: Qfrom,
        key_dict: dict[str, str] = None,
        join_left_outer: bool = False,
        join_right_outer: bool = False
        ) -> Qfrom:
        operation = (table.join, {
            'other': other,
            'key_dict': key_dict,
            'join_left_outer': join_left_outer,
            'join_right_outer': join_right_outer,
        })
        return Qfrom(
            operation_list=self.__operation_list+[operation],
            table_dict=self.table_dict,
            )
    # - join_cross
    def join_cross(self, other: Qfrom) -> Qfrom:
        operation = (table.join_cross, {'other': other.table_dict})
        return Qfrom(
            operation_list=self.__operation_list+[operation],
            table_dict=self.table_dict,
            )
    # - join_outer
    def join_outer(self, other: Qfrom, key_dict: dict[str, str] = None) -> Qfrom:
        return self.join(other, key_dict, join_left_outer=True, join_right_outer=True)
    # - join_outer_left
    def join_outer_left(self, other: Qfrom, key_dict: dict[str, str] = None) -> Qfrom:
        return self.join(other, key_dict, join_left_outer=True, join_right_outer=False)
    # - join_outer_right
    def join_outer_right(self, other: Qfrom, key_dict: dict[str, str] = None) -> Qfrom:
        return self.join(other, key_dict, join_left_outer=False, join_right_outer=True)
    # - join_id
    def join_id(
        self,
        other: Qfrom,
        join_left_outer: bool = False,
        join_right_outer: bool = False
        ) -> Qfrom:
        operation = (table.join_id, {
            'other': other.table_dict,
            'join_left_outer': join_left_outer,
            'join_right_outer': join_right_outer,
        })
        return Qfrom(
            operation_list=self.__operation_list+[operation],
            table_dict=self.table_dict,
            )
    # - join_id_outer
    def join_id_outer(self, other: Qfrom) -> Qfrom:
        return self.join_id(other, join_left_outer=True, join_right_outer=True)
    # - join_id_outer_left
    def join_id_outer_left(self, other: Qfrom) -> Qfrom:
        return self.join_id(other, join_left_outer=True, join_right_outer=False)
    # - join_id_outer_right
    def join_id_outer_right(self, other: Qfrom) -> Qfrom:
        return self.join_id(other, join_left_outer=False, join_right_outer=True)

    # - concat
    def concat(
        self,
        other: Qfrom|list[Qfrom],
        join_left_outer=False,
        join_right_outer=False
        ) -> Qfrom:
        operation = (table.concat_multiple, {
            'others': [q.table_dict for q in other] if type(other) is list else [other.table_dict],
            'join_left_outer': join_left_outer,
            'join_right_outer': join_right_outer,
        })
        return Qfrom(
            operation_list=self.__operation_list+[operation],
            table_dict=self.table_dict,
            )
    # - concat_outer
    def concat_outer(self, other: Qfrom|list[Qfrom]) -> Qfrom:
        return self.concat(other, join_left_outer=True, join_right_outer=True)
    # - concat_outer_left
    def concat_outer_left(self, other: Qfrom|list[Qfrom]) -> Qfrom:
        return self.concat(other, join_left_outer=True, join_right_outer=False)
    # - concat_outer_right
    def concat_outer_right(self, other: Qfrom|list[Qfrom]) -> Qfrom:
        return self.concat(other, join_left_outer=False, join_right_outer=True)

    # - calc
    def calculate(self) -> dict[str, np.ndarray]:
        if any(self.__operation_list):
            for func, kwrgs in self.__operation_list:
                self.table_dict = func(**kwrgs)(self.table_dict)
            
            self.__operation_list = []
        return self.table_dict
    # - call
    def __call__(self, func: Callable[[Qfrom,*Any], Any], *args: Any, **kwds: Any) -> Any:
        return func(self, *args, **kwds)







def reduce(iterable, f: callable):
    if type(iterable) not in [tuple, list]:
        iterable = list(iterable)
    if len(iterable) == 0:
        raise ValueError('iterable must contain at least one element')
    result = iterable[0]
    for item in iterable:
        result = f(result, item)
    return result


class col():
    # = 0 -> 1
    #   (- <Any> -> singleton to array)
    #   - id
    #
    # = 1 -> 1
    #   - pass_none
    #   - normalize
    #   - abs
    #   - center -> set a new origin for a column: [1, 2, 3], origin=2 -> [-1, 0, 1]
    #   - shift(steps=...)
    #   - not
    #
    # = n -> 1
    #   - any
    #   - all
    #   - min
    #   - min_colname
    #   - max
    #   - max_colname
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



    # = 0 -> 1
    #   (- <Any> -> singleton to array)
    #   - id
    @classmethod
    def id(cls):
        i = 0
        while True:
            yield i
            i += 1
    #
    # = 1 -> 1
    #   - pass_none
    #   - normalize
    @classmethod
    def normalize(cls, array: np.ndarray):
        maximum = np.max(np.abs(array))
        return array/maximum
    #   - abs
    @classmethod
    def abs(cls, array: np.ndarray):
        return np.abs(array)
    #   - center -> set a new origin for a column: [1, 2, 3], origin=2 -> [-1, 0, 1]
    #   - shift(steps=...)
    @classmethod
    def shift(cls, steps: int, default_value = 0):
        def shift_func(array: np.ndarray):
            result_array = np.full(len(array), default_value)
            if steps > 0:
                result_array[steps:] = array[:-steps]
            else:
                result_array[:steps] = array[-steps:]
            return result_array
        return shift_func
    #   - not
    @classmethod
    def log_not(cls, array: np.ndarray):
        return np.logical_not(array)
    #
    # = n -> 1
    #   - any
    @classmethod
    def any(cls, *arrays):
        array_tuple = np.array(arrays)
        return np.any(array_tuple, axis=0)
    #   - all
    @classmethod
    def all(cls, *arrays):
        array_tuple = np.array(arrays)
        return np.all(array_tuple, axis=0)
    #   - min
    @classmethod
    def min(cls, *arrays):
        #return reduce(arrays, np.minimum)
        array_tuple = np.array(arrays)
        return np.min(array_tuple, axis=0)
    #   - min_colname
    @classmethod
    def min_colname(cls, **kwrgs):
        array_tuple = np.array(list(kwrgs.values()))
        #array_tuple = parse.iter_to_array(kwrgs.values())
        ids = np.argmin(array_tuple, axis=0)
        #key_array = np.array(list(kwrgs.keys()))
        key_array = parse.iter_to_array(kwrgs.keys())
        return key_array[ids]
    #   - max
    @classmethod
    def max(cls, *arrays):
        #return reduce(arrays, np.maximum)
        array_tuple = np.array(arrays)
        return np.max(array_tuple, axis=0)
    #   - max_colname
    @classmethod
    def max_colname(cls, **kwrgs):
        array_tuple = np.array(list(kwrgs.values()))
        #array_tuple = parse.iter_to_array(kwrgs.values())
        ids = np.argmax(array_tuple, axis=0)
        #key_array = np.array(list(kwrgs.keys()))
        key_array = parse.iter_to_array(kwrgs.keys())
        return key_array[ids]
    #   - sum
    @classmethod
    def sum(cls, *arrays):
        #return reduce(arrays, np.add)
        array_tuple = np.array(arrays)
        return np.sum(array_tuple, axis=0)
    #   - mean
    @classmethod
    def mean(cls, *arrays):
        #s = cls.sum(arrays)
        #return s/len(arrays)
        array_tuple = np.array(arrays)
        return np.mean(array_tuple, axis=0)
    #   - median
    @classmethod
    def median(cls, *arrays):
        array_tuple = np.array(arrays)
        return np.median(array_tuple, axis=0)
    #   - var
    @classmethod
    def var(cls, *arrays):
        array_tuple = np.array(arrays)
        return np.var(array_tuple, axis=0)
    #   - eq
    @classmethod
    def eq(cls, *arrays):
        return reduce(arrays, np.equal)
    #   - agg(colcount) -> combines multible cols to one 2d col
    @classmethod
    def agg(cls, *arrays):
        l = list(zip(*arrays))
        a = np.empty(len(l), dtype=object)
        a[:] = l
        return a
    #   - state(rules: func|dict[func]) -> iterates over col. for each item the result state of the last item is feed in. 
    @classmethod
    def state(cls, func:callable, start_state=None):
        def get_state_col(*arrays):
            current_state = start_state
            result_l = []
            for t in zip(*arrays):
                current_state = func(current_state, t)
                result_l.append(current_state)
            return result_l
        return get_state_col
    #   - lod_and
    @classmethod
    def log_and(cls, *arrays):
        return reduce(arrays, np.logical_and)
    #   - lod_or
    @classmethod
    def log_or(cls, *arrays):
        return reduce(arrays, np.logical_or)
    #   - lod_xor
    @classmethod
    def log_xor(cls, *arrays):
        return reduce(arrays, np.logical_xor)
    #
    # = 1 -> n
    #   - copy(n)
    #   - flatten -> autodetect out count
    @classmethod
    def flatten(cls, array: np.ndarray) -> np.ndarray|tuple[np.ndarray]|Table:
        if len(array) == 0:
            return array
        out_count = len(array[0])
        if type(array[0]) is dict:
            flatten_func = np.frompyfunc(lambda iterable: tuple(iterable.values()), 1, out_count)
            result = flatten_func(array)
            result = tuple(optimize_array_dtype(a) if a.dtype == np.dtype(object) else a for a in result)
            return dict(zip(array[0].keys(), result))
        flatten_func = np.frompyfunc(lambda iterable: tuple(iterable), 1, out_count)
        result = flatten_func(array)
        result = tuple(optimize_array_dtype(a) if a.dtype == np.dtype(object) else a for a in result)
        return result
    #
    # = n -> m
    #   - ml_models

class func():
    # - __call__(func, in: int, out: int) -> verpackt func in lambda, so dass lambda-parameter in-count entsprechen und output tuple out-count entspricht.
    # - vec(func) -> vectorize func, autodetect in and out counts
    # - vec(func, in: int, out: int)
    # - multicol(repetitioncount: int)
    # (- args(func))
    # (   -> ex. args(lambda a,b: a+b) -> {('a', 'b'): lambda a,b: a+b})
    # (   -> ex. args(lambda a,b, *args: (a+b, *args)) -> {('a', 'b', '*'): lambda a,b, *args: (a+b, *args)})



    # - __call__(func, in: int, out: int) -> verpackt func in lambda, so dass lambda-parameter in-count entsprechen und output tuple out-count entspricht.
    # - vec(func) -> vectorize func, autodetect in and out counts
    # - vec(func, in: int, out: int)
    @classmethod
    def vec(cls, func: callable, in_count: int=None, out: int=None, correct_dtype=False):
        paras = []
        if in_count is None:
            sig = inspect.signature(func)
            names = list(sig.parameters.keys())
            paras = [str(v) for v in sig.parameters.values()]
            
            in_count = len(paras)
        else:
            names = [f'x{i}' for i in range(in_count)]
            paras = names
        if out is None:
            out_count = 1
        else:
            out_count = out

        func_vec = None
        if correct_dtype:
            func_vec = np.vectorize(func)
        else:
            func_vec = np.frompyfunc(func, in_count, out_count)

        def np_func_wrapper(*args, **kwrgs):
            func_result = func_vec(*args, **kwrgs)
            if out is None and len(func_result) > 1 and type(func_result[0]) in [tuple, dict]:
                return col.flatten(func_result)
            return func_result

        args_str = ', '.join(paras)
        arg_names_str = ', '.join(names)
        func_str = f'lambda {args_str}: np_func_wrapper({arg_names_str})'

        return eval(func_str, {'np_func_wrapper': np_func_wrapper})
    # - multicol(repetitioncount: int)
    # (- args(func))
    # (   -> ex. args(lambda a,b: a+b) -> {('a', 'b'): lambda a,b: a+b})
    # (   -> ex. args(lambda a,b, *args: (a+b, *args)) -> {('a', 'b', '*'): lambda a,b, *args: (a+b, *args)})

class agg():
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



    # - any
    @classmethod
    def any(cls, array: np.ndarray):
        return np.any(array)
    # - all
    @classmethod
    def all(cls, array: np.ndarray):
        return np.all(array)
    # - min
    @classmethod
    def min(cls, array: np.ndarray):
        return np.min(array)
    # - min_id
    @classmethod
    def min_id(cls, array: np.ndarray):
        return np.argmin(array)
    # - max
    @classmethod
    def max(cls, array: np.ndarray):
        return np.max(array)
    # - max_id
    @classmethod
    def max_id(cls, array: np.ndarray):
        return np.argmax(array)
    # - sum
    @classmethod
    def sum(cls, array: np.ndarray):
        return np.sum(array)
    # - mean
    @classmethod
    def mean(cls, array: np.ndarray):
        return np.mean(array)
    # - median
    @classmethod
    def median(cls, array: np.ndarray):
        return np.median(array)
    # - var
    @classmethod
    def var(cls, array: np.ndarray):
        return np.var(array)
    # - len
    @classmethod
    def len(cls, array: np.ndarray):
        return np.len(array)
    # - state(rules: func|dict[func]) -> returns the last state of col.state

class plot():
    # - plot
    # - bar
    # - hist
    # - box
    # - scatter
    
    
    
    # - plot
    @classmethod
    def plot(cls, q: Qfrom, x=None, show_legend=True, title=None, x_scale_log=False, y_scale_log=False, axis=None, figsize=None, order_by_x=True) -> None:
        q.calculate()

        ax = axis
        if axis is None:
            fig = plt.figure()
            if figsize:
                fig.set_figwidth(figsize[0])
                fig.set_figheight(figsize[1])
            ax = fig.add_subplot(1,1,1)

        if x is None:
            x = first(q.keys())
        
        q_data = q.orderby(x) if order_by_x else q
        q_data.calculate()
        col_list = [key for key in q_data.keys() if key != x]

        x_list = q_data[x](out.array)
        ax.set_xlabel(x)
        for c in col_list:
            c_list = q_data[c](out.array)
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
    # - bar
    # - hist
    # - box
    # - scatter
    pass

class out():
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



    # - tolist
    @classmethod
    def list(cls, q: Qfrom) -> list:
        q.calculate()

        if len(q.table_dict.keys()) == 1:
            return list(first(q.table_dict.values()))
        if len(q.table_dict.keys()) > 1:
            return [q[i] for i in range(len(first(q.table_dict.values())))]
        return []
    # - (toset)
    # - todict
    @classmethod
    def dict(cls, q: Qfrom) -> Table:
        q.calculate()
        return q.table_dict
    # - toarray
    @classmethod
    def array(cls, q: Qfrom) -> np.ndarray:
        q.calculate()

        if len(q.table_dict) == 0:
            return np.empty()
        if len(q.table_dict) == 1:
            return first(q.table_dict.values())

        #result = np.array(list(q.table_dict.values())[::-1])
        result = parse.iter_to_array(q.table_dict.values())[::-1]
        result = np.rot90(result, 3)
        return result
    # - (tomtx)
    # - todf
    # - tocsv
    @classmethod
    def csv(cls, q: Qfrom, delimiter=',', header=True) -> str:
        q.calculate()
        if len(q) == 0:
            return ''
        
        header_str = delimiter.join(q.keys())
        data = [[str(item(out.dict)) if type(item) is Qfrom else str(item) for item in row] for row in q]
        data = [delimiter.join([f'"{item}"' if delimiter in item else item for item in row]) for row in data]
        data_str = '\n'.join(data)

        return f'{header_str}\n{data_str}' if header else data_str
    # - (tocsvfile)
    @classmethod
    def tocsvfile(cls, q: Qfrom, path, encoding='UTF8', delimiter=',', header=True) -> None:
        with open(path, 'w', encoding=encoding, newline='') as f:
            f.write(cls.csv(q, delimiter=delimiter, header=header))
    # - (tojson)
    # - (tojsonfile)

class trans():
    # - (shuffle)
    pass