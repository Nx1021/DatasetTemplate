from typing import List, Tuple, Dict, Any, Union, Callable, Optional, TypedDict, Literal
from typing import TypeVar, Generic, Iterable, Generator, MappingView
from typing_extensions import deprecated
from enum import Enum
import os
import shutil
import copy
from tqdm import tqdm
from functools import partial, reduce
import numpy as np

from abc import ABC, abstractmethod

import warnings

from utils import JsonIO
from functools import reduce

DATA_TYPE = TypeVar("DATA_TYPE")
_T = TypeVar("_T")

def inherit_private(child:type, parent:type, name):
    name = f"_{parent.__name__}{name}"
    setattr(child, name, getattr(parent, name))

class SingleData(ABC, Generic[DATA_TYPE]):

    O_MODE_IMIDIATE = 1
    O_MODE_EXIT     = 2
    O_MODE_MANUAL   = 4

    def __init__(self, sub_dir:str,
                 read_func:Callable[[str],None],
                 write_func:Callable[[str, DATA_TYPE],int]) -> None:
        self.__dataset = None
        self.__sub_dir = sub_dir
        self._data_cache:dict[int, DATA_TYPE] = {}
        self.__save_mode = self.O_MODE_IMIDIATE
        self._read_func = read_func
        self._write_func = write_func

    @property
    def dataset(self) -> "Dataset":
        return self.__dataset

    def _set_dataset(self, dataset:"Dataset") -> None:
        if isinstance(dataset, Dataset):
            self.__dataset = dataset
            path = self.get_path(0)
            os.makedirs(os.path.dirname(path), exist_ok=True)
        elif dataset is None:
            self.__dataset = None
        else:
            raise TypeError("dataset should be an instance of Dataset or None")

    @property
    def root_dir(self) -> str:
        return self.dataset.directory

    @property
    def sub_dir(self) -> str:
        return self.__sub_dir

    @property
    def data_cache(self) -> dict[int, DATA_TYPE]:
        return MappingView(self._data_cache)
    
    @property
    def keys(self) -> list[int]:
        return self._data_cache.keys()

    @abstractmethod
    def _write(self, idx, data):
        pass

    @abstractmethod
    def _read(self, idx):
        pass

    @abstractmethod
    def _remove(self, idx):
        pass

    @abstractmethod
    def _move(self, old_idx, new_idx):
        pass

    @abstractmethod
    def _copy(self, old_idx, new_idx):
        pass

    @abstractmethod
    def _save(self):
        pass

    def set_save_mode(self, mode:int) -> None:
        assert mode in (self.O_MODE_IMIDIATE, self.O_MODE_EXIT, self.O_MODE_MANUAL), f"Invalid save_mode {mode}"
        self.__save_mode = mode

    def get_save_mode(self) -> int:
        return self.__save_mode
    
    @abstractmethod
    def get_path(self, idx:int) -> str:
        pass

def filename_generator_builder(format_str:str, rjust:tuple[int, str] = (4, "0")) -> Callable[[int], str]:
    return lambda idx: format_str.format(str(idx).rjust(*rjust))

class DataCluster(SingleData[DATA_TYPE], Generic[DATA_TYPE]):
    def __init__(self, sub_dir:str, 
                 read_func:Callable[[str], None],
                 write_func:Callable[[str, DATA_TYPE],int],
                 path_generator:Callable) -> None:
        super().__init__(sub_dir, read_func, write_func)
        self.path_generator = path_generator

    def _read(self, idx):
        path = self.get_path(idx)
        if not os.path.exists(path):
            warnings.warn(f"File {path} not exists")
            return None
        return self._read_func(self.get_path(idx))
    
    def _write(self, idx, value):
        if self.get_save_mode() == self.O_MODE_IMIDIATE:
            path = self.get_path(idx)
            self._write_func(path, value)
        elif self.get_save_mode() & (self.O_MODE_EXIT | self.O_MODE_MANUAL):
            assert value is not None, f"Data {idx} should not be None, if you want to remove it, please use remove method"
            self._data_cache[idx] = value

    def _remove(self, idx):
        if self.get_save_mode() == self.O_MODE_IMIDIATE:
            path = self.get_path(idx)
            if not os.path.exists(path):
                _IdxNotFoundError(f"File {path} not exists")
                return
            try:
                os.remove(path)
            except Exception as e:
                pass 
        elif self.get_save_mode() & (self.O_MODE_EXIT | self.O_MODE_MANUAL):
            if idx in self._data_cache:
                self._data_cache[idx] = None

    def _move(self, old_idx, new_idx):
        if self.get_save_mode() == self.O_MODE_IMIDIATE:
            old_path = self.get_path(old_idx)
            new_path = self.get_path(new_idx)
            if not os.path.exists(old_path):
                raise _IdxNotFoundError(f"File {old_path} not exists")
            try:
                os.rename(old_path, new_path)
            except Exception as e:
                pass
        elif self.get_save_mode() & (self.O_MODE_EXIT | self.O_MODE_MANUAL):
            if old_idx in self._data_cache:
                self._data_cache[new_idx] = self._data_cache[old_idx]
                self._data_cache[old_idx] = None

    def _copy(self, old_idx, new_idx):
        if self.get_save_mode() == self.O_MODE_IMIDIATE:
            old_path = self.get_path(old_idx)
            new_path = self.get_path(new_idx)
            if not os.path.exists(old_path):
                raise _IdxNotFoundError(f"File {old_path} not exists")
            try:
                shutil.copy(old_path, new_path)
            except Exception as e:
                pass
        elif self.get_save_mode() & (self.O_MODE_EXIT | self.O_MODE_MANUAL):
            if old_idx in self._data_cache:
                self._data_cache[new_idx] = self._data_cache[old_idx]

    def _save(self):
        for idx, data in self._data_cache.items():
            if data is not None:
                self._write(idx, data)
            else:
                self._remove(idx)
        self._data_cache.clear()

    def get_path(self, idx):
        return os.path.join(self.root_dir, self.sub_dir, self.path_generator(idx))
    
    def scan(self, idx_range:Iterable[int], verbose = False):
        not_exist = []
        for idx in idx_range:
            path = self.get_path(idx)
            if not os.path.exists(path):
                not_exist.append(idx)
                if verbose:
                    print(f"File {path} not exists")
        return not_exist

class DataFile(SingleData[DATA_TYPE], Generic[DATA_TYPE]):
    def __init__(self, sub_dir:str,
                 read_func:Callable[[str],None],
                 write_func:Callable[[str, DATA_TYPE],int]) -> None:
        super().__init__(sub_dir, read_func, write_func)
        self.set_save_mode(self.O_MODE_EXIT)
        self._data_cache = {}

    def _set_dataset(self, dataset: "Dataset") -> None:
        super()._set_dataset(dataset)
        if not os.path.exists(self.get_path()):
            warnings.warn(f"initialize {self.__class__.__name__}: File {self.get_path()} not exists")
            self._data_cache = {}
        else:
            data = self._read_func(self.get_path())
            self._data_cache = self.cvt_to_data_cache(data)

    def _read(self, idx):
        return self._data_cache[idx]

    def _write(self, idx, value):
        self._data_cache[idx] = value
        if self.get_save_mode() == self.O_MODE_IMIDIATE:
            self._save()    

    def _remove(self, idx):
        self._data_cache[idx] = None
        if self.get_save_mode() == self.O_MODE_IMIDIATE:
            self._save()
    
    def _move(self, old_idx, new_idx):
        self._data_cache[new_idx] = self._data_cache[old_idx]
        self._data_cache[old_idx] = None
        if self.get_save_mode() == self.O_MODE_IMIDIATE:
            self._save()

    def _copy(self, old_idx, new_idx):
        self._data_cache[new_idx] = self._data_cache[old_idx]
        if self.get_save_mode() == self.O_MODE_IMIDIATE:
            self._save()

    def cvt_to_data_cache(self, data):
        return data
    
    def cvt_from_data_cache(self):
        return self._data_cache

    def _save(self):
        self._data_cache = {k: self._data_cache[k] for k in sorted(self._data_cache.keys()) if self._data_cache[k] is not None}
        data = self.cvt_from_data_cache()
        self._write_func(self.get_path(), data)

    def set_save_mode(self, mode: bool) -> None:
        if mode == self.O_MODE_IMIDIATE:
            warnings.warn("set save_mode to IMIDIATE will cause frequent IO operation and may slow down the program")
        return super().set_save_mode(mode)
    
    def get_path(self, idx:int = None) -> str:
        return os.path.join(self.root_dir, self.sub_dir)

class _OverviewDict(TypedDict):
    tags: set[str] = []
    data_exists: dict[str, dict[str, bool]] = {}

def _overview_dict_builder(tags:Optional[list[str]] = None, data_exists:Optional[dict[str, dict[str, bool]]] = None) -> _OverviewDict:
    if tags is None:
        tags = set([])
    else:
        assert isinstance(tags, Iterable), "tags should be a Iterable"
        assert all([isinstance(tag, str) for tag in tags]), "tags should be a list of str"
        tags = set(tags)
    if data_exists is None:
        data_exists = {}
    else:
        assert isinstance(data_exists, dict), "data_exists should be a dict"
    return {"tags": tags, "data_exists": data_exists}

class _SingleDataMng():
    def __init__(self, data:SingleData) -> None:
        self.__data:SingleData = data
        self.__read_required   = True
        self.__write_required  = True

    @property
    def read_required(self) -> bool:
        return self.__read_required
    
    @property
    def write_required(self) -> bool:
        return self.__write_required
    
    @read_required.setter
    def read_required(self, value:bool) -> None:
        self.__read_required = bool(value)

    @write_required.setter
    def write_required(self, value:bool) -> None:
        self.__write_required = bool(value)

    @property
    def data(self) -> SingleData:
        return self.__data

class _RequirementError(ValueError):
    pass

class _IdxNotFoundError(ValueError):
    pass

class DatasetIOMode(Enum):
    MODE_READ = 1
    MODE_APPEND = 2
    MODE_WRITE = 4

class _RegisterInstance():
    """
    A base class for registering instances with unique identities.

    * Every instance of a subclass of _RegisterInstance will be registered in 
    `_RegisterInstance._registry` with its identity_string.

    Attributes
    -----
    _registry (dict[str, RGSITEM])
        A dictionary to store registered instances.
    """

    __registry:dict[str, "_DatasetBase"] = {}

    def _register(self, name:str, alterable = False) -> str:
        """
        brief
        -----
        register datasets

        parameters
        -----
        name:str
            name: name of datasets
        alterable: bool
            alterable: whether the name can be altered if the name already exists
        """
        if not alterable:
            if name in self.__registry:
                raise ValueError(f"Dataset with the same name({name}) already exists")
            else:
                self.__registry[name] = self
                return name
        else:
            if name in self.__registry:
                name = self.__alter_name(name)
            self.__registry[name] = self
            return name
            
    def __alter_name(self, old_name):
        for i in range(1, 9999):
            new_name = old_name + f"_{str(i).rjust(4, '0')}"
            if new_name in self.__registry:
                continue
            else:
                return new_name
        raise RuntimeError("too many datasets with the same name!")
    
    @classmethod
    def _query_registered(cls, name):
        if name in cls.__registry:
            return cls.__registry[name]
        else:
            warnings.warn(f"Dataset {name} not exists")
            return None

    def __hash__(self):
        return hash(self.__repr__())

def _resort_dict(d:dict[int, Any]) -> dict[int, Any]:
    assert isinstance(d, dict), "d should be a dict"
    indices = np.array(sorted(list(d.keys())), np.int32)
    target  = np.arange(len(indices), dtype=np.int32)
    new_d = {k:v for k, v in zip(target, map(lambda x: d[x], indices))} 
    return new_d

class _DatasetBase(ABC, _RegisterInstance):
    def __init__(self, dataset_name:Optional[str] = None, alterable = False) -> None:
        if dataset_name is None:
            dataset_name = self.__class__.__name__
            alterable = True
        self.__dataset_name = self._register(dataset_name, alterable=alterable)

    def __repr__(self):
        return self.__dataset_name

    @property
    def dataset_name(self):
        return self.__dataset_name

    @abstractmethod
    def _get_overview(self) -> dict[int, _OverviewDict]:
        pass

    @abstractmethod
    def make_continuous(self) -> None:
        pass

    @classmethod
    @abstractmethod
    def from_cfg(cls, path):
        pass

    @abstractmethod
    def save_cfg(self, cfg_file:Optional[str] = None) -> None:
        pass

    @abstractmethod
    def read(self, idx:int) -> dict[str, Any]:
        pass

    @abstractmethod
    def remove(self, idx:int) -> None:
        pass

    @abstractmethod
    def move(self, old_idx:int, new_idx:int) -> None:
        pass

    @abstractmethod
    def copy(self, old_idx:int, new_idx:int) -> None:
        pass

    def clear(self, selected_indices = None) -> None:
        selected_indices = selected_indices if selected_indices is not None else list(self.keys())
        for idx in selected_indices:
            self.remove(idx)

    @abstractmethod
    def is_complete(self, idx) -> bool:
        pass

    @abstractmethod
    def _is_empty(self, idx) -> bool:
        pass

    @property
    def size(self) -> int:
        return len(self._get_overview())
    
    def is_continuous(self) -> bool:
        overview = self._get_overview()
        indices = np.array(sorted(list(overview.keys())), np.int32)
        diff = np.diff(indices)
        return not np.any(diff > 1)

    # region: idx operations
    def has(self, idx) -> bool:
        return idx in self._get_overview()
    
    def _get_empty_idx(self, fill_uncontinuous = False) -> int:
        if not fill_uncontinuous:
            return int(max(self.keys(), default=-1) + 1)
        else:
            indices = np.array(list(self._get_overview().keys()))
            inc = np.where(np.diff(indices) > 1)[0]
            if len(inc) == 0:
                return int(max(indices) + 1)
            else:
                return int(indices[inc[0]] + 1)
    # endregion

    # region:dictlike operations
    def get(self, idx:int) -> dict[str, Any]:
        return self.read(idx)
    
    def __getitem__(self, idx:int) -> dict[str, Any]:
        return self.read(idx)
        
    def __contains__(self, idx:int) -> bool:
        return self.has(idx)
    
    def keys(self):
        return self._get_overview().keys()
    
    def values(self):
        for idx in self.keys():
            yield self.read(idx)
    
    def items(self):
        for idx in self.keys():
            yield idx, self.read(idx)
    
    def __iter__(self):
        return self.keys()
    
    def __len__(self):
        return self.size
    # endregion

    # region: tags
    def __check_tags(self, tags:Iterable[str]|str) -> Iterable[str]:
        if isinstance(tags, str):
            tags = [tags]
        assert isinstance(tags, Iterable), "tags should be a list"
        assert all([isinstance(tag, str) for tag in tags]), "tags should be a list of str"
        return tags

    def get_tags(self, idx:int) -> set[str]:
        if idx not in self._get_overview():
            warnings.warn(f"Data {idx} not exists, can not get tags")
            return set()
        else:
            return self._get_overview()[idx]["tags"]
    
    def set_tags(self, idx:int, tags:Iterable[str]|str|None) -> None:
        if idx not in self._get_overview():
            warnings.warn(f"Data {idx} not exists, can not add tags")
            return
        if tags is None:
            return
        tags = self.__check_tags(tags)
        self._get_overview()[idx]["tags"].clear()
        self._get_overview()[idx]["tags"].update(tags)

    def add_tags(self, idx:int, tags:Iterable[str]|str|None) -> None:
        if idx not in self._get_overview():
            warnings.warn(f"Data {idx} not exists, can not add tags")
            return
        if tags is None:
            return
        tags = self.__check_tags(tags)
        self._get_overview()[idx]["tags"].update(tags)

    def remove_tags(self, idx:int, tags:Iterable[str]|str) -> None:
        if idx not in self._get_overview():
            warnings.warn(f"Data {idx} not exists, can not remove tags")
            return
        tags = self.__check_tags(tags)
        for tag in tags:
            if tag in self._get_overview()[idx]["tags"]:
                self._get_overview()[idx]["tags"].remove(tag)
    
    def clear_tags(self, idx:int) -> None:
        if idx not in self._get_overview():
            warnings.warn(f"Data {idx} not exists, can not clear tags")
            return
        self._get_overview()[idx]["tags"].clear()
    # endregion

    # region: dataset mapping
    def _select(self, include:Optional[Iterable[_T]|_T], exclude:Optional[Iterable[_T]|_T], _type:type[_T]):
        def has_intersection(a, b):
            return len(set(a) & set(b)) > 0
        
        def filter_one(include, exclude, value):
            return (not include or has_intersection(value, include)) and\
                   (not exclude or not has_intersection(value, exclude))
        
        assert isinstance(_type, type), f"_type should be an instance of type"
        assert include is None or isinstance(include, (_type, Iterable)) or include is None, f"include should be None or an instance of {_type.__name__} or Iterable"
        assert exclude is None or isinstance(exclude, (_type, Iterable)) or exclude is None, f"exclude should be None or an instance of {_type.__name__} or Iterable"
        include = [include] if isinstance(include, _type) else include
        exclude = [exclude] if isinstance(exclude, _type) else exclude

        def func(value: Iterable[_T]|_T):
            if isinstance(value, Iterable) and not isinstance(value, _type):
                assert all([isinstance(v, _type) for v in value]), f"value should be an Iterable of {_type.__name__}"
            elif isinstance(value, _type):
                value = [value]
            else:
                raise ValueError(f"value should be an instance of {_type.__name__} or Iterable[{_type.__name__}]")
        
            return filter_one(include, exclude, value)

        return func

    def select( self, 
                include_tags:Optional[Iterable[str]|str] = None, 
                exclude_tags:Optional[Iterable[str]|str] = None,
                include_indices:Optional[Iterable[int]|int] = None,
                exclude_indices:Optional[Iterable[int]|int] = None) -> list[int]:
        
        tags_filter    = self._select(include_tags,    exclude_tags,    str)
        indices_filter = self._select(include_indices, exclude_indices, int)
        
        selected_indices = []
        for idx in self.keys():
            tags = self.get_tags(idx)
            if tags_filter(tags) and indices_filter(idx):
                selected_indices.append(idx)

        return selected_indices

        # def has_intersection(a, b):
        #     return len(set(a) & set(b)) > 0

        # include_tags    = [include_tags]    if isinstance(include_tags, str)    else include_tags
        # include_indices = [include_indices] if isinstance(include_indices, int) else include_indices
        # exclude_tags    = exclude_tags    if exclude_tags    is not None else []
        # exclude_indices = exclude_indices if exclude_indices is not None else []

        # selected_indices:list[int] = []
        # for idx in self.keys():
        #     tags = self.get_tags(idx)
        #     if  (not include_tags or has_intersection(tags, include_tags)) and\
        #         (tags not in exclude_tags) and\
        #         (not include_indices or has_intersection(idx, include_indices)) and\
        #         (idx not in exclude_indices):
        #         selected_indices.append(idx)   
        # return selected_indices

    def gen_subset(self, view_name:Optional[str] = None, 
                   selected_indices = None) -> "DatasetView":
        view = DatasetView(view_name)
        view.add_dataset(self, selected_indices)

        return view
    # endregion

class DatasetCfg(TypedDict):
    dataset_name:str
    classname:str
    directory:str
    overview:dict[int, _OverviewDict]

class _Dataset(_DatasetBase):
    def __init__(self, directory:str, 
                 dataset_name:Optional[str] = None,
                 ) -> None:
        super().__init__(dataset_name)
        self.__directory = directory
        os.makedirs(directory, exist_ok=True)
        _cfg_path = self.get_cfg_path()
        if os.path.exists(_cfg_path):
            cfg:DatasetCfg = JsonIO.load_json(_cfg_path)
            self.__overview:dict[int, _OverviewDict] = cfg["overview"]
        else:
            self.__overview:dict[int, _OverviewDict] = {}
        self.__single_datas:dict[str, _SingleDataMng] = {}

        self.__IO_Mode = DatasetIOMode.MODE_READ
        self.__is_writing = False

    def __init_subclass__(cls) -> None:
        cls.write   = cls.__O_operations_decorator(cls.write)
        cls.remove  = cls.__O_operations_decorator(cls.remove)
        cls.move    = cls.__O_operations_decorator(cls.move)
        cls.copy    = cls.__O_operations_decorator(cls.copy)

    def _get_overview(self) -> dict[int, _OverviewDict]:
        return self.__overview

    def get_cfg_path(self) -> str:
        return os.path.join(self.directory, f"{self.dataset_name}_overview.json")

    @classmethod
    def from_cfg(cls, path):
        cfg:DatasetCfg = JsonIO.load_json(path)
        dataset = cls(cfg["directory"], cfg["dataset_name"])
        dataset.__overview = cfg["overview"]
        return dataset

    def save_cfg(self, cfg_file:Optional[str] = None) -> None:
        if cfg_file is None:
            cfg_file = self.get_cfg_path()
        cfg:DatasetCfg = {}
        cfg["directory"] = self.__directory
        cfg["dataset_name"] = self.dataset_name
        cfg["classname"] = self.__class__.__name__

        sorted_overview = dict(sorted(self.__overview.items()))
        self.__overview = sorted_overview
        cfg["overview"] = self.__overview
        JsonIO.dump_json(cfg_file, cfg)

    @property
    def IO_Mode(self):
        return self.__IO_Mode

    @property
    def is_writing(self):
        return self.__is_writing

    @property
    def directory(self) -> str:
        return self.__directory

    # region:overview operations
    def get_overview(self) -> MappingView:
        return MappingView(self.__overview)

    def __O_operations_decorator(func:Callable):
        """
        brief
        -----
        decorator for write, remove, move, copy
        """
        def error_catcher(self:"Dataset", *args, **kwargs):
            try:
                return True, func(self, *args, **kwargs)
            except _RequirementError as e:
                print(f"Warning occurs when run '{func.__name__}' of '{self.dataset_name}': {e}")
                return False, None
            except _IdxNotFoundError as e:
                print(f"Warning occurs when run '{func.__name__}' of '{self.dataset_name}': {e}")
                return False, None

        def wrapper(self:"Dataset", *args, **kwargs):
            if self.__IO_Mode == DatasetIOMode.MODE_READ:
                warnings.warn("You are trying to write data in read mode, please set IO_Mode to MODE_APPEND or MODE_WRITE mode")
                return
            operated_names = list(self._get_required_single_datas('w', 'keys'))
            if func.__name__ == "write":
                # before
                if self.has(args[0]) and self.__IO_Mode != DatasetIOMode.MODE_WRITE:
                    warnings.warn(f"Data {args[0]} exists, you are trying to overwrite it. Please set IO_Mode to MODE_WRITE mode before writing.")
                    return 
                # func
                success, rlt = error_catcher(self, *args, **kwargs)
                # after
                if success:
                    self.__overview.setdefault(args[0], _overview_dict_builder())
                    for n in operated_names:
                        self.__overview[args[0]]["data_exists"][n] = True # update data exists
                    self.add_tags(args[0], kwargs.get("tags"))
                return rlt
            elif func.__name__ == "remove":
                # before
                if self.__IO_Mode != DatasetIOMode.MODE_WRITE:
                    warnings.warn(f"Data {args[0]} exists, you are trying to remove it. Please set IO_Mode to MODE_WRITE mode before writing.")
                    return 
                # func
                success, rlt = error_catcher(self, *args, **kwargs)
                # after
                if success:
                    for n in operated_names:
                        self._set_data_exists(args[0], n, False) # update data exists
                    if self._is_empty(args[0]):
                        self.__overview.pop(args[0])
                return rlt
            elif func.__name__ == "move":
                # before
                if self.has(args[0]) and self.__IO_Mode != DatasetIOMode.MODE_WRITE:
                    warnings.warn(f"Data {args[0]} exists, you are trying to overwrite it. Please set IO_Mode to MODE_WRITE mode before writing.")
                    return 
                # func
                success, rlt = error_catcher(self, *args, **kwargs)
                # after
                if success:
                    self.__overview.setdefault(args[1], _overview_dict_builder())
                    for n in operated_names:
                        self._set_data_exists(args[0], n, False) # update data exists
                        self._set_data_exists(args[1], n, True)
                    tags = kwargs.get("tags")
                    if not(isinstance(tags, bool) and not tags):
                        tags = self.__overview[args[0]]["tags"] # move with tags, once not specified False
                    self.set_tags(args[1], tags)
                    if self._is_empty(args[0]):
                        self.__overview.pop(args[0])
                return rlt
            elif func.__name__ == "copy":
                # before
                if self.has(args[0]) and self.__IO_Mode != DatasetIOMode.MODE_WRITE:
                    warnings.warn(f"Data {args[0]} exists, you are trying to overwrite it. Please set IO_Mode to MODE_WRITE mode before writing.")
                    return 
                # func
                success, rlt = error_catcher(self, *args, **kwargs)
                # after
                if success:
                    self.__overview[args[1]] = self.__overview[args[0]]
                    for n in operated_names:
                        self._set_data_exists(args[0], n, False) # update data exists
                    tags = kwargs.get("tags")
                    if isinstance(tags, bool) and tags:
                        tags = self.__overview[args[0]]["tags"] # copy with tags, only when specified True
                    self.set_tags(args[1], tags)
                return rlt
        return wrapper

    def scan(self, max_size = 100000, verbose = False, uncontinuous = False, rescan = False) -> None:
        if rescan:
            self.__overview.clear()
        
        not_exists:dict[str, list[int]] = {}
        for k, d in tqdm(self._get_required_single_datas('none'), desc="Scanning", leave=True):
            for idx in tqdm(range(max_size), desc=f"Scanning {k}", leave=False):
                if isinstance(d, DataCluster):
                    path = d.get_path(idx)
                    if not os.path.exists(path):
                        not_exists.setdefault(k, []).append(idx)
                        if not uncontinuous:
                            break
                        if verbose:
                            print(f"File {path} not exists in {k}")
                else:
                    if idx not in d.keys:
                        not_exists.setdefault(k, []).append(idx)
                        if not uncontinuous:
                            break
                        if verbose:
                            print(f"Data {idx} not exists in {k}")

        if not uncontinuous:
            size = max(reduce(lambda x, y: set(x) | set(y), not_exists.values()), default=max_size)
        else:
            size = max_size
        empty = reduce(lambda x, y: set(x).intersection(y), not_exists.values()) # not exist index
        exist_idx = sorted(list(set(range(size)) - empty)) # exist index
        # incomplete = reduce(lambda x, y: set(x).union(y), not_exists.values()) - empty

        for idx in empty:
            if idx in self.__overview:
                del self.__overview[idx]

        # initialize overview
        _default_data_exists_dict = {name: True for name in not_exists.keys()}
        for idx in tqdm(exist_idx):
            if idx not in self.__overview:
                self.__overview[idx] = _overview_dict_builder()
            self.__overview[idx]["data_exists"] = copy.deepcopy(_default_data_exists_dict)

        # mark those not exists
        for name in not_exists:
            for idx in not_exists[name]:
                if idx in self.__overview:
                    self.__overview[idx]["data_exists"][name] = False
                
        return not_exists

    def delete_incomplete(self) -> None:
        for k, v in self.__overview:
            if v["incomplete"]:
                self.remove(k)

    def make_continuous(self) -> None:
        if self.is_continuous():
            return
        overview = self._get_overview()
        self.__overview = _resort_dict(overview)
        # indices = np.array(sorted(list(overview.keys())), np.int32)
        # target = np.arange(len(indices), dtype=np.int32)
        # new_overview = {}
        # for i, idx in enumerate(indices):
        #     new_overview[target[i]] = overview[idx]
        # self.__overview = new_overview

    def _set_data_exists(self, idx:int, name:str, exists:bool) -> None:
        if exists:
            self.__overview.setdefault(idx, _overview_dict_builder())
            self.__overview[idx]["data_exists"][name] = True
        else:
            if idx in self.__overview:
                self.__overview[idx]["data_exists"][name] = False
            else:
                pass
    # endregion

    # region: manage single data
    def add_single_data(self, name:str, data:SingleData) -> None:
        self.__single_datas[name] = _SingleDataMng(data)
        data._set_dataset(self)

    def remove_single_data(self, name:str) -> None:
        if name in self.__single_datas:
            self.__single_datas[name].data._set_dataset(None)
            del self.__single_datas[name]

    def sort_single_data(self, sorted_keys:Optional[list[str]] = None) -> None:
        if sorted_keys is None:
            sorted_keys = sorted(list(self.__single_datas.keys()))
        else:
            assert set(sorted_keys) == set(self.__single_datas.keys()), "sorted_keys should be the same as the keys of single_datas"
        sorted_datas = {key: self.__single_datas[key] for key in sorted_keys}
        self.__single_datas = sorted_datas

    def get_single_data_names(self, as_tuple = True) -> list[str]:
        if as_tuple:
            return tuple(self.__single_datas.keys())
        else:
            return self.__single_datas.keys()

    def is_complete(self, idx) -> bool:
        if idx not in self._get_overview():
            return False
        return any([not self._get_overview()[idx]["data_exists"].get(name, False) for name in self.get_single_data_names()])

    def _is_empty(self, idx) -> bool:
        # have overview but no data
        if idx not in self._get_overview():
            return False
        return all([not self._get_overview()[idx]["data_exists"].get(name, False) for name in self.get_single_data_names()])

    def _get_single_data(self, name:str) -> SingleData:
        if name not in self.__single_datas:
            return None
        else:
            return self.__single_datas[name].data
    
    def _get_required_single_datas(self, mode = 'r', iter = "items") -> Generator[Tuple[str, SingleData], None, None]:
        """
        mode should be 'r' or 'w' or 'both' or 'none'

        iter should be 'items' or 'values' or 'keys'
        """
        assert mode in ('r', 'w', 'both', 'none'), "mode should be 'r' or 'w'"
        for name, datamng in self.__single_datas.items():
            if  (mode == 'r'    and datamng.read_required)  or\
                (mode == 'w'    and datamng.write_required) or\
                (mode == 'both' and datamng.read_required and datamng.write_required) or\
                (mode == 'none'):
                if iter == 'items':
                    yield name, datamng.data
                elif iter == 'values':
                    yield datamng.data
                elif iter == 'keys':
                    yield name
    
    def set_single_datas_requirement(self, mode = 'r', name:Optional[str] = None, required:Optional[bool] = None, _dict:Optional[dict[str, bool]] = None) -> None:
        """
        mode should be 'r' or 'w'
        """
        assert mode in ('r', 'w'), "mode should be 'r' or 'w'"
        if _dict is None:
            assert name is not None, "name should not be None"
            assert required is not None, "required should not be None"
            _dict = {name: required}
        assert _dict is not None and isinstance(_dict, dict), "_dict should be a dict"
        for name, required in _dict.items():
            if name in self.__single_datas:
                if mode == 'r':
                    self.__single_datas[name].read_required = bool(required)
                else:
                    if not bool(required):
                        warnings.warn(f"You are setting {self.dataset_name}-{name} not to be required when writing, this may cause incomplete data".format(name))
                    self.__single_datas[name].write_required = bool(required)
    # endregion

    # region: read and write operations
    def _raw_read(self, idx:int) -> dict[str, Any]:
        values = {}
        for name, data in self._get_required_single_datas('r'):
            values[name] = data._read(idx)
        return values
    
    def _raw_write(self, idx:int, values:dict[str, Any]) -> None:
        write_required = {name: data for name, data in self._get_required_single_datas('w')}
        if not set(write_required.keys()).issubset(set(values.keys())):
            raise _RequirementError(f"Missing required data {set(write_required.keys()) - set(values.keys())}")
        for name, data in write_required.items():
            data._write(idx, values[name])
    
    def read(self, idx:int) -> dict[str, Any]:
        return self._raw_read(idx)
    
    def write(self, idx:int, values:dict[str, Any], tags:Optional[list[str]|str] = None) -> None:
        return self._raw_write(idx, values)

    def remove(self, idx:int) -> None:
        for name, data in self._get_required_single_datas('w'):
            data._remove(idx)

    def move(self, old_idx:int, new_idx:int, tags:Optional[list[str]|str|bool] = None) -> None:
        for name, data in self._get_required_single_datas('w'):
            data._move(old_idx, new_idx)

    def copy(self, old_idx:int, new_idx:int, tags:Optional[list[str]|str|bool] = None) -> None:
        for name, data in self._get_required_single_datas('w'):
            data._copy(old_idx, new_idx)

    def append(self, values:dict[str, Any], tags:Optional[list[str]|str] = None) -> None:
        idx = self._get_empty_idx()
        self.write(idx, values, tags = tags)

    def save(self, force = True) -> None:
        for name, data in self._get_required_single_datas('w'):
            if not force and data.get_save_mode() == data.O_MODE_MANUAL:
                pass
            else:
                data._save()

    # region:dictlike operations
    def __setitem__(self, idx:int, values:dict[str, Any]) -> None:
        return self.write(idx, values)
    
    def update(self, *args: dict[int, dict[str, Any]]) -> None:
        for d in args:
            for idx, values in d.items():
                return self.write(idx, values)
    # endregion
    # endregion

    # region: IO Controler
    def start_writing(self, mode:Literal[DatasetIOMode.MODE_APPEND, DatasetIOMode.MODE_WRITE] = 2) -> None:
        self.__is_writing = True
        self.__IO_Mode = DatasetIOMode(mode)
        return self
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            raise exc_type(exc_val).with_traceback(exc_tb)
        self.save(force=False)
        self.save_cfg()
        self.__IO_Mode = DatasetIOMode.MODE_READ
        self.__is_writing = False
    # endregion

class Dataset(_Dataset):
    pass

class _DatasetViewItem(TypedDict):
    source_name:str
    source_idx:int

class DatasetViewCfg(TypedDict):
    dataset_name:str
    classname:str
    idx_map: dict[int, _DatasetViewItem]

class _IdxMap(dict[int, _OverviewDict]):
    """
    对外显示的行为是 overview
    """
    def _get_mapping(self, idx) -> _OverviewDict:
        source_name = self._get_inner(idx)["source_name"]
        source_idx  = self._get_inner(idx)["source_idx"]
        datasetlike = _DatasetBase._query_registered(source_name)
        if isinstance(datasetlike, DatasetView):
            return datasetlike._get_overview()._get_mapping(source_idx)
        elif isinstance(datasetlike, Dataset):
            return datasetlike._get_overview()[source_idx]
        else:
            raise RuntimeError(f"Dataset {source_name} not exists")

        # sidx, sds = self.__view._query_source(key)
        # if sds is None:
        #     raise RuntimeError(f"Dataset {self._get_inner(key)['source_name']} not exists")
        # else:
        #     return sds._get_overview()[sidx]
        
    def _get_inner(self, key:int) -> _DatasetViewItem:
        return super().__getitem__(key)
    
    def _set_inner(self, key:int, value: _DatasetViewItem) -> None:
        return super().__setitem__(key, value)
    
    def _del_inner(self, key:int) -> None:
        return super().__delitem__(key)
    
    def _clear_inner(self) -> None:
        return super().clear()
    
    @deprecated("Not call this method, an error will be raised")
    def __delitem__(self, key: int) -> None:
        raise RuntimeError("IdxMap is read-only")
    
    @deprecated("Not call this method, an error will be raised")
    def __setitem__(self, key: int, value: _OverviewDict) -> None:
        raise RuntimeError("IdxMap is read-only")
    
    @deprecated("Not call this method, an error will be raised")
    def clear(self) -> None:
        raise RuntimeError("IdxMap is read-only")
    
    @deprecated("Not call this method, an error will be raised")
    def pop(self, key, default):
        raise RuntimeError("IdxMap is read-only")
    
    @deprecated("Not call this method, an error will be raised")
    def popitem(self) -> Tuple:
        raise RuntimeError("IdxMap is read-only")
    
    @deprecated("Not call this method, an error will be raised")
    def setdefault(self, key, default):
        raise RuntimeError("IdxMap is read-only")
    
    @deprecated("Not call this method, an error will be raised")
    def update(self, *arg, **kwargs):
        raise RuntimeError("IdxMap is read-only")
    
    def __getitem__(self, key: int) -> _OverviewDict:
        return self._get_mapping(key)
    
    def get(self, key, default:Any = None):
        if key in self:
            return self[key]
        else:
            return default
    
    def items(self) -> Generator[Tuple[int, _OverviewDict], None, None]:
        for k in self.keys():
            yield k, self[k]
    
    def values(self) -> Generator[_OverviewDict, None, None]:
        for k in self.keys():
            yield self[k]

class DatasetView(_DatasetBase):
    def __init__(self, dataset_name:str) -> None:
        super().__init__(dataset_name, alterable=True)
        self.__idx_map:_IdxMap = _IdxMap()

    def _get_overview(self) -> _IdxMap:
        return self.__idx_map

    def make_continuous(self) -> None:
        if self.is_continuous():
            return
        idx_map = dict(self._get_overview())
        new_idx_map = _resort_dict(idx_map)
        self.__idx_map = _IdxMap(new_idx_map)
        # indices = np.array(sorted(list(idx_map.keys())), np.int32)
        # target = np.arange(len(indices), dtype=np.int32)
        # new_overview = _IdxMap()
        # for i, idx in enumerate(indices):
        #     new_overview._set_inner(target[i], idx_map._get_inner(idx))
        # self.__idx_map = new_overview

    @classmethod
    def from_cfg(cls, path, dataset_name = None):
        cfg:DatasetViewCfg = JsonIO.load_json(path)
        dataset_name = cfg["dataset_name"] if dataset_name is None else dataset_name
        datasetview = cls(dataset_name)
        datasetview.__idx_map = _IdxMap(cfg["idx_map"])
        return datasetview

    def save_cfg(self, cfg_file:str) -> None:
        cfg:DatasetViewCfg = {}
        cfg["dataset_name"] = self.dataset_name
        cfg["classname"] = self.__class__.__name__
        self.make_continuous()
        # sorted_overview = dict(sorted(self.__idx_map.items()))
        # self.__idx_map = sorted_overview
        cfg["idx_map"] = dict(self.__idx_map)
        JsonIO.dump_json(cfg_file, cfg)

    def read(self, idx:int) -> dict[str, Any]:
        source_dataset, source_idx = self._query_source(idx)
        return source_dataset.read(source_idx)

    def remove(self, idx:int) -> None:
        self.__idx_map._del_inner(idx)

    def move(self, old_idx:int, new_idx:int) -> None:
        self.__idx_map._set_inner(new_idx, self.__idx_map._get_inner(old_idx))
        self.__idx_map._del_inner(old_idx)
    
    def copy(self, old_idx:int, new_idx:int) -> None:
        self.__idx_map._set_inner(new_idx, self.__idx_map._get_inner(old_idx))

    def is_complete(self, idx) -> bool:
        source_dataset, source_idx = self._query_source(idx)
        return source_dataset.is_complete(source_idx)

    def _is_empty(self, idx) -> bool:
        source_dataset, source_idx = self._query_source(idx)
        return source_dataset._is_empty(source_idx)

    # region: datasetview operations
    def add_dataset(self, dataset:_DatasetBase, indices:Optional[Iterable[int]] = None) -> None:
        indices = dataset.keys() if indices is None else indices
        length = len(indices)
        start = self._get_empty_idx()
        for idx, source_idx in zip(range(start, start + length), indices):
            self.__idx_map._set_inner(idx, {"source_name": dataset.dataset_name, "source_idx": source_idx})

    def add(self, idx:int, source_name:Dataset|str, source_idx:int) -> None:
        # TODO
        if isinstance(source_name, Dataset):
            source_name = source_name.dataset_name
        assert isinstance(source_name, str), "source_name should be a str"
        self.__idx_map._set_inner(idx, {"source_name": source_name, "source_idx": source_idx})

    def select( self, 
                include_tags:Optional[Iterable[str]|str]    = None, 
                exclude_tags:Optional[Iterable[str]|str]    = None,
                include_indices:Optional[Iterable[int]|int] = None,
                exclude_indices:Optional[Iterable[int]]     = None,
                include_shallow_dataset_names:Optional[Iterable[str]|str]   = None,
                exclude_shallow_dataset_names:Optional[Iterable[str]|str]    = None,
                include_source_dataset_names:Optional[Iterable[str]|str]    = None,
                exclude_source_dataset_names:Optional[Iterable[str]|str]     = None) -> list[int]:
                
        tags_filter    = self._select(include_tags,                     exclude_tags,                   str)
        indices_filter = self._select(include_indices,                  exclude_indices,                int)
        shallow_filter = self._select(include_shallow_dataset_names,    exclude_shallow_dataset_names,  str)
        source_filter  = self._select(include_source_dataset_names,     exclude_source_dataset_names,    str)
        
        selected_indices = []
        for idx in self.keys():
            tags = self.get_tags(idx)
            shallow_ds_name = self.__idx_map._get_inner(idx)["source_name"]
            source_ds_name  = self._query_source(idx)[0].dataset_name
            if  tags_filter(tags) and \
                indices_filter(idx) and\
                shallow_filter(shallow_ds_name) and\
                source_filter(source_ds_name):
                selected_indices.append(idx)

        return selected_indices
    
    def _get_ref_chain(self, idx):
        chain:list[tuple[Dataset, int]] = []
        def func(ds:Dataset|DatasetView, idx:int):
            source_name = ds.__idx_map._get_inner(idx)["source_name"]
            source_idx  = ds.__idx_map._get_inner(idx)["source_idx"]
            source_dataset = ds._query_registered(source_name)
            chain.append((source_dataset, source_idx))
            if isinstance(source_dataset, DatasetView):
                func(source_dataset, source_idx)
            else:
                return
        func(self, idx)
        return chain
    
    def _query_source(self, idx:int) -> tuple[Dataset, int]:
        chain = self._get_ref_chain(idx)
        return chain[-1]
    
    def print_ref_chain(self, idx:int) -> None:
        chain = self._get_ref_chain(idx)
        string = f"{self}[{idx}] -> " + " -> ".join([f"{ds.dataset_name}[{idx}]" for ds, idx in chain])
        print(string)
        
    def copy_view(self, new_name = None):
        if new_name is None:
            new_name = self.dataset_name + "_copy"
        new_view = self.__class__(new_name)
        new_view.__idx_map = _IdxMap(self.__idx_map)
        return new_view

    # endregion