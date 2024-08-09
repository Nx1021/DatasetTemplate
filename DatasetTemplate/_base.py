from typing import List, Tuple, Dict, Any, Union, Callable, Optional, TypedDict, Literal, Self
from typing import TypeVar, Generic, Iterable, Generator, MappingView, Mapping
from typing_extensions import deprecated
from enum import Enum
import os
import shutil
import copy
from tqdm import tqdm
from functools import partial, reduce
import numpy as np

from abc import ABC, abstractmethod, ABCMeta

import warnings

from .utils import JsonIO
from functools import reduce

def custom_warning_format(message, category, filename, lineno, file=None, line=None):
    return f"{category.__name__} at {filename}:{lineno}: {message}\n"

# 设置自定义警告格式
warnings.formatwarning = custom_warning_format

DEBUG = False

DATA_TYPE = TypeVar("DATA_TYPE")
_T = TypeVar("_T")

def inherit_private(child:type, parent:type, name):
    name = f"_{parent.__name__}{name}"
    setattr(child, name, getattr(parent, name))

def method_hook_decorator(cls, func:Callable,
                          enter_hook_func:Optional[Callable[..., None]] = None,
                          exit_hook_func:Optional[Callable[..., None]] = None,
                          enter_condition_func:Optional[Callable[..., bool]] = None, 
                          not_enter_warning = "") -> Callable:
    """
    Decorator function that adds an exit hook to a method based on an enter condition.
    """
    # enter_hook_func        = enter_hook_func       if enter_hook_func is not None else (lambda obj, *args, **kw : None)
    # exit_hook_func         = exit_hook_func        if exit_hook_func is not None else (lambda obj, *args, **kw : None)
    # enter_condition_func   = enter_condition_func  if enter_condition_func is not None else (lambda obj, *args, **kw : True) 

    def wrapper(self, *args, **kw):
        if enter_condition_func is None or enter_condition_func(self, *args, **kw):
            if cls == self.__class__ and enter_hook_func is not None:
                enter_hook_func(self, *args, **kw)
            rlt = func(self, *args, **kw)
            if cls == self.__class__ and exit_hook_func is not None:
                exit_hook_func(self, *args, **kw)
                if DEBUG:
                    print(f"{self} runs {func.__name__}")
            return rlt
        else:
            if not_enter_warning:
                warnings.warn(f"{self.__class__.__name__}.{func.__name__} is not executed. {not_enter_warning}")
            return None
    wrapper.__name__ = func.__name__
    return wrapper

def _resort_dict(d:dict[int, Any]) -> dict[int, Any]:
    assert isinstance(d, dict), "d should be a dict"
    indices = np.array(sorted(list(d.keys())), np.int32)
    target  = np.arange(len(indices), dtype=np.int32)
    new_d = {k:v for k, v in zip(target, map(lambda x: d[x], indices))} 
    return new_d

class O_Mode(Enum):
    IMIDIATE = 1
    EXIT     = 2
    MANUAL   = 4

class _DecoratingAfterInit(ABCMeta, type):
    __is_decorating = False

    def __init__(cls, name, bases, dct):
        super(_DecoratingAfterInit, cls).__init__(name, bases, dct)
        if hasattr(cls, "_decorating_after_init"):
            with cls:
                getattr(cls, "_decorating_after_init")()

    def __setattr__(cls, name, value):
        # 在设置属性时执行的自定义操作
        if cls.__is_decorating:
            if name in cls.__dict__:
                super().__setattr__(name, value)
            else:
                return
        else:
            super().__setattr__(name, value)

    def is_decorating(cls):
        return cls.__is_decorating
    
    def __enter__(cls):
        cls.__is_decorating = True

    def __exit__(cls, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print(f"An error occurred: {exc_type}, {exc_val}")
        cls.__is_decorating = False

class _SingleData(ABC, Generic[DATA_TYPE], metaclass=_DecoratingAfterInit):
    """
    Brief
    -----
    a class to read/write data in a single format. Each index corresponds to a piece of data.

    It's **an abstract class**, you should inherit this class and implement the following methods:
    - `_write`
    - `_read`
    - `_remove`
    - `_move`
    - `_copy`
    - `_save`
    - `get_path`

    Attributes
    ----
    O_MODE_IMIDIATE: int = 1
        Specify how to save datas to the disk. `set_save_mode` and `get_save_mode` are used to set and get the save mode.
    O_MODE_EXIT: int = 2
        save data when exit
    O_MODE_MANUAL: int = 4
        save data manually
    """

    # O_MODE_IMIDIATE = 1
    # O_MODE_EXIT     = 2
    # O_MODE_MANUAL   = 4

    TYPE_CLUSTER = "cluster"
    TYPE_FILE    = "file"

    __subclass_registry = {}

    def __init__(self,
                 read_func:Optional[Callable[[str], DATA_TYPE]],
                 write_func:Optional[Callable[[str, DATA_TYPE],Any]]) -> None:
        self.__dataset = None
        self._data_cache:dict[int, DATA_TYPE] = {}
        self.__save_mode = O_Mode.IMIDIATE
        self._read_func = read_func
        self._write_func = write_func

    @classmethod
    def _decorating_after_init(cls):
        def init_check_func(self:"_SingleData", *args, **kw):
            if self._read_func is None:
                warnings.warn(f"read_func for {self.__class__.__name__} is None, you should set it manually")
            elif not isinstance(self._read_func, Callable):
                raise ValueError(f"read_func should be a Callable, but got {type(self._read_func)}")
            if self._write_func is None:
                warnings.warn("write_func for {self.__class__.__name__} is None, you should set it manually")
            elif not isinstance(self._write_func, Callable):
                raise ValueError(f"write_func should be a Callable, but got {type(self._write_func)}")

        def ready_check_func(self:"_SingleData", *args, **kw):
            if self.is_ready():
                return True
            else:
                warnings.warn(f"{self.__class__.__name__} instance is not ready.")
        
        cls.__init__ = method_hook_decorator(cls, cls.__init__, exit_hook_func=init_check_func)
        cls._read    = method_hook_decorator(cls, cls._read,    enter_condition_func=ready_check_func, not_enter_warning="It's not ready!")
        cls._write   = method_hook_decorator(cls, cls._write,   enter_condition_func=ready_check_func, not_enter_warning="It's not ready!")
        cls._remove  = method_hook_decorator(cls, cls._remove,  enter_condition_func=ready_check_func, not_enter_warning="It's not ready!")
        cls._move    = method_hook_decorator(cls, cls._move,    enter_condition_func=ready_check_func, not_enter_warning="It's not ready!")
        cls._copy    = method_hook_decorator(cls, cls._copy,    enter_condition_func=ready_check_func, not_enter_warning="It's not ready!")
        cls._save    = method_hook_decorator(cls, cls._save,    enter_condition_func=ready_check_func, not_enter_warning="It's not ready!")                   

    def __init_subclass__(cls) -> None:
        _SingleData.__subclass_registry[cls.__name__] = cls

    @classmethod
    def _query_SingleData_Subclass(cls, name:str) -> "_SingleData":
        return cls.__subclass_registry[name]

    @property
    def dataset(self):
        """
        property, return the dataset that this SingleData is binded to.
        """
        return self.__dataset

    def _set_dataset(self, dataset:Union["Dataset",None]) -> None:
        """
        Brief
        -----
        set the dataset that this SingleData is binded to.

        Parameters
        -----
        dataset: Dataset | None
            the dataset that this SingleData is binded to. If `dataset` is None, the SingleData will be unbinded to any dataset.
            If `dataset` is an instance of `Dataset`, this SingleData will be binded to the dataset. 
            Other types will raise a TypeError.
        """
        if isinstance(dataset, Dataset):
            self.__dataset = dataset
            path = self.get_path(0)
            os.makedirs(os.path.dirname(path), exist_ok=True)
        elif dataset is None:
            self.__dataset = None
        else:
            raise TypeError("dataset should be an instance of Dataset or None")

    @property
    def root_dir(self) -> str|None:
        """
        return the `directory` of the dataset that this SingleData is binded to.
        """
        return self.dataset.directory if self.dataset is not None else None

    @property
    def data_cache(self) -> Mapping[int, DATA_TYPE]:
        """
        return the a MappingView of the `_data_cache`. `_data_cache` is a dictionary that stores the data in memory. 
        When `__save_mode` is set to `O_MODE_EXIT` or `O_MODE_MANUAL`, any modification will be stored in `_data_cache` firstly instead of the disk. 
        """
        return MappingView(self._data_cache)

    @abstractmethod
    def _write(self, idx:int, data:DATA_TYPE):
        pass

    @abstractmethod
    def _read(self, idx:int):
        pass

    @abstractmethod
    def _remove(self, idx:int):
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

    def set_save_mode(self, mode:Literal["IMIDIATE", "EXIT", "MANUAL", 1, 2, 4, O_Mode.IMIDIATE, O_Mode.EXIT, O_Mode.MANUAL]) -> None:
        """
        Parameters
        -----
        mode: int = O_MODE_IMIDIATE(1) | O_MODE_EXIT(2) | O_MODE_MANUAL(4)
            Specify how to save datas to the disk. `set_save_mode` and `get_save_mode` are used to set and get the `__save_mode`.
        """
        if isinstance(mode, str):
            assert mode in ("IMIDIATE", "EXIT", "MANUAL"), f"Invalid save_mode {mode}, expected 'IMIDIATE', 'EXIT', 'MANUAL'"
            mode = O_Mode[mode]
        elif isinstance(mode, int):
            assert mode in (1, 2, 4), f"Invalid save_mode {mode}, expected 1, 2, 4"
            mode = O_Mode(mode)
        assert mode in (O_Mode.IMIDIATE, O_Mode.EXIT, O_Mode.MANUAL), f"Invalid save_mode {mode}"
        self.__save_mode = mode

    def get_save_mode(self) -> O_Mode:
        """
        Return the `__save_mode` of this SingleData.
        """
        return self.__save_mode
    
    @abstractmethod
    def get_path(self, idx:int) -> str:
        pass

    def is_ready(self, verbose = True) -> bool:
        if self._read_func is None:
            if verbose:
                warnings.warn(f"read_func for {self.__class__.__name__} is None, you should set it manually")
            return False
        elif not isinstance(self._read_func, Callable):
            if verbose:
                raise ValueError(f"read_func should be a Callable, but got {type(self._read_func)}")
            return False
        if self._write_func is None:
            if verbose:
                warnings.warn("write_func for {self.__class__.__name__} is None, you should set it manually")
            return False
        elif not isinstance(self._write_func, Callable):
            if verbose:
                raise ValueError(f"write_func should be a Callable, but got {type(self._write_func)}")
            return False
        if self.dataset is None:
            if verbose:
                warnings.warn(f"{self.__class__.__name__} instance is not binded to any dataset")
            return False
        elif not isinstance(self.dataset, Dataset):
            if verbose:
                warnings.warn(f"{self.__class__.__name__} instance is not correctly binded. Expected Dataset, but got {type(self.dataset)}")
            return False
        return True

class default_filepath_generator():
    """
    Brief
    -----
    a class to generate file path from index with a format string.
    filename_generator_builder instances are callable. Calling an instance of filename_generator_builder will return a file path.

    Examples
    -----
    ```python
    fg = filename_generator_builder("pre_{}_apd.txt", (4, "0"))
    path = fg(1)
    print(path) # "pre_0001_apd.txt"

    fg = filename_generator_builder("subdir/subsubdir/pre_{}_apd.txt", (6, "x"))
    path = fg(100)
    print(path) # "subdir/subsubdir/pre_xxx100_apd.txt"
    ```
    """
    def __init__(self, format_str:str, rjust_params:tuple[int, str] = (4, "0")) -> None:
        """
        Brief
        -----
        Build a filename generator with a format string and `rjust_params` parameters.

        Parameters
        -----
        format_str: str
            a format string to generate file path from index
        rjust_params: tuple[int, str] = (4, "0")
            a tuple of two elements, the first element is the width of the index, the second element is the fillchar of the
            index. It's used to fill the index to the specified width.
        
        Examples
        -----
        ```python
        fg = filename_generator_builder("pre_{}_apd.txt", (4, "0"))
        path = fg(1)
        print(path) # "pre_0001_apd.txt"

        fg = filename_generator_builder("subdir/subsubdir/pre_{}_apd.txt", (6, "x"))
        path = fg(100)
        print(path) # "subdir/subsubdir/pre_xxx100_apd.txt"
        ```
        """
        self.format_str = format_str
        self.rjust_params = rjust_params

    def __call__(self, idx:int) -> str:
        return self.format_str.format(str(idx).rjust(*self.rjust_params))

class DataCluster(_SingleData[DATA_TYPE], Generic[DATA_TYPE]):
    """
    Brief
    -----
    a class to read/write data in a cluster of files. Each file contains one piece of data. 

    It's **highly recommended** to inherit this class and bind the parameters: `read_func`/`write_func` of `__init__` method, 
    which is to facilitate building DataCluster objects directly by `from_cfg` without any other operations.

    Example
    -----
    ```python
    ## Example 1: binding and `read_func`/`write_func`
    class NdarrayNpyCluster(DataCluster):
        def __init__(self, path_generator:Callable[[int], str]) -> None:
            from numpy import load, save
            super().__init__(path_generator, load, save)
    
    ## Example 2: usage
    # files directory:
    # - root
    #   - numpydata
    #     - 0000.npy
    #     - 0001.npy
    #     - 0002.npy
    #     - ...
    #     - 9999.npy
    cluster = NdarrayNpyCluster(filename_generator_builder("numpydata/{}.npy", (4, "0")))
    dataset = Dataset("root")
    dataset.add_single_data("numpydata", cluster)
    data_0 = cluster._read(0) # equivalent to np.load("root/numpydata/0000.npy")
    cluster._write(0, data_0) # equivalent to np.save("root/numpydata/0000.npy", data_0)
    ```

    NOTE
    -----
    It's **not recommended** to use `_read`/`_write`/`_remove`/`_move`/`_copy`/`_save` method directly. 
    Because there are always multiple kinds of data format in one dataset, directly modifying one single-modal data may cause inconsistency.
    You should use `read`/`write`/`remove`/`move`/`move`/`copy`/`save` method of `Dataset` instance instead.
    """
    def __init__(self, 
                 path_generator:Optional[Callable[[int], str]] = None,
                 read_func:Optional[Callable[[str], DATA_TYPE]] = None,
                 write_func:Optional[Callable[[str, DATA_TYPE], Any]] = None
                 ) -> None:
        """
        Brief
        -----
        a class to read/write data in a cluster of files. Each file contains one piece of data. 

        It's **highly recommended** to inherit this class and bind the parameters: `read_func`/`write_func` of `__init__` method.
        
        parameters
        -----
        path_generator: Callable[[int], str]
            a function to generate file path from index
        read_func: Callable[[str], dict[int, DATA_TYPE]]
            a function to read data from the `file_path`
        write_func: Callable[[str, dict[int, DATA_TYPE]], Any]
            a function to write data to the `file_path`
        """
        super().__init__(read_func, write_func)
        self.path_generator = path_generator
        self.__operating_files = False

    def _read(self, idx:int):
        """
        Brief
        -----
        read data from a file, the file path is generated by `path_generator`.

        Parameters
        -----
        idx: int
            the index of the data, which is used to generate the file path.
        """
        path = self.get_path(idx)
        if not os.path.exists(path):
            warnings.warn(f"File {path} not exists")
            return None
        return self._read_func(self.get_path(idx))
    
    def _write(self, idx:int, value:DATA_TYPE):
        """
        Warning
        -----
        Not recommended to use this method directly. You should use `append`/`write`/`__setitem__` method of `Dataset` instance instead.
        
        Brief
        -----
        write data to a file, the file path is generated by `path_generator`.

        Parameters
        -----
        idx: int
            the index of the data, which is used to generate the file path.
        value: DATA_TYPE
            the data to be written to the file.

        - If `self.set_save_mode(O_MODE_IMIDIATE)` is called, the data will be written to the file immediately.
        - If `self.set_save_mode(O_MODE_EXIT)` is called,     the data will be stored in `_data_cache` instead of the file. `_save` will be executed when the writing context of `self.dataset` exits.
        - If `self.set_save_mode(O_MODE_MANUAL)` is called,   the data will be stored in `_data_cache` instead of the file. `_save` or `self.dataset.save` should be executed manually. 

        Examples
        -----
        ```python
        import numpy as np
        import os

        cluster = NdarrayNpyCluster(filename_generator_builder("numpydata/{}.npy", (4, "0")))
        dataset = Dataset("root")
        dataset.add_single_data("numpydata", cluster)
        with dataset.start_writing('w'):
            dataset.clear()

        data_0 = np.random.rand(10, 10)
        # default save_mode is O_MODE_IMIDIATE
        cluster._write(0, data_0) 
        print(os.path.exists("root/numpydata/0000.npy")) # True
        cluster._move(0, 1)
        print(os.path.exists("root/numpydata/0000.npy")) # False
        print(os.path.exists("root/numpydata/0001.npy")) # True
        cluster._copy(1, 2)
        print(os.path.exists("root/numpydata/0002.npy")) # True
        cluster._remove(1)
        cluster._remove(2)

        # set save_mode to O_MODE_EXIT
        cluster.set_save_mode(O_Mode._EXIT)
        cluster._write(1, data_0)
        print(os.path.exists("root/numpydata/0001.npy")) # False
        cluster._remove(1)
        with dataset.start_writing('w'):
            dataset.write(1, {"numpydata": data_0}) # equivalent to cluster._write(1, data_0), but not recommended
        print(os.path.exists("root/numpydata/0001.npy")) # True

        # set save_mode to O_MODE_EXIT
        cluster.set_save_mode(O_Mode.MANUAL)
        cluster._write(2, data_0)
        print(os.path.exists("root/numpydata/0002.npy")) # False
        cluster._remove(2)
        with dataset.start_writing('w'):
            dataset.write(2, {"numpydata": data_0}) # equivalent to cluster._write(2, data_0), but not recommended
        print(os.path.exists("root/numpydata/0002.npy")) # False
        dataset.save() # equivalent to cluster._save(2), but not recommended
        print(os.path.exists("root/numpydata/0002.npy")) # True
        ```
        """
        if self.get_save_mode() == O_Mode.IMIDIATE or self.__operating_files:
            path = self.get_path(idx)
            self._write_func(path, value)
        elif self.get_save_mode() == O_Mode.EXIT or self.get_save_mode() == O_Mode.MANUAL:
            assert value is not None, f"Data {idx} should not be None, if you want to remove it, please use remove method"
            self._data_cache[idx] = value

    def _remove(self, idx:int):
        """
        Warning
        -----
        Not recommended to use this method directly. You should use `remove` method of `Dataset` instance instead.

        Brief
        -----
        remove a file, the file path is generated by `path_generator`.

        Parameters
        -----
        idx: int
            the index of the data, which is used to generate the file path.

        Examples
        -----
        see `_write` method
        """
        if self.get_save_mode() == O_Mode.IMIDIATE or self.__operating_files:
            path = self.get_path(idx)
            if not os.path.exists(path):
                _IdxNotFoundError(f"File {path} not exists")
                return
            try:
                os.remove(path)
            except Exception as e:
                pass 
        elif self.get_save_mode() == O_Mode.EXIT or self.get_save_mode() == O_Mode.MANUAL:
            if idx in self._data_cache:
                self._data_cache[idx] = None

    def _move(self, old_idx:int, new_idx:int):
        """
        Warning
        -----
        Not recommended to use this method directly. You should use `move` method of `Dataset` instance instead.

        Brief
        -----
        move a file to another path, the file path is generated by `path_generator`.

        Parameters
        -----
        idx: int
            the index of the data, which is used to generate the file path.

        Examples
        -----
        see `_write` method
        """
        if self.get_save_mode() == O_Mode.IMIDIATE or self.__operating_files:
            old_path = self.get_path(old_idx)
            new_path = self.get_path(new_idx)
            if not os.path.exists(old_path):
                raise _IdxNotFoundError(f"File {old_path} not exists")
            try:
                os.rename(old_path, new_path)
            except Exception as e:
                pass
        elif self.get_save_mode() == O_Mode.EXIT or self.get_save_mode() == O_Mode.MANUAL:
            if old_idx in self._data_cache:
                self._data_cache[new_idx] = self._data_cache[old_idx]
                self._data_cache[old_idx] = None

    def _copy(self, old_idx:int, new_idx:int):
        """
        Warning
        -----
        Not recommended to use this method directly. You should use `copy` method of `Dataset` instance instead.

        Brief
        -----
        copy a file to another path, the file path is generated by `path_generator`.

        Parameters
        -----
        idx: int
            the index of the data, which is used to generate the file path.

        Examples
        -----
        see `_write` method
        """
        if self.get_save_mode() == O_Mode.IMIDIATE or self.__operating_files:
            old_path = self.get_path(old_idx)
            new_path = self.get_path(new_idx)
            if not os.path.exists(old_path):
                raise _IdxNotFoundError(f"File {old_path} not exists")
            try:
                shutil.copy(old_path, new_path)
            except Exception as e:
                pass
        elif self.get_save_mode() == O_Mode.EXIT or self.get_save_mode() == O_Mode.MANUAL:
            if old_idx in self._data_cache:
                self._data_cache[new_idx] = self._data_cache[old_idx]

    def _save(self):
        """
        Warning
        -----
        Not recommended to use this method directly. You should use `save` method of `Dataset` instance instead.

        Brief
        -----
        save all data in `_data_cache` to the disk.
        """
        self.__operating_files = True
        for idx, data in self._data_cache.items():
            if data is not None:
                self._write(idx, data)
            else:
                self._remove(idx)
        self.__operating_files = False
        self._data_cache.clear()

    def get_path(self, idx:int):
        """
        Brief
        -----
        generate file path from index. If `self.dataset` is `None`, return `None`.
        """
        if self.root_dir is None:
            return None
        else:
            return os.path.join(self.root_dir, self.path_generator(idx))
        
    def is_ready(self):
        return super().is_ready() and isinstance(self.path_generator, Callable)
    
APD_IDX_TYPE = TypeVar("APD_IDX_TYPE")
class _DataCluster_MultiToOne(_SingleData[DATA_TYPE], Generic[DATA_TYPE, APD_IDX_TYPE]):
    """
    a variant of DataCluster, which is used to store one data in multiple files
    """
    def __init__(self,
                 read_func:Callable[[str], None],
                 write_func:Callable[[str, DATA_TYPE],int],
                 path_generator:Callable[[int, Iterable[APD_IDX_TYPE]], Iterable[str]]) -> None:
        super().__init__(read_func, write_func)
        self.path_generator = path_generator

    def _read(self, idx, append_indices:list[APD_IDX_TYPE]):
        paths= self.get_path(idx, append_indices)
        results:dict[APD_IDX_TYPE, DATA_TYPE|None] = {}
        for path, apn in zip(paths, append_indices):
            if not os.path.exists(path):
                warnings.warn(f"File {path} not exists")
                results[apn] = None
            else:
                results[apn] = self._read_func(path)
        return results
    
    def _write(self, idx, value:dict[APD_IDX_TYPE, DATA_TYPE]):
        if self.get_save_mode() == O_Mode.IMIDIATE:
            append_indices = list(value.keys())
            paths = self.get_path(idx, append_indices)
            for p, v in zip(paths, value.values()):
                self._write_func(p, v)
        elif self.get_save_mode() == O_Mode.EXIT or self.get_save_mode() == O_Mode.MANUAL:
            assert value is not None, f"Data {idx} should not be None, if you want to remove it, please use remove method"
            self._data_cache[idx] = value

    def _remove(self, idx, append_indices:list[APD_IDX_TYPE]):
        if self.get_save_mode() == O_Mode.IMIDIATE:
            paths= self.get_path(idx, append_indices)
            for path in paths:
                if not os.path.exists(path):
                    _IdxNotFoundError(f"File {path} not exists")
                    return
                try:
                    os.remove(path)
                except Exception as e:
                    pass 
        elif self.get_save_mode() == O_Mode.EXIT or self.get_save_mode() == O_Mode.MANUAL:
            if idx in self._data_cache:
                self._data_cache[idx] = None

    def _move(self, old_idx, new_idx, append_indices:list[APD_IDX_TYPE]):
        if self.get_save_mode() == O_Mode.IMIDIATE:
            old_paths = self.get_path(old_idx, append_indices)
            new_paths = self.get_path(new_idx, append_indices)
            for old_path, new_path in zip(old_paths, new_paths):
                if not os.path.exists(old_path):
                    raise _IdxNotFoundError(f"File {old_path} not exists")
                try:
                    os.rename(old_path, new_path)
                except Exception as e:
                    pass
        elif self.get_save_mode() == O_Mode.EXIT or self.get_save_mode() == O_Mode.MANUAL:
            if old_idx in self._data_cache:
                self._data_cache[new_idx] = self._data_cache[old_idx]
                self._data_cache[old_idx] = None

    def _copy(self, old_idx, new_idx, append_indices:list[APD_IDX_TYPE]):
        if self.get_save_mode() == O_Mode.IMIDIATE:
            old_paths = self.get_path(old_idx, append_indices)
            new_paths = self.get_path(new_idx, append_indices)
            for old_path, new_path in zip(old_paths, new_paths):
                if not os.path.exists(old_path):
                    raise _IdxNotFoundError(f"File {old_path} not exists")
                try:
                    shutil.copy(old_path, new_path)
                except Exception as e:
                    pass
        elif self.get_save_mode() == O_Mode.EXIT or self.get_save_mode() == O_Mode.MANUAL:
            if old_idx in self._data_cache:
                self._data_cache[new_idx] = self._data_cache[old_idx]

    def _save(self):
        for idx, data in self._data_cache.items():
            if data is not None:
                self._write(idx, data)
            else:
                self._remove(idx)
        self._data_cache.clear()

    def get_path(self, idx:int, append_indices:Iterable[APD_IDX_TYPE]):
        # if not _return_files:
        #     return [os.path.join(self.root_dir, x) for x in self.path_generator(idx, append_indices)]
        # else:
        files = self.path_generator(idx, append_indices)
        return [os.path.join(self.root_dir, x) for x in files]

    def scan(self, idx_range:Iterable[int], append_indices:list[APD_IDX_TYPE], all_required = True, verbose = False):
        not_exist = []
        for idx in idx_range:
            paths = self.get_path(idx, append_indices)
            if  (all_required     and not any([os.path.exists(path) for path in paths])) or\
                (not all_required and any([os.path.exists(path) for path in paths])):
                not_exist.append(idx)
                if verbose:
                    print(f"Files of {idx} not exists")
        return not_exist

class DataFile(_SingleData[DATA_TYPE], Generic[DATA_TYPE]):
    """
    Brief
    -----
    a class to read/write data in a single file. 
    All data are stored in one file in a `dict` format, such as a json/xml file.

    It's **highly recommended** to inherit this class and bind the parameters: `read_func`/`write_func` of `__init__` method, 
    which is to facilitate building DataCluster objects directly by `from_cfg` without any other operations.
    
    Example
    -----
    ```python
    ## Example 1: binding and `read_func`/`write_func`
    class JsonFile(DataFile):
        def __init__(self, file_path:str) -> None:
            super().__init__(JsonIO.load_json, JsonIO.dump_json, file_path)

    ## Example 2: usage
    # files directory:
    # - root
    #   - file.json
    file = JsonFile("file.json")
    dataset = Dataset("root")
    dataset.add_single_data("file", file)

    data = "Hello, World!"
    data = file._read(0) # equivalent to JsonIO.load_json("root/file.json")
    file._write(0, data) # equivalent to JsonIO.dump_json("root/file.json", data)
    ```

    NOTE
    -----
    It's **not recommended** to use `_read`/`_write`/`_remove`/`_move`/`_copy`/`_save` method directly. 
    Because there are always multiple kinds of data format in one dataset, directly modifying one single data in the cluster may cause inconsistency.
    You should use `read`/`write`/`remove`/`move`/`move`/`copy`/`save` method of `Dataset` instance instead.

    It's **not recommended** to `set_save_mode` to `O_MODE_IMIDIATE`. Because it will cause frequent IO operation and may slow down the program.
    """
    def __init__(self, file_path:str,
                 read_func:Callable[[str],None] = None,
                 write_func:Callable[[str, DATA_TYPE],int] = None) -> None:
        """
        Brief
        -----
        a class to read/write data in a single file. 
        All data are stored in one file in a `dict` format, such as a json/xml file.

        It's **highly recommended** to inherit this class and bind the parameters: `read_func`/`write_func` of `__init__` method.
        
        parameters
        -----
        read_func: Callable[[str], DATA_TYPE]
            a function to read data from a file
        write_func: Callable[[str, DATA_TYPE], Any]
            a function to write data to a file
        path_generator: Callable[[int], str]
            a function to generate file path from index
        """
        super().__init__(read_func, write_func)
        self.file_path = file_path
        self.set_save_mode(O_Mode.EXIT)
        self._data_cache = {}

    @property
    def keys(self) -> list[int]:
        return self._data_cache.keys()

    def _set_dataset(self, dataset: "Dataset") -> None:
        super()._set_dataset(dataset)
        if not os.path.exists(self.get_path()):
            warnings.warn(f"initialize {self.__class__.__name__}: File {self.get_path()} not exists")
            self._data_cache = {}
        else:
            data = self._read_func(self.get_path())
            self._data_cache = self._cvt_to_data_cache(data)

    def _read(self, idx:int):
        """
        Brief
        -----
        read data in `_data_cache`.

        Parameters
        -----
        idx: int
            the index of the data.
        """
        return self._data_cache[idx]

    def _write(self, idx:int, value:DATA_TYPE):
        """
        Warning
        -----
        Not recommended to use this method directly. You should use `append`/`write`/`__setitem__` method of `Dataset` instance instead.
        
        Brief
        -----
        write data to a file, the file path is generated by `path_generator`.

        Parameters
        -----
        idx: int
            the index of the data, which is used to generate the file path.
        value: DATA_TYPE
            the data to be written to the file.

        - If `self.set_save_mode(O_MODE_IMIDIATE)` is called, the data will be written to the file immediately.
        - If `self.set_save_mode(O_MODE_EXIT)` is called,     the data will be stored in `_data_cache` instead of the file. `_save` will be executed when the writing context of `self.dataset` exits.
        - If `self.set_save_mode(O_MODE_MANUAL)` is called,   the data will be stored in `_data_cache` instead of the file. `_save` or `self.dataset.save` should be executed manually. 

        Examples
        -----
        ```python
        import numpy as np
        import os

        file = JsonFile("file.json")
        dataset = Dataset("root")
        dataset.add_single_data("file", file)
        with dataset.start_writing('w'):
            dataset.clear()

        data_0 = {"a": 1, "b": 2}
        file._write(0, data_0) 
        file._move(0, 1)
        print(file._read(0)) # None
        file._copy(1, 2)
        print(file._read(2)) # {"a": 1, "b": 2}
        file._remove(1)
        print(file._read(1)) # None
        file._save()
        ```
        """
        self._data_cache[idx] = value
        if self.get_save_mode() == O_Mode.IMIDIATE:
            self._save()    

    def _remove(self, idx:int):
        """
        Warning
        -----
        Not recommended to use this method directly. You should use `remove` method of `Dataset` instance instead.

        Brief
        -----
        remove an item in `_data_cache`.

        Parameters
        -----
        idx: int
            the index of the data.

        Examples
        -----
        see `_write` method
        """
        self._data_cache[idx] = None
        if self.get_save_mode() == O_Mode.IMIDIATE:
            self._save()
    
    def _move(self, old_idx:int, new_idx:int):
        """
        Warning
        -----
        Not recommended to use this method directly. You should use `move` method of `Dataset` instance instead.

        Brief
        -----
        move a value in `_data_cache` to another index.

        Parameters
        -----
        idx: int
            the index of the data.

        Examples
        -----
        see `_write` method
        """
        self._data_cache[new_idx] = self._data_cache[old_idx]
        self._data_cache[old_idx] = None
        if self.get_save_mode() == O_Mode.IMIDIATE:
            self._save()

    def _copy(self, old_idx:int, new_idx:int):
        """
        Warning
        -----
        Not recommended to use this method directly. You should use `copy` method of `Dataset` instance instead.

        Brief
        -----
        copy a value in `_data_cache` to another index.

        Parameters
        -----
        idx: int
            the index of the data.

        Examples
        -----
        see `_write` method
        """
        self._data_cache[new_idx] = self._data_cache[old_idx]
        if self.get_save_mode() == O_Mode.IMIDIATE:
            self._save()

    def _cvt_to_data_cache(self, data):
        """
        Brief
        -----
        convert data to `_data_cache` format. If `data` is not a `dict`, it will raise an error.
        If the data is a list-like object, such as `np.ndarray`, you should use `DataFile_ListLike` instead of `DataFile`.
        Or you should implement `_cvt_to_data_cache` and `_cvt_from_data_cache` method in your subclass.
        """
        if not isinstance(data, dict):
            raise NotImplementedError(f"Data should be a dict, but got {type(data)}. You should implement `_cvt_to_data_cache` and `_cvt_from_data_cache` method in your subclass")
        return data
    
    def _cvt_from_data_cache(self):
        """
        Brief
        -----
        convert `_data_cache` to data.
        If the data is a list-like object, such as `np.ndarray`, you should use `DataFile_ListLike` instead of `DataFile`.
        Or you should implement `cvt_to_data_cache` and `cvt_from_data_cache` method in your subclass.
        """
        return self._data_cache

    def _save(self):
        """
        Warning
        -----
        Not recommended to use this method directly. You should use `save` method of `Dataset` instance instead.

        Brief
        -----
        Save all data in `_data_cache` to the disk.
        """
        self._data_cache = {k: self._data_cache[k] for k in sorted(self._data_cache.keys()) if self._data_cache[k] is not None}
        data = self._cvt_from_data_cache()
        self._write_func(self.get_path(), data)

    def set_save_mode(self, mode: bool) -> None:
        if mode == O_Mode.IMIDIATE:
            warnings.warn("set save_mode to IMIDIATE will cause frequent IO operation and may slow down the program")
        return super().set_save_mode(mode)
    
    def get_path(self, idx:int = None) -> str:
        """
        Brief
        -----
        Get file path. parameter `idx` is not used in this method.
        """
        return os.path.join(self.root_dir, self.file_path)

class DataFile_ListLike(DataFile[DATA_TYPE], Generic[DATA_TYPE]):
    """
    A variant of DataFile, which is used to store list-like data in a single file.
    """
    def __init__(self, file_path:str,
                read_func:Callable[[str],None],
                write_func:Callable[[str, DATA_TYPE],int],
                cvt_func:Callable[[list[DATA_TYPE]], dict[int, DATA_TYPE]]) -> None:
        """
        Brief
        -----
        A variant of `DataFile`, which is used to store list-like data in a single file.
        a class to read/write data in a single file. 
        All data are stored in one file in a `dict` format, such as a json/xml file.

        It's **highly recommended** to inherit this class and bind the parameters: `read_func`/`write_func`/`cvt_func` of `__init__` method.
        
        parameters
        -----
        file_path: str
            a file path to store data
        read_func: Callable[[str], dict[int, DATA_TYPE]]
            a function to read data from the `file_path`
        write_func: Callable[[str, dict[int, DATA_TYPE]], Any]
            a function to write data to the `file_path`
        cvt_func: Callable[[list[DATA_TYPE]], dict[int, DATA_TYPE]]
            a function to convert list-like data to dict-like data
        """
        super().__init__(file_path, read_func, write_func)
        self.__cvt_func = cvt_func

    def _cvt_to_data_cache(self, data):
        assert isinstance(data, Iterable), "data should be a list"
        _data_cache = {}
        for i, value in enumerate(data):
            _data_cache[i] = value            
        return _data_cache
    
    def _cvt_from_data_cache(self):
        data = self.__cvt_func([x for x in self._data_cache.values()])
        return data

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
    def __init__(self, data:_SingleData) -> None:
        self.__data:_SingleData = data
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
    def data(self) -> _SingleData:
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

    class __Alter():
        def __enter__(self):
            setattr(_RegisterInstance, "_RegisterInstance__allow_alter_name", True)

        def __exit__(self, exc_type, exc_value, traceback):
            if exc_type is not None:
                raise exc_type(exc_value)
            _RegisterInstance.__allow_alter_name = False

    __registry:dict[str, Union["Dataset","DatasetView"]] = {}
    __allow_alter_name = False

    def _register(self, name:str) -> str:
        """
        Brief
        -----
        register datasets

        parameters
        -----
        name:str
            name: name of datasets
        """
        alterable = _RegisterInstance.__allow_alter_name
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
    
    @classmethod
    def allow_alter_name(cls):
        """
        Brief
        -----
        A context manager to allow altering the name of the dataset instance.
        """
        return cls.__Alter()

class _DatasetBase(ABC, _RegisterInstance, metaclass = _DecoratingAfterInit):
    """
    `_DatasetBase` is a base class for `_Dataset` and `DatasetView`
    """

    idx_assertion_func = {
        "read":         [0],
        "remove":       [0],
        "move":         [0, 1],
        "copy":         [0, 1],
        "is_complete":  [0],
        "_is_empty":    [0],
        "has":          [0],
        "get_tags":     [0],
        "set_tags":     [0],
        "add_tags":     [0],
        "remove_tags":  [0],
        "clear_tags":   [0]
    }

    def __init__(self, dataset_name:Optional[str] = None) -> None:
        """
        Parameters
        -----
        dataset_name: str
            the name of the dataset, which is used to register the dataset.
        """
        if dataset_name is None:
            dataset_name = self.__class__.__name__
            with self.allow_alter_name():
                self.__dataset_name = self._register(dataset_name)
        else:
            self.__dataset_name = self._register(dataset_name)

    def __repr__(self):
        return self.__dataset_name

    @classmethod
    def _decorating_after_init(cls) -> None:
        for func_name, idxs in cls.idx_assertion_func.items():
            if func_name in cls.__dict__:
                cls._idx_type_assert(func_name, idxs)
        # cls.read        = cls._idx_type_assert(cls.read)
        # cls.remove      = cls._idx_type_assert(cls.remove)
        # cls.move        = cls._idx_type_assert(cls.move, [0, 1])
        # cls.copy        = cls._idx_type_assert(cls.copy, [0, 1])
        # cls.is_complete = cls._idx_type_assert(cls.is_complete)
        # cls._is_empty   = cls._idx_type_assert(cls._is_empty)
        # cls.has         = cls._idx_type_assert(cls.has)
        # cls.get_tags    = cls._idx_type_assert(cls.get_tags)
        # cls.set_tags    = cls._idx_type_assert(cls.set_tags)
        # cls.add_tags    = cls._idx_type_assert(cls.add_tags)
        # cls.remove_tags = cls._idx_type_assert(cls.remove_tags)
        # cls.clear_tags  = cls._idx_type_assert(cls.clear_tags)

    @classmethod
    def _idx_type_assert(cls, func_name:str, idx_para_pos:int|Iterable[int] = 0):
        
        # which parameter is the index
        if isinstance(idx_para_pos, int):
            idx_para_pos = [idx_para_pos]
        else:   
            assert isinstance(idx_para_pos, Iterable), "idx_para_pos should be an integer or a list of integers"
            assert all([isinstance(idx, int) for idx in idx_para_pos]), "idx_para_pos should be a list of integers"
        
        func = getattr(cls, func_name) # get the function
        if func is None:
            raise NotImplementedError(f"Method {func_name} should be implemented in the subclass")
        
        def wrapper(self, *args, **kwargs):
            assert all([isinstance(args[pos], int) for pos in idx_para_pos]), f"parameters at {idx_para_pos} should be an integer, but got {args}"
            return func(self, *args, **kwargs)

        wrapper.__name__ = func.__name__
        setattr(wrapper, "_idx_type_assert_decorated", True)
        setattr(cls, func_name, wrapper)

    @property
    def dataset_name(self):
        """
        Return 
        -----
        dataset_name: str
            `dataset_name` is unique for each dataset instance. You can use `self._query_registered(dataset_name)` to get the dataset instance by `dataset_name`.
        """
        return self.__dataset_name

    @abstractmethod
    def _get_overview(self) -> dict[int, _OverviewDict]:
        """
        NOTE
        -----
        This method is an abstract method, which should be implemented in the subclass.

        Get the overview of the dataset. The overview should be a dictionary-like instance, 
        which contains the tags and data_exists of each data.
        """
        pass

    @abstractmethod
    def make_continuous(self) -> None:
        """
        make the dataset continuous. For example, if the dataset has data with indices [0, 1, 2, 4, 6],
        after calling this method, the dataset will have data with indices [0, 1, 2, 3, 4], where 4 -> 3, 6 -> 4.
        """
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
        """
        Brief
        -----
        read data by index.

        Parameters
        -----
        idx: int
            the index of the data.
        
        Return
        -----
        values: dict[str, Any]
            the data read from the all single datas.
        
        Note
        -----
        - If the data is not exists, a warning will be raised.
        - You can also use `dataset[idx]` to read data.
        """
        pass

    @abstractmethod
    def remove(self, idx:int) -> None:
        """
        Brief
        -----
        remove data by index.

        Parameters
        -----
        idx: int
            the index of the data.

        Note
        -----
        - If the data is not exists, a warning will be raised.
        """
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
    def is_complete(self, idx:int) -> bool:
        """
        Get whether the data is complete. The data is complete if `idx` dose not exist in some single data but exists in others.
        """
        pass

    @abstractmethod
    def _is_empty(self, idx:int) -> bool:
        """
        Get whether the data is empty. The data is empty if `idx` dose not exist in any single data.
        """
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
        """
        get tags of a data, tags can be used to filter data, see `select`.
        """
        if idx not in self._get_overview():
            warnings.warn(f"Data {idx} not exists, can not get tags")
            return set()
        else:
            return self._get_overview()[idx]["tags"]
    
    def set_tags(self, idx:int, tags:Iterable[str]|str|None) -> None:
        """
        Brief
        -----
        set tags of a data, tags can be used to filter data, see `select`.

        Parameters
        -----
        idx: int
            the index of the data
        tags: Iterable[str]|str|None
            the tags of the data, it can be a list of str, a str or None. If it's None, nothing will be done.
            
        Note
        -----
        - If the data is not exists, a warning will be raised. 
        - All old tags will be covered by the new tags. If you only want to add tags, use `add_tags` instead.
        """
        if idx not in self._get_overview():
            warnings.warn(f"Data {idx} not exists, can not add tags")
            return
        if tags is None:
            return
        tags = self.__check_tags(tags)
        self._get_overview()[idx]["tags"].clear()
        self._get_overview()[idx]["tags"].update(tags)

    def add_tags(self, idx:int, tags:Iterable[str]|str|None) -> None:
        """
        Brief
        -----
        add tags to a data, tags can be used to filter data, see `select`.

        Parameters
        -----
        idx: int
            the index of the data
        tags: Iterable[str]|str|None
            the tags of the data, it can be a list of str, a str or None. If it's None, nothing will be done.

        Note
        -----
        - If the data is not exists, a warning will be raised.  
        - This method only add new tags, if you want to cover all old tags, use `set_tags` instead.
        """
        if idx not in self._get_overview():
            warnings.warn(f"Data {idx} not exists, can not add tags")
            return
        if tags is None:
            return
        tags = self.__check_tags(tags)
        self._get_overview()[idx]["tags"].update(tags)

    def remove_tags(self, idx:int, tags:Iterable[str]|str) -> None:
        """
        Brief
        -----
        remove tags from a data.

        Parameters
        -----
        idx: int
            the index of the data
        tags: Iterable[str]|str
            the tags to be removed, it can be a list of str or a str.

        Note
        -----
        - If the data is not exists, a warning will be raised.
        - If the tag is not in the data, nothing will be done.
        """
        if idx not in self._get_overview():
            warnings.warn(f"Data {idx} not exists, can not remove tags")
            return
        tags = self.__check_tags(tags)
        for tag in tags:
            if tag in self._get_overview()[idx]["tags"]:
                self._get_overview()[idx]["tags"].remove(tag)
    
    def clear_tags(self, idx:int) -> None:
        """
        Brief
        -----
        clear all tags of a data.

        Parameters
        -----
        idx: int
            the index of the data
        """
        if idx not in self._get_overview():
            warnings.warn(f"Data {idx} not exists, can not clear tags")
            return
        self._get_overview()[idx]["tags"].clear()
    # endregion

    # region: dataset mapping
    def _select(self, include:Optional[Iterable[_T]|_T], exclude:Optional[Iterable[_T]|_T], _type:type[_T]):
        """
        Brief
        -----
        return a decorated function to filter data by include/exclude. It's used in `select` method.
        """
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
        """
        Brief
        -----
        select data by tags and indices. return the selected indices.

        Parameters
        -----
        include_tags: Iterable[str]|str|None
            the tags to be included. If it's None, means all tags will be included.
        exclude_tags: Iterable[str]|str|None
            the tags to be excluded. If it's None, means no tags will be excluded. 
            `exclude_tags` has a higher priority than `include_tags`.
        include_indices: Iterable[int]|int|None
            the indices to be included. If it's None, means all indices will be included.
        exclude_indices: Iterable[int]|int|None
            the indices to be excluded. If it's None, means no indices will be excluded.

        Return
        -----
        selected_indices: list[int]
            the selected indices.
        """
        tags_filter    = self._select(include_tags,    exclude_tags,    str)
        indices_filter = self._select(include_indices, exclude_indices, int)
        
        selected_indices = []
        for idx in self.keys():
            tags = self.get_tags(idx)
            if tags_filter(tags) and indices_filter(idx):
                selected_indices.append(idx)

        return selected_indices

    def gen_subset(self, view_name:Optional[str] = None, 
                   selected_indices:Optional[Iterable[int]] = None) -> "DatasetView":
        """
        Brief
        -----
        generate a subset of the dataset.

        Parameters
        -----
        view_name: str
            the name of the view. If it's None, a unique name will be generated.
        selected_indices: Iterable[int]
            the indices of the data to be included in the view. If it's None, all data will be included.

        Return
        -----
        view: DatasetView
            the generated view.
        """
        view = DatasetView(view_name)
        view.add_dataset(self, selected_indices)

        return view
    # endregion

class DatasetCfg(TypedDict):
    class SingleDataCfg(TypedDict):
        classname:str
        parameters:str|tuple[str, tuple[int, str]]
        read_required:bool
        write_required:bool
    dataset_name:str
    classname:str
    directory:str
    overview:dict[int, _OverviewDict]
    singledatas:dict[str, SingleDataCfg]

class Dataset(_DatasetBase):
    """
    `_Dataset` is a base class for dataset
    """

    idx_assertion_func = _DatasetBase.idx_assertion_func.copy()
    idx_assertion_func.update({
        "write":        [0]
    })

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

    @classmethod
    def _decorating_after_init(cls) -> None:
        super()._decorating_after_init()
        cls.write   = cls.__O_operations_decorator(cls.write)
        cls.remove  = cls.__O_operations_decorator(cls.remove)
        cls.move    = cls.__O_operations_decorator(cls.move)
        cls.copy    = cls.__O_operations_decorator(cls.copy)

    def _get_overview(self) -> dict[int, _OverviewDict]:
        """
        get the overview of the dataset

        Warning
        -----
        Be careful with the returned overview. It's **highly not recommanded** to modify it.
        """
        return self.__overview

    def get_cfg_path(self) -> str:
        """
        Get default cfg path. It's used to save the configuration of the dataset.
        """
        return os.path.join(self.directory, f"{self.dataset_name}_cfg.json")

    @classmethod
    def from_cfg(cls, path):
        """
        Initialize a dataset from a cfg file.

        NOTE
        -----
        In the configuration file, the key `singledatas` is a dictionary, which contains the name of the single data and the configuration of the single data.
        In each single data configuration:
        - `"classname"` is the class name of the single data, which is used to query the class of the single data.
        - `"parameters"` is `file_path` for `DataFile`. And for `DataCluster`, it is the parameters for initializing the `default_filepath_generator`.
        - `"read_required"` is a bool value, which indicates whether the single data is required to read.
        - `"write_required"` is a bool value, which indicates whether the single data is required to be writen.
        """
        cfg:DatasetCfg = JsonIO.load_json(path)
        with _RegisterInstance.allow_alter_name():
            dataset = cls(cfg["directory"], cfg["dataset_name"])
        dataset.__overview = cfg["overview"]

        for name, mng_cfg in cfg["singledatas"].items():
            _class:type[_SingleData] = _SingleData._query_SingleData_Subclass(mng_cfg["classname"])
            if _class is None:
                continue
            if issubclass(_class, DataFile):
                _data = _class(file_path=mng_cfg["parameters"])
            elif issubclass(_class, DataCluster):
                if mng_cfg["parameters"] is not None:
                    path_generator = default_filepath_generator(mng_cfg["parameters"][0], mng_cfg["parameters"][1])
                else:
                    path_generator = None
                _data = _class(path_generator=path_generator)
            dataset.add_single_data(name, _data)
            dataset._set_single_datas_requirement(name=name, mode="r", required=mng_cfg["read_required"])
            dataset._set_single_datas_requirement(name=name, mode="w", required=mng_cfg["write_required"])
        return dataset

    def save_cfg(self, cfg_file:Optional[str] = None) -> None:
        """
        Brief
        -----
        save the configuration of the dataset.

        Parameters
        -----
        cfg_file: str
            the path to save the configuration. If it's None, the default path will be used.

        Warning
        -----
        the `read_func` and `write_func` of the `SingleData` will not be saved in the configuration. Here are 2 ways to settle it:
        - Implement a specific subclass of `DataFile`/`DataCluster` and bind the `read_func`/`write_func` and other probable parameters of the `__init__` method,
          making it only require the `file_path`(for `DataFile`) or `path_generator`(for `DataCluster`) to initialize.
        - Manually set the `read_func`/`write_func` of the `SingleData` after loading the configuration.

        For `DataCluster`, only when the `path_generator` is an instance of `default_filepath_generator`, 
        the parameters of the `path_generator` will be saved in the configuration.
        """
        if cfg_file is None:
            cfg_file = self.get_cfg_path()
        cfg:DatasetCfg = {}
        cfg["directory"] = self.__directory
        cfg["dataset_name"] = self.dataset_name
        cfg["classname"] = self.__class__.__name__

        sorted_overview = dict(sorted(self.__overview.items()))
        self.__overview = sorted_overview
        cfg["overview"] = self.__overview

        cfg["singledatas"] = {}
        for name, mng in self.__single_datas.items():
            cfg["singledatas"][name] = {
                "classname": mng.data.__class__.__name__,
                "parameters": None,
                "read_required": mng.read_required,
                "write_required": mng.write_required
            }
            if isinstance(mng.data, DataFile):
                cfg["singledatas"][name]["parameters"] = mng.data.file_path
            elif isinstance(mng.data, DataCluster) and isinstance(mng.data.path_generator, default_filepath_generator):
                cfg["singledatas"][name]["parameters"] = (mng.data.path_generator.format_str, mng.data.path_generator.rjust_params)

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

    @classmethod
    def __O_operations_decorator(cls, func:Callable):
        """
        Brief
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
            if self.__class__ != cls:
                return func(self, *args, **kwargs)

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
                    if not(isinstance(tags, bool) and not tags):
                        tags = self.__overview[args[0]]["tags"] # copy with tags, once not specified False
                    self.set_tags(args[1], tags)
                return rlt
        return wrapper

    def scan(self, max_size=100000, verbose=False, uncontinuous=False, rescan=False) -> None:
        """
        Brief
        -----
        scan the dataset to check if the data exists.

        Parameters
        -----
        max_size: int
            the maximum size of the dataset to be scanned.
        verbose: bool
            whether to show the scanning process.
        uncontinuous: bool
            whether to continue to scan when the data is not continuous. For example, if the dataset has data with indices [0, 1, 2, 4], 
            when `uncontinuous` is False, the scanning process will stop at 2.
        rescan: bool
            whether to rescan the dataset. If it's True, the previous overview will be cleared.
        """
        if rescan:
            self.__overview.clear()
        
        def check_DataCluster(idx):
            path = d.get_path(idx)
            rlt = True
            if not os.path.exists(path):
                rlt = False
                if verbose:
                    print(f"File {path} not exists in {k}")
            return rlt

        def check_DataFile(idx):
            if idx not in d.keys:
                rlt = False
                if verbose:
                    print(f"Data {idx} not exists in {k}")
            return rlt

        default_data_exists_dict = {name: True for name in self._get_required_single_datas('none', 'keys')}

        process = tqdm(range(max_size), desc=f"Scanning", leave=False) if verbose else range(max_size)
        for idx in process:
            self.__overview[idx] = _overview_dict_builder()
            self.__overview[idx]["data_exists"] = default_data_exists_dict.copy()
            for k, d in self._get_required_single_datas('none'):
                if isinstance(d, DataCluster):
                    rlt = check_DataCluster(idx)
                else:
                    rlt = check_DataFile(idx)
                self.__overview[idx]["data_exists"][k] = rlt
            if self._is_empty(idx):
                self.__overview.pop(idx, None)
                if not uncontinuous:
                    break

    def delete_incomplete(self) -> None:
        """
        Incomplete data is the data that has no data for some single data. This method will
        delete incomplete data in the dataset. 
        """
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
    def add_single_data(self, name:str, data:_SingleData) -> None:
        """
        Brief
        -----
        add a single data to the dataset.

        Parameters
        -----
        name: str
            the name of the single data.
        data: _SingleData
            the single data to be added.
        """
        self.__single_datas[name] = _SingleDataMng(data)
        data._set_dataset(self)

    def remove_single_data(self, name:str) -> None:
        """
        Brief
        -----
        remove a single data from the dataset.

        Parameters
        -----
        name: str
            the name of the single data.
        """
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
        """
        get all single data names in the dataset.
        """
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

    def _get_single_data(self, name:str) -> _SingleData:
        """
        get the single data by name.
        """
        if name not in self.__single_datas:
            return None
        else:
            return self.__single_datas[name].data
    
    def _get_required_single_datas(self, mode:Literal['r', 'w', 'both', 'none'] = 'r', iter:Literal['items', 'values', 'key'] = "items") -> Generator[Tuple[str, _SingleData], None, None]:
        """
        Brief
        -----
        Get the single data that is required to read or write. See `_set_single_datas_requirement` for more details.

        Parameters
        -----
        mode: Literal['r', 'w', 'both', 'none']
            the mode to get the single data. 'r' means read, 'w' means write, 'both' means both read and write, 'none' means all.
        iter: Literal['items', 'values', 'key']
            the way to iterate the single data. The iter should be 'items' or 'values' or 'keys'
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
    
    def _set_single_datas_requirement(self, name:Optional[str] = None, mode:Literal['r', 'w'] = 'r', required:Optional[bool] = None, _dict:Optional[dict[str, bool]] = None) -> None:
        """
        Brief
        -----
        set the requirement of the single data. 
        If the single data is required to read, it will be read when reading the dataset. Otherwise, it will be ignored.
        If the single data is required to write, the value of it is required when writing data to the dataset. Otherwise, it will be ignored.

        * All single data are required to read and write by default.

        Parameters
        -----
        name: str
            the name of the single data.
        mode: Literal['r', 'w']
            the mode to set the requirement. 'r' means read, 'w' means write.
        required: bool
            whether the single data is required. If it's None, nothing will be done.
        _dict: dict[str, bool]
            a dict to set the requirement of multiple single data. If it's not None, `name` and `required` will be ignored.
        
        Examples
        -----
        ```python

        # dataset = Dataset()
        # ...
        print(dataset.get_single_data_names()) # ('data1', 'data2', 'data3')
        print(dataset[0]) # {'data1': 1, 'data2': 2, 'data3': 3}
        dataset._set_single_datas_requirement('data1', 'r', False)
        print(dataset[0]) # {'data2': 2, 'data3': 3}
        dataset[0] = {'data3': 3}
        # _RequirementError: Missing required data {'data2'}
        dataset._set_single_datas_requirement('data2', 'w', False)
        dataset[0] = {'data3': 3} # OK
        ```
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

    def is_ready(self, verbose = False) -> bool:
        """
        Get whether the dataset is ready. The dataset is ready if all single data in the dataset is ready.
        """
        rlt = True
        for name, mng in self.__single_datas.items():
            if mng.read_required and mng.write_required:
                if not mng.data.is_ready():
                    rlt = False
                    if verbose:
                        print(f"{name} in {self.dataset_name} is not ready")
        return rlt
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
        """
        Brief
        -----
        write data by index.

        Parameters
        -----
        idx: int
            the index of the data.
        values: dict[str, Any]
            the data to be written.
        tags: list[str]|str|None
            the tags of the data. If it's None, nothing will be done.
        
        Note
        -----
        - You can also use `dataset[idx] = {"key": value}` to write data.
        """
        return self._raw_write(idx, values)

    def remove(self, idx:int) -> None:
        for name, data in self._get_required_single_datas('w'):
            data._remove(idx)

    def move(self, old_idx:int, new_idx:int, tags:Optional[list[str]|str|bool] = None) -> None:
        """
        Brief
        -----
        move data from `old_idx` to `new_idx`.

        Parameters
        -----
        old_idx: int
            the index of the data to be moved.
        new_idx: int
            the index to move the data to.
        tags: list[str]|str|bool
            the tags of the data. If it's `None` or `True`, the tags of the new data will be the same as the old data. 
            If it's `False`, the tags of the new data will be empty.
            If it's a `list[str]` or a `str`, the tags of the new data will be the specified tags.
        """
        for name, data in self._get_required_single_datas('w'):
            data._move(old_idx, new_idx)

    def copy(self, old_idx:int, new_idx:int, tags:Optional[list[str]|str|bool] = None) -> None:
        """
        Brief
        -----
        move data from `old_idx` to `new_idx`.

        Parameters
        -----
        old_idx: int
            the index of the data to be moved.
        new_idx: int
            the index to move the data to.
        tags: list[str]|str|bool
            the tags of the data. If it's `None` or `True`, the tags of the new data will be the same as the old data. 
            If it's `False`, the tags of the new data will be empty.
            If it's a `list[str]` or a `str`, the tags of the new data will be the specified tags.
        """
        for name, data in self._get_required_single_datas('w'):
            data._copy(old_idx, new_idx)

    def append(self, values:dict[str, Any], tags:Optional[list[str]|str] = None) -> None:
        """
        Brief
        -----
        `write` data to the end of the dataset.

        Parameters
        -----
        values: dict[str, Any]
            the data to be appended.
        tags: list[str]|str|None
            the tags of the data. If it's None, nothing will be done.
        """
        idx = self._get_empty_idx()
        self.write(idx, values, tags = tags)

    def save(self, force = True) -> None:
        """
        save all single datas that are required to be writen.
        """
        for name, data in self._get_required_single_datas('w'):
            if not force and data.get_save_mode() == O_Mode.MANUAL:
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
    def start_writing(self, mode:Literal["a", "w"]|Literal[2,4]|Literal[DatasetIOMode.MODE_APPEND, DatasetIOMode.MODE_WRITE] = 2) -> None:
        """
        Brief
        -----
        start writing data to the dataset.

        Examples
        -----
        ```python

        dataset = Dataset()
        # ...
        with dataset.start_writing("a"):
            dataset[0] = {"key": value}
        with dataset.start_writing("a"):
            dataset[0] = {"key": value}
            # Error: You are trying to append data to the dataset, but the data already exists.
        with dataset.start_writing("w"):
            dataset[0] = {"key": value} # OK
        ```
        """
        if isinstance(mode, str):
            assert mode in ('append', 'write', 'a', 'w'), "mode should be 'append' or 'write' or 'a' or 'w'"
            mode = DatasetIOMode.MODE_APPEND if mode in ('append', 'a') else DatasetIOMode.MODE_WRITE
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

# class Dataset(_Dataset):
#     pass

class _DatasetViewItem(TypedDict):
    source_name:str
    source_idx:int

class DatasetViewCfg(TypedDict):
    dataset_name:str
    classname:str
    idx_map: dict[int, _DatasetViewItem]

class _IdxMap(dict[int, _OverviewDict]):
    """
    act like overview
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

    idx_assertion_func = _DatasetBase.idx_assertion_func.copy()
    idx_assertion_func.update({
        "add":              [0],
        "_get_ref_chain":   [0],
        "_query_source":    [0],
        "print_ref_chain":  [0]
    })

    def __init__(self, dataset_name:str) -> None:
        """
        Parameters
        -----
        dataset_name: str
            the name of the dataset. It is not required to be unique.
        """
        with _RegisterInstance.allow_alter_name():
            super().__init__(dataset_name)
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
    def from_cfg(cls, path, dataset_name:Optional[str] = None):
        """
        Brief
        -----
        initialize the DatasetView from the configuration file.

        Parameters
        -----
        path: str
            the path of the configuration file.
        dataset_name: str
            the name of the dataset. If it's None, the name in the configuration file will be used.

        Examples
        -----
        configuraion file should be like:
        ```json
        {
            "dataset_name": "dataset1",
            "classname": "DatasetView",
            "idx_map": {
                0: {"source_name": "dataset2", "source_idx": 0},
                1: {"source_name": "dataset2", "source_idx": 1},
                2: {"source_name": "dataset3", "source_idx": 0}
            }
        }
        ```
        """
        cfg:DatasetViewCfg = JsonIO.load_json(path)
        if dataset_name is None:
            dataset_name = cfg["dataset_name"] 
        assert isinstance(dataset_name, str), "dataset_name should be a str"
        datasetview = cls(dataset_name)
        datasetview.__idx_map = _IdxMap(cfg["idx_map"])
        return datasetview

    def save_cfg(self, cfg_file:str) -> None:
        """
        Brief
        -----
        save the configuration of the DatasetView as a json file. See Examples in `from_cfg` for more details.

        Parameters
        -----
        cfg_file: str
            the path of the configuration file.
        """
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
    def add_dataset(self,   dataset:Union[Dataset, "DatasetView"], 
                            source_indices:Optional[Iterable[int]] = None,
                            target_indices:Optional[Iterable[int]] = None,) -> None:
        if source_indices is None:
            source_indices = dataset.keys()
        else:
            assert isinstance(source_indices, Iterable), "indices should be an Iterable"
            assert all([isinstance(idx, int) for idx in source_indices]), "indices should be int"
            if target_indices is not None:
                assert isinstance(target_indices, Iterable), "indices should be an Iterable"
                assert len(source_indices) == len(target_indices), "source_indices and target_indices should have the same length"
                assert all([isinstance(idx, int) for idx in target_indices]), "indices should be int"
        
        length = len(source_indices)
        if target_indices is None:
            start = self._get_empty_idx()
            target_indices:Iterable[int] = range(start, start + length)
        else:
            pass
        for idx, source_idx in zip(target_indices, source_indices):
            self.__idx_map._set_inner(idx, {"source_name": dataset.dataset_name, "source_idx": source_idx})

    def add(self, idx:int, source_name:Dataset|str, source_idx:int) -> None:
        # TODO
        assert isinstance(idx, int), "idx should be an int"
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
    
    def _get_ref_chain(self, idx:int):
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

# class DatasetView(_DatasetView):
#     """
#     Read-only Dataset.
#     """
#     pass