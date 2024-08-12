from ._base import DataFile, DataCluster, Dataset, DatasetView, DataFile_ListLike
from .utils import read_text, write_text, JsonIO, load_xml, dump_xml
from typing import List, Tuple, Dict, Any, Union, Callable, Optional, TypedDict, Literal

class NdarrayNpyCluster(DataCluster):
    def __init__(self, path_generator:Callable[[int], str]) -> None:
        from numpy import load, save
        super().__init__(path_generator, load, save)

class NdarrayTxtCluster(DataCluster):
    def __init__(self, path_generator:Callable[[int], str]) -> None:
        from numpy import loadtxt, savetxt
        super().__init__(path_generator, loadtxt, savetxt)

class NdarrayNpzCluster(DataCluster):
    def __init__(self, path_generator:Callable[[int], str]) -> None:
        from numpy import load, savez
        super().__init__(path_generator, load, savez)

class NdarrayNpzDictCluster(DataCluster):
    def __init__(self, path_generator:Callable[[int], str]) -> None:
        from numpy import load, savez
        super().__init__(path_generator, load, savez)

class CvGrayImageCluster(DataCluster):
    def __init__(self, path_generator:Callable[[int], str]) -> None:
        import cv2
        super().__init__(path_generator, lambda x: cv2.imread(x, cv2.IMREAD_GRAYSCALE), cv2.imwrite)

class CvColorImageCluster(DataCluster):
    def __init__(self, path_generator:Callable[[int], str]) -> None:
        import cv2
        super().__init__(path_generator, cv2.imread, cv2.imwrite)

class CvColorImageWithAlphaCluster(DataCluster):
    def __init__(self, path_generator:Callable[[int], str]) -> None:
        import cv2
        super().__init__(path_generator, lambda x: cv2.imread(x, cv2.IMREAD_UNCHANGED), cv2.imwrite)

class CvDepthImageCluster(DataCluster):
    def __init__(self, path_generator:Callable[[int], str]) -> None:
        import cv2
        super().__init__(path_generator, lambda x: cv2.imread(x, cv2.IMREAD_ANYDEPTH), lambda x, y: cv2.imwrite(x, y.astype('uint16')))

class TxtCluster(DataCluster):
    def __init__(self, path_generator:Callable[[int], str]) -> None:
        super().__init__(path_generator, read_text, write_text)

class MatCluster(DataCluster):
    def __init__(self, path_generator:Callable[[int], str]) -> None:
        from scipy.io import loadmat, savemat
        super().__init__(path_generator, loadmat, savemat)

class JsonCluster(DataCluster):
    def __init__(self, path_generator:Callable[[int], str]) -> None:
        super().__init__(path_generator, JsonIO.load_json, JsonIO.dump_json)

class XmlCluster(DataCluster):
    def __init__(self, path_generator:Callable[[int], str]) -> None:
        super().__init__(path_generator, load_xml, dump_xml)


class JsonFile(DataFile):
    def __init__(self, file_path:str) -> None:
        super().__init__(file_path, JsonIO.load_json, JsonIO.dump_json)

class XmlFile(DataFile):
    def __init__(self, file_path:str) -> None:
        super().__init__(file_path, load_xml, dump_xml)

class NdarrayNpyFile(DataFile_ListLike):
    def __init__(self, file_path:str) -> None:
        from numpy import load, save, array
        super().__init__(file_path, load, save, array)