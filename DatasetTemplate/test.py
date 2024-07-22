# from _base import Dataset, DataCluster, DataFile, filename_generator_builder
# from utils import JsonIO
# import numpy as np
# import cv2

# color = DataCluster("color", cv2.imread, cv2.imwrite, filename_generator_builder("pre_{}_apd.png"))
# label = DataCluster("", np.loadtxt, np.savetxt,  filename_generator_builder("label_2/pre_{}_apd.txt"))
# file  = DataFile("file.json", JsonIO.load_json, JsonIO.dump_json)

# ds = Dataset("datasettest")
# ds.add_single_data("color", color)
# ds.add_single_data("label", label)
# ds.add_single_data("file",  file)

# ds.scan(20, verbose=True, uncontinuous=True, rescan=True)
# ds.save_cfg()



# with ds.start_writing(mode = 4):
#     ds.clear()

# with ds.start_writing(mode = 4):
#     for i in range(10):
#         ds.append({"color": np.zeros((480, 640), np.uint8), "label": np.ones(4), "file": np.array([2,])}, tags="tag")
#     ds.move(0, 11)
#     ds.copy(0, 12)
#     ds.remove(10)


import unittest
import os
from unittest.mock import MagicMock, patch
from _base import Dataset, _SingleDataMng, JsonIO, DatasetIOMode, _RequirementError, _IdxNotFoundError, SingleData, \
DataCluster, DataFile, filename_generator_builder

import numpy as np
import cv2
from utils import JsonIO, write_text, read_text

class TestDataset(unittest.TestCase):

    def setUp(self):
        # Setup the directory and mock dependencies
        self.directory = "test_dataset"
        os.makedirs(self.directory, exist_ok=True)
        self.cfg_file = os.path.join(self.directory, "Dataset_cfg.json")
        
        self.color = DataCluster("color", read_text, write_text, filename_generator_builder("pre_{}_apd.txt"))
        self.file  = DataFile("file.json", JsonIO.load_json, JsonIO.dump_json)

        # # Mocking JsonIO
        self.json_data = {}
        # JsonIO.load_json = MagicMock(return_value=self.json_data)
        # JsonIO.dump_json = MagicMock()

        # # Mocking _SingleDataMng and SingleData
        # self.single_data_mock = MagicMock(spec=SingleData)
        # self.single_data_mng_mock = MagicMock(spec=_SingleDataMng)
        # self.single_data_mng_mock.data = self.single_data_mock

        # Initialize the Dataset
        self.dataset = Dataset(self.directory)

    def tearDown(self):
        # Clean up the directory after tests
        if os.path.exists(self.directory):
            for root, dirs, files in os.walk(self.directory, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.directory)

    def test_initialization(self):
        # Test initialization
        self.assertEqual(self.dataset.directory, self.directory)
        self.assertTrue(self.dataset.dataset_name.startswith("Dataset"))
        self.assertFalse(self.dataset.is_writing)

    def test_add_single_data(self):
        # Test adding single data
        self.dataset.add_single_data("test_data", self.color)
        self.assertIn("test_data", self.dataset.get_single_data_names())

    def test_remove_single_data(self):
        # Test removing single data
        self.dataset.add_single_data("test_data", self.color)
        self.dataset.remove_single_data("test_data")
        self.assertNotIn("test_data", self.dataset.get_single_data_names())

    def test_scan(self):
        # Test scan method
        self.dataset.add_single_data("test_data", self.color)
        self.dataset.add_single_data("test_data2", self.file)
        not_exists = self.dataset.scan()
        self.assertIn("test_data", not_exists)

    def test_save_cfg(self):
        # Test save configuration
        self.dataset.save_cfg()

    def test_write_read_operations(self):
        # Test write and read operations
        self.dataset.add_single_data("test_data", self.color)
        self.dataset.add_single_data("test_data2", self.file)
        self.dataset.set_single_datas_requirement(mode='w', name='test_data', required=True)
        self.dataset.set_single_datas_requirement(mode='r', name='test_data', required=True)

        values = {"test_data": "test_value", "test_data2": np.array([1])}
        with self.dataset.start_writing(DatasetIOMode.MODE_WRITE):
            self.dataset.write(0, values)
        read_values = self.dataset.read(0)
        self.assertEqual(read_values, values)

    def test_remove(self):
        # Test remove operation
        self.dataset.add_single_data("test_data", self.color)
        self.dataset.add_single_data("test_data2", self.file)
        values = {"test_data": "test_value", "test_data2": np.array([1])}
        with self.dataset.start_writing(DatasetIOMode.MODE_WRITE):
            self.dataset.write(0, values)
            self.dataset.remove(0)
        self.assertFalse(self.dataset.has(0))

    def test_move(self):
        # Test move operation
        self.dataset.add_single_data("test_data", self.color)
        self.dataset.add_single_data("test_data2", self.file)
        values = {"test_data": "test_value", "test_data2": np.array([1])}
        with self.dataset.start_writing(DatasetIOMode.MODE_WRITE):
            self.dataset.write(0, values)
            self.dataset.move(0, 1)
        self.assertFalse(self.dataset.has(0))
        self.assertTrue(self.dataset.has(1))

    def test_copy(self):
        # Test copy operation
        self.dataset.add_single_data("test_data", self.color)
        self.dataset.add_single_data("test_data2", self.file)
        values = {"test_data": "test_value", "test_data2": np.array([1])}
        with self.dataset.start_writing(DatasetIOMode.MODE_WRITE):
            self.dataset.write(0, values)
            self.dataset.copy(0, 1)
        self.assertTrue(self.dataset.has(0))
        self.assertTrue(self.dataset.has(1))

    def test_append(self):
        # Test append operation
        self.dataset.add_single_data("test_data", self.color)
        self.dataset.add_single_data("test_data2", self.file)
        values = {"test_data": "test_value", "test_data2": np.array([1])}
        with self.dataset.start_writing(DatasetIOMode.MODE_WRITE):
            self.dataset.append(values)
        self.dataset.append(values)
        self.assertTrue(self.dataset.has(0))

    def test_clear(self):
        # Test clear operation
        self.dataset.add_single_data("test_data", self.color)
        self.dataset.add_single_data("test_data2", self.file)
        values = {"test_data": "test_value", "test_data2": np.array([1])}
        with self.dataset.start_writing(DatasetIOMode.MODE_WRITE):
            self.dataset.write(0, values)
            self.dataset.clear()
        self.assertFalse(self.dataset.has(0))

    def test_dictlike_operations(self):
        # Test dict-like operations
        self.dataset.add_single_data("test_data", self.color)
        self.dataset.add_single_data("test_data2", self.file)
        values = {"test_data": "test_value", "test_data2": np.array([1])}
        with self.dataset.start_writing(DatasetIOMode.MODE_WRITE):
            self.dataset[0] = values
        self.assertEqual(self.dataset[0], values)
        self.assertIn(0, self.dataset)
        with self.dataset.start_writing(DatasetIOMode.MODE_WRITE):
            self.dataset.update({1: values})
        self.assertEqual(self.dataset[1], values)
        self.assertEqual(len(self.dataset), 2)

    def test_IO_control(self):
        # Test IO control methods
        with self.dataset.start_writing(DatasetIOMode.MODE_WRITE):
            self.assertEqual(self.dataset.IO_Mode, DatasetIOMode.MODE_WRITE)
        self.assertEqual(self.dataset.IO_Mode, DatasetIOMode.MODE_READ)

if __name__ == '__main__':
    unittest.main()
