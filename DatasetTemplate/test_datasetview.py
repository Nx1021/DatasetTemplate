from _base import Dataset, DataCluster, DataFile, filename_generator_builder, DatasetView
from utils import JsonIO, write_text, read_text
import numpy as np
import cv2

def create_label_cluster():
    return DataCluster("label", read_text, write_text,  filename_generator_builder("pre_{}_apd.txt"))

def create_file_data():
    return DataFile("file.json", JsonIO.load_json, JsonIO.dump_json)

ds1 = Dataset("datasettest/ds1", dataset_name="ds1")
ds1.add_single_data("label", create_label_cluster())
ds1.add_single_data("file",  create_file_data())

ds2 = Dataset("datasettest/ds2", dataset_name="ds2")
ds2.add_single_data("label_2", create_label_cluster())


with ds1.start_writing(mode = 4):
    ds1.clear()
    for i in range(10):
        ds1.append({"label": f"ds1_{i}", "file": np.array([2,])}, tags=[f"tag_{i%2}", f"tag_{i%3}"])

with ds2.start_writing(mode = 4):
    ds2.clear()
    for i in range(10):
        ds2.append({"label_2": f"ds2_{i}"}, tags=[f"tag_{i%2}", f"tag_{i%3}"])

ds_view_1 = DatasetView("view_1")
ds_view_1.add_dataset(ds1, [0,1,2,3,4])
ds_view_1.add_dataset(ds2, [5,6,7,8,9])

ds_view_2 = DatasetView("view_2")
ds_view_2.add_dataset(ds1, [5,6,7,8,9])
ds_view_2.add_dataset(ds2, [0,1,2,3,4])

ds_view_3 = DatasetView("view_3")
subset = ds1.gen_subset("tag_1", ds1.select(include_tags="tag_1", exclude_tags="tag_2", exclude_indices=[0,1,2]))
subset._get_overview()[0]
subset._get_overview()._get_inner(0)
ds_view_3.add_dataset(subset)

ds_view_top = DatasetView("view")
ds_view_top.add_dataset(ds_view_1)
ds_view_top.add_dataset(ds_view_2)
ds_view_top.add_dataset(ds_view_3)

# test_1
for k, v in ds_view_top.items():
    print(k, v)
    print(ds_view_top.get_tags(k))
    ds_view_top.print_ref_chain(k)
    print()

# test_2
ds_view_top.save_cfg("ds_view_top.json")
new_ds_view_top = DatasetView.from_cfg("ds_view_top.json")

# test_3
selected = ds_view_top.select(
    include_indices=list(range(20)), exclude_indices=[0, 2],
    include_tags="tag_1",            exclude_tags="tag_2",
    include_shallow_dataset_names="view_1", exclude_shallow_dataset_names="view_3",
    include_source_dataset_names ="ds1",    exclude_source_dataset_names="ds2")

print()
for idx in selected:
    print(ds_view_top[idx])
    print(ds_view_top.get_tags(idx))
    ds_view_top.print_ref_chain(idx)
# 


