import os
import zipfile
from pathlib import Path
import numpy as np

def _parse_data(data: bytes, drop_header=True):
    res = np.array([x.split(",") for x in data.decode("utf-8").split("\r\n")[1:] if len(x) > 0])
    if not drop_header:
        header = (data.decode("utf-8").split("\r\n")[0]).split(",")
        return np.insert(res, 0,header, 0)
    return res


def _read_zip_file(path):
    with zipfile.ZipFile(path, "r") as z:
        if '.csv' in path:
            return _parse_data(z.read(z.filelist[0]), False)
        raise ValueError("Only CSV files are supported")
    

def load_data(data_path):
    """ path inside the data folder """
    data_dir = Path(os.path.dirname(__file__)).parent / 'datasets'
    file_path = os.path.join(data_dir, data_path)
    if file_path.endswith(".zip"):
        read = _read_zip_file(file_path)
    return read