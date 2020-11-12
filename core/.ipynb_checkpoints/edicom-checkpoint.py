import pydicom, re

from .ecf import *

from .eclogging import load_logger
logger = load_logger()


def construct_wid_element(dicom_path):
    """
    
    Parameter
    ---------
    total : for multiprocessing
    
    """
    
    pid = re.search(r"([^\n/\\]+).dcm", dicom_path).group(1) # patient id
    pd_data = pydicom.read_file(dicom_path)
    pd_image_array = pd_data.pixel_array
    
    return pid, pd_image_array