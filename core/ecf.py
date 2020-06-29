### Reference ###
# - all stdout and stderr to log file.
#         https://stackoverflow.com/questions/616645/how-to-duplicate-sys-stdout-to-a-log-file
#         https://stackoverflow.com/questions/1956142/how-to-redirect-stderr-in-python/1956228#1956228
#         https://stackoverflow.com/questions/19425736/how-to-redirect-stdout-and-stderr-to-logger-in-python
#         https://stackoverflow.com/questions/47325506/making-python-loggers-log-all-stdout-and-stderr-messages


import glob, re, os, shutil, sys
from .eclogging import load_logger
import time # For TimeChecker class.
import numpy as np
import math
from IPython.display import Audio, display

logger = load_logger()
module_name = re.search("(^[^.\n]+).*$", __name__).group(1)
module_root_path = re.search(f"(^.+{module_name})/[^\n/]+", __file__).group(1)
sound_path = module_root_path + "/misc/sounds"

### 
sys_excepthook = sys.excepthook # Backup Original sys.excepthook

def cus_excepthook(logger):
    """
    Custom excepthook function to log exception information.
    logger will log exception information automatically.
    This doesn't work in ipython(including jupyter). Use `get_ipython().set_custom_execs((Exception,), your_exception_function)` instead in ipython environment.
    
    Parameters
    ----------
    logger: a logger object.
    
    Examples
    --------
    import sys
    sys.excepthook = cus_exception(logger)
    
    """
    def _excepthook(etype, value, tb):
        sys_excepthook(etype, value, tb)
        
        logger.debug("Got exception.\n", exc_info = (etype, value, tb))
    
    return _excepthook    


###
def exception_sound(logger, audio_path = sound_path + "/Nope.m4a"):
    
    def _exception_sound(self, etype, value, tb, tb_offset=None):
        """
        https://stackoverflow.com/questions/40722417/play-sound-when-jupyter-notebook-cell-fails
        """
        self.showtraceback((etype, value, tb), tb_offset=tb_offset)
        logger.debug("Got exception.\n", exc_info= True)
        display(Audio(audio_path, autoplay=True))
    
    return _exception_sound


###
def sound_alert(audio_path = sound_path + "/sc2-psh-rc.mp3", **kwargs):
    display(Audio(audio_path, autoplay = True))


###
def print_progressbar(total, i, details = ""):
    """
    total : total iteration number.
    i : iteration count, starting from 0.
    details : string you want to show next to the progress bar. e.g. current file name.
    """
    import math
    step = 25 / total

    # Print the progress bar
    print("\r" + " " * 150, end="")
    print('\r' + f'Progress: '
        f"[{'=' * int((i+1) * step) + ' ' * (25 - int((i+1) * step))}]"
        f"({math.floor((i+1) * 100 / (total))} %) ({i+1}/{total}) " + details,
        end='')
    if (i+1) == total:
        print("")

    
    
###
def get_savepath(save_path=None):
    """
    Get save path you want but tagged with suffix (_[0-9]+) if a file whose name is what you want already exists.
    
    save_path : desired save path.
    """
    
    if save_path is None: return None
    
    # sepearte save_path into file name and file ext
    fe_s = re.search('(\.[^.]+$)', save_path) ; logger.debug(f"file ext search result: {fe_s}")
    
    if fe_s is None: # there is no file ext.
        file_ext = ''
        file_name = save_path ; logger.debug(f"file name and file ext: {file_name}, {file_ext}")
        
    elif fe_s is not None:
        file_ext = fe_s.group(1)
        file_name = re.search('(^.*)(\..+?$)', save_path).group(1) ; logger.debug(f"file name and file ext: {file_name}, {file_ext}")
        
    # Search files with file name
    gl1 = glob.glob(f"{file_name}*{file_ext}") ; logger.debug(f"gl1: {gl1}")
    re1 = list(filter(lambda x:re.search(f"{file_name}(?:_[0-9]+){file_ext}$", x) is not None, gl1)) ; logger.debug(f"re1: {re1}")
    re2 = re1 + glob.glob(save_path)
        
    if len(re2) == 0:
        logger.info(f"savepath: {save_path}")
        return save_path # There is no file named save_path.
    if (len(re2) == 1) & (len(re1) == 0):
        filepath = file_name + '_2' + file_ext
        logger.info(f"savepath: {filepath}")
        return filepath
    
    
    fi1 = list(map(lambda x:int(re.search(f"_([0-9]+){file_ext}$", x).group(1)), re1)) ; logger.debug(f"fi1: {fi1}")
    
    # Find max number of suffix
    mi = max(fi1) ; logger.debug(f"mi: {mi}")
    
    # concatenate file name and suffix and file ext.
    filepath = file_name + '_' + str(mi+1) + file_ext
    logger.info(f"savepath: {filepath}")
    return filepath


###
def arange(*args):
    template = list(range(*args))
    if len(args) == 3: template = template + [args[-2]]
        
    return template


###
class TimeChecker():
    """
    Made for checking elapsed time.
    """
    
    def __init__(self):
        self.start=time.time()
    
    def set_start(self):
        self.start = time.time()
    
    def set_end(self):
        self.end=time.time()
    
    def set_and_show(self):
        self.end=time.time()
        
        return self.show_elapsed_time()
    
    def show_elapsed_time(self):
        hours, rem = divmod(self.end - self.start, 3600)
        minutes, seconds = divmod(rem, 60)
        rs="{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
        print(rs)        
        return rs
    
    
###
def del_vars(*args, main_global):
    """
    Delete variables.
    
    Parameters
    ----------
    args : variables which you want to remove.
    main_global : global namespace dictionary. i.e. globals()
    
    """
    for i in args:
        try:
            del main_global[i] ; logger.info(f"{i} was deleted successfully.")
        except Exception as e:
            logger.info(f"{e.__class__.__name__} {e}")

            
### copying files keeping the directory structure.

def copy_keeping_structure(srcfile, srcroot_folder, dstroot, copy_function = shutil.copy):
    """
    Parameters
    ----------
    srcfile : file path which you want to copy.
    scrroot : source root folder.
    dstroot : the root path of destination path. The directory structure of the source file is copied also under this path.
    copy_function : A function object which you want to use. e.g. shutil.move. / Default : shutil.copy
    
    Examples
    --------
    Let's suppose that current working directory is /workspace/, you want to copy '/workspace/Brats/data/part1/T1.nii.gz' to '/data/eck/backup/Brats/data/part1/T1.nii.gz'.
    Then,
    
    Use this function like the following:
    
    copy_keeping_structure('Brats/data/part1/T1.nii.gz', 'Brats', '/data/eck/backup/Brats/')
    """
    
    srcfile = os.path.abspath(srcfile)
    srcroot_folder = os.path.abspath(srcroot_folder)
    srcleaf = re.search(f"{srcroot_folder}/(.*)", srcfile).group(1)
    
    dstroot = os.path.abspath(dstroot)
    dstdir =  os.path.join(dstroot, os.path.dirname(srcleaf))
    
    os.makedirs(dstdir, exist_ok = True)

    copy_function(srcfile, dstroot + '/' + srcleaf)
    operation = copy_function.__name__
    logger.info(f"copy_keeping_structure: {operation}({srcfile}, {dstroot + '/' + srcleaf}) was carried out.")
        
#     except Exception as e:
#         logger.info(f"copy keeping structure failed. sourcefile: {srcfile}.\n{e.__class__.__name__}: {e}")


### 
def get_filesize(file):
    """
    Get file size. (GB)
    """
    fs = os.path.getsize(file) 
    print(f"file size: {fs / 1024 ** 3:.3f} GB")
    return fs / 1024 ** 3
    
    
###
def if_found_return_groups(pattern, iterable, group_index = None, flags = 0):

    r = []
    for i in iterable:
        sr1 = re.search(pattern, i, flags = flags) #search result 
        
        if sr1:
            r.append(sr1.groups()) if group_index is None else r.append(sr1.group(group_index))
        else:
            pass
        
    return r


###
def flatten_dict(d):
    """
    Flatten dictionary. This code is from:
    
    https://stackoverflow.com/questions/52081545/python-3-flattening-nested-dictionaries-and-lists-within-dictionaries
    """
    
    out = {}
    for key, val in d.items():
        if isinstance(val, dict):
            val = [val]
        if isinstance(val, list):
            for subdict in val:
                deeper = flatten_dict(subdict).items()
                out.update({key + '_' + key2: val2 for key2, val2 in deeper})
        else:
            out[key] = val
    return out
