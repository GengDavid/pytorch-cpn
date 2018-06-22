import os
import errno

def mkdir_p(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def isfile(fname):
    return os.path.isfile(fname) 

def isdir(dirname):
    return os.path.isdir(dirname)

def join(path, *paths):
    return os.path.join(path, *paths)
    
def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)