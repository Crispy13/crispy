import logging, sys, os

# create logger
def create_logger(file_handler_path = None, logger_name = 'logger', file_handler_mode = 'a',
                  logger_level = logging.DEBUG, file_handler_level = logging.DEBUG, use_stream_handler = True, stream_handler_level = logging.INFO):
    assert logger_name not in logging.root.manager.loggerDict, f"{logger_name} logger already exists!"
        
    logger = logging.getLogger(logger_name)
#     logger.handlers = []
    logger.setLevel(logger_level)
    logger.propagate = False
    
    # create console handler and set level to debug
    if file_handler_path is not None:
        fh = logging.FileHandler(file_handler_path, mode = file_handler_mode)
        fh.setLevel(file_handler_level)

    if use_stream_handler is True:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(stream_handler_level)
        logger.addHandler(ch)
        
    # create formatter
    formatter = logging.Formatter('(%(asctime)s - %(name)s - %(levelname)s) %(message)s\n', datefmt = "%b %d, %Y %H:%M:%S")

    # add handlers to logger
    if file_handler_path is not None:
        # add formatter to ch
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    logger.debug(f"{logger_name} logger was initialized.")
    
    return logger

# load logger
def load_logger(logger_name = 'crispy13'):
    if logger_name in logging.root.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        return logger
    else:
        raise Exception("Any logger was not made. Before using the functions of crispy13 package, create a logger using create_logger function first.")
    
    
###
def make_file_handler(file_handler_path, file_handler_mode = 'a', file_handler_level = logging.DEBUG):
    fh = logging.FileHandler(file_handler_path, mode = file_handler_mode)
    fh.setLevel(file_handler_level)

    formatter = logging.Formatter('(%(asctime)s - %(name)s - %(levelname)s) %(message)s\n', datefmt = "%b %d, %Y %H:%M:%S")

    fh.setFormatter(formatter)

    return fh
        