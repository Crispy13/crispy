from .core.eclogging import create_logger, make_file_handler
import datetime, os
from .misc.email_alert import email_alert
import sys

log_dir = os.path.dirname(__file__) + '/logs'

if not os.path.isdir(log_dir):
    print(f"crispy: Make a directory for logging at {log_dir}...")
    os.makedirs(log_dir)
    
logger = create_logger(file_handler_path = log_dir + f'''/crispy_log_{datetime.datetime.now().strftime("%Y-%m")}.log''', logger_name = 'crispy')

from .core.ecf import *