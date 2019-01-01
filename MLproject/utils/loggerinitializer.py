'''
Created on Nov 9, 2018

@author: mofir
'''

import logging
import os.path

def initialize_logger(output_dir):
    #Get main logger to add new specifications. Then set global log level to debug
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to info
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO) #on the console we want only INFO or higher messages presented
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler) #add my console handler to my main logger.

    # create error file handler and set level to error
    handler = logging.FileHandler(os.path.join(output_dir, "run_error.log"),"w", encoding=None, delay="true")
    handler.setLevel(logging.ERROR)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # create info file handler and set level to info
    handler = logging.FileHandler(os.path.join(output_dir, "info_messages.log"),"w")
    handler.setLevel(logging.INFO) #minimum level - all messages will be written
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # create debug file handler and set level to debug
    handler = logging.FileHandler(os.path.join(output_dir, "all_messages.log"),"w")
    handler.setLevel(logging.DEBUG) #minimum level - all messages will be written
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)