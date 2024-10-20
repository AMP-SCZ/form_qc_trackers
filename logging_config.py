import logging

logger = logging.getLogger('unit_test_logger')
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('test_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
