import sys

__author__ = 'Arthur'


def check_error_file(file_check):
    try:
        open(file_check)
    except IOError:
        print('Error: File cannot be empty or file is invalid: ' + str(file_check))
        sys.exit()
