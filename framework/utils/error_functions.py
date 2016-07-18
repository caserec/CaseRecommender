import sys

__author__ = 'Arthur'


def check_error_file(file_check):
    try:
        open(file_check)
    except IOError:
        print('Error: File cannot be empty or file is invalid: ' + str(file_check))
        sys.exit()


def check_len_lists(list1, list2):
    if len(list1) != len(list2):
        print('Error: Number of files in train list and rank list must be equal!')
        sys.exit()
