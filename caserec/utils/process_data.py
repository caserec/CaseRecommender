# coding=utf-8
""""
    These classes are responsible for read, write and process external files and information.

"""

# © 2018. Case Recommender (MIT License)

import pandas as pd

from caserec.utils.extra_functions import check_error_file

__author__ = 'Arthur Fortes <fortes.arthur@gmail.com>'


class ReadFile(object):
    def __init__(self, input_file, sep='\t', header=None, names=None, as_binary=False, binary_col=2):
        """
        ReadFile is responsible to read and process all input files in the Case Recommender

        We used as default csv files with delimiter '\t'. e.g: user item    score\n

        :param input_file: Input File with at least 2 columns.
        :type input_file: str

        :param sep: Delimiter for input files
        :type sep: str, default '\t'

        :param header: Skip header line (only work with method: read_with_pandas)
        :type header: int, default None

        :param names: Name of columns (only work with method: read_with_pandas)
        :type names: str, default None

        :param as_binary: If True, the explicit feedback will be transform to binary
        :type as_binary: bool, default False

        :param binary_col: Index of columns to read as binary (only work with method: read_with_pandas)
        :type binary_col: int, default 2

        """

        self.input_file = input_file
        self.sep = sep
        self.header = header
        self.names = names
        self.as_binary = as_binary
        self.binary_col = binary_col

        check_error_file(self.input_file)

    def read(self):
        """
        Method to read files and collect important information.

        :return: Dictionary with file information
        :rtype: dict

        """

        list_users = set()
        list_items = set()

        list_feedback = []

        dict_feedback = {}
        items_unobserved = {}
        items_seen_by_user = {}
        users_viewed_item = {}

        mean_value = 0
        number_interactions = 0

        with open(self.input_file) as infile:
            for line in infile:
                if line.strip():
                    inline = line.split(self.sep)
                    if len(inline) == 1:
                        raise TypeError("Error: Space type (sep) is invalid!")

                    user, item, value = int(inline[0]), int(inline[1]), float(inline[2])

                    dict_feedback.setdefault(user, {}).update({item: 1.0 if self.as_binary else value})
                    items_seen_by_user.setdefault(user, set()).add(item)
                    users_viewed_item.setdefault(item, set()).add(user)

                    list_users.add(user)
                    list_items.add(item)

                    mean_value += 1.0 if self.as_binary else value
                    list_feedback.append(1.0 if self.as_binary else value)
                    number_interactions += 1

        mean_value /= float(number_interactions)

        list_users = sorted(list(list_users))
        list_items = sorted(list(list_items))

        # Create a dictionary with unobserved items for each user / Map user with its respective id
        for user in list_users:
            items_unobserved[user] = list(set(list_items) - set(items_seen_by_user[user]))

        # Calculate the sparsity of the set: N / (nu * ni)
        sparsity = (1 - (number_interactions / float(len(list_users) * len(list_items)))) * 100

        dict_file = {
                        'feedback': dict_feedback,
                        'users': list_users,
                        'items': list_items,
                        'sparsity': sparsity,
                        'number_interactions': number_interactions,
                        'users_viewed_item': users_viewed_item,
                        'items_unobserved': items_unobserved,
                        'items_seen_by_user': items_seen_by_user,
                        'mean_value': mean_value,
                        'max_value': max(list_feedback),
                        'min_value': min(list_feedback),
                     }

        return dict_file

    def read_metadata_or_similarity(self):
        """
        Method to read metadata or similarity files. Expects at least 2 columns for metadata file (item metadata or
        item metadata score) and 3 columns for similarity files (item item similarity)

        :return: Dictionary with file information
        :rtype: dict
        """

        dict_values = {}
        list_col_1 = set()
        list_col_2 = set()
        mean_value = 0
        number_interactions = 0

        with open(self.input_file) as infile:
            for line in infile:
                if line.strip():
                    inline = line.split(self.sep)

                    if len(inline) == 1:
                        raise TypeError("Error: Space type (sep) is invalid!")

                    if len(inline) == 2:
                        attr1, attr2 = int(inline[0]), inline[1]
                        dict_values.setdefault(attr1, {}).update({attr2: 1.0})
                        list_col_1.add(attr1)
                        list_col_2.add(attr2)
                        number_interactions += 1
                    else:
                        attr1, attr2, value = int(inline[0]), inline[1], float(inline[2])
                        dict_values.setdefault(attr1, {}).update({attr2: 1.0 if self.as_binary else value})
                        list_col_1.add(attr1)
                        list_col_2.add(attr2)
                        mean_value += value
                        number_interactions += 1

        dict_file = {
                        'dict': dict_values,
                        'col_1': list(list_col_1),
                        'col_2': list(list_col_2),
                        'mean_value': mean_value,
                        'number_interactions': number_interactions
                     }

        return dict_file

    def read_like_triple(self):
        """
        Method to return information in the file as a triple. eg. (user, item, value)

        :return: List with triples in the file
        :rtype: list

        """

        triple_list = []

        with open(self.input_file) as infile:
            for line in infile:
                if line.strip():
                    inline = line.split(self.sep)
                    if len(inline) == 1:
                        raise TypeError("Error: Space type (sep) is invalid!")

                    user, item, value = int(inline[0]), int(inline[1]), float(inline[2])
                    triple_list.append((user, item, value))

        return triple_list

    def read_with_pandas(self):
        """
        Method to read file with pandas
        
        :return DataFrame with file lines

        """
        
        df = pd.read_csv(self.input_file, sep=self.sep, skiprows=self.header, header=None, names=self.names)

        if self.header is not None:
            df.columns = [i for i in range(len(df.columns))]

        if self.as_binary:
            df.iloc[:, self.binary_col] = 1
        return df.sort_values(by=[0, 1])

    def read_item_category(self):
        list_item_category = []
        dict_category = {}
        set_items = set()
        dict_item_category = {}

        with open(self.input_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                if line.strip():
                    inline = line.split(self.sep)
                    item, category = int(inline[0]), inline[1].rstrip()

                    list_item_category.append([item, category])

                    if category in dict_category:
                        dict_category[category] += 1
                    else:
                        dict_category[category] = 1

                    set_items.add(item)

                    if item not in dict_item_category:
                        dict_item_category[item] = []
                        dict_item_category[item].append(category)
                    else:
                        dict_item_category[item].append(category)

        return [dict_category, list_item_category, set_items, dict_item_category]


class WriteFile(object):
    def __init__(self, output_file, data=None, sep="\t", mode='w', as_binary=False):
        """
        Class to write predictions and information
        
        :param output_file: File with dir to write the information
        :type output_file: str
        
        :param data: Data to be write
        :type data: list, dict, set

        :param sep: Delimiter for input files
        :type sep: str, default '\t'
        
        :param mode: Method to write file
        :type mode: str, default 'w'
        
        :param as_binary: If True, write score equals 1
        :type as_binary: bool, default False
        
        """
        
        self.output_file = output_file
        self.data = data
        self.sep = sep
        self.mode = mode
        self.as_binary = as_binary

    def write(self):
        """
        Method to write using data as list. e.g.: [user, item, score]
        
        """
        
        with open(self.output_file, self.mode) as infile:
            for triple in self.data:
                if self.as_binary:
                    infile.write('%d%s%d%s%f\n' % (triple[0], self.sep, triple[1], self.sep, 1.0))
                else:
                    infile.write('%d%s%d%s%f\n' % (triple[0], self.sep, triple[1], self.sep, triple[2]))

    def write_with_dict(self):
        """
        Method to write using data as dictionary. e.g.: user: {item : score}
                        
        """
        
        with open(self.output_file, self.mode) as infile:
            for user in self.data:
                for pair in self.data[user]:
                    infile.write('%d%s%d%s%f\n' % (user, self.sep, pair[0], self.sep, pair[1]))

    def write_with_pandas(self, df):
        """
        Method to use a pandas DataFrame as data
        
        :param df: Data to write in output file
        :type df: DataFrame
        
        """

        df.to_csv(self.output_file, sep=self.sep, mode=self.mode, header=None, index=False)
