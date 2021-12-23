import sys
import csv


class Data:
    # todo: add documentation once the class is more complete
    def __init__(self, matrix, find_headers=False, headers=None):
        if not isinstance(matrix, Data):
            self.original = matrix
            self.find_headers = find_headers
            self.headers = headers
            self.matrix = self._check_shape()
            self.m, self.n = len(self.matrix), len(self.matrix[0])
            self.shape = (self.m, self.n)

            if self.headers:
                self.data_dict = {self.headers[c]: self._construct_column(self.headers[c]) for c in range(self.n)}
            else:
                self.data_dict = {c: self._construct_column(c) for c in range(self.n)}
        else:
            self.original = matrix.original
            self.matrix = matrix.matrix
            self.m, self.n = matrix.m, matrix.n
            self.shape = (self.m, self.n)
            self.headers = matrix.headers
            self.data_dict = matrix.data_dict

    def _check_shape(self):
        """
        This checks the shape of a matrix. If the matrix is a 1-D list, it converts it to a 2-D matrix in which there is
        one column. It also adjusts the shape of the matrix. If there are rows with unequal lengths, it finds the row
        with the largest length and extends all other rows to the same length. WARNING: The added values are empty
        strings. Attempts of performing calculations may be hinder the process if row lengths are not the same.

        :return: A 2-D matrix
        """
        row_lengths, matrix = [], []
        for r in self.original:
            try:
                if not isinstance(r, list):
                    r = list(r)
                row_lengths.append(len(r))
                matrix.append(r)
            except Exception:
                r = [r]
                row_lengths.append(1)
                matrix.append(r)
        n = max(row_lengths)
        for r in matrix:
            for _ in range(n - len(r)):
                r.append('')

        if self.find_headers and not self.headers:
            self.headers = [str(matrix[0][c]) for c in range(n)]
            if '' in self.headers:
                raise Exception('There cannot be an empty string in a header.')
            return matrix[1:]

        if self.headers:
            if '' in self.headers:
                raise Exception('There cannot be an empty string in a header.')
        return matrix

    def _construct_column(self, item):
        """
        Creates a generator for specified column. Used for the __getitem__ special method.

        :param item: The column header or the column index if there are no headers.
        :return: A generator for the column of data
        """
        if self.headers:
            c = self.headers.index(item)
            for r in range(self.m):
                yield self.matrix[r][c]
        else:
            for r in range(self.m):
                yield self.matrix[r][item]

    def print(self, table=False, underline=False, columns=None):
        """
        Prints out the dataframe in a presentable manner. If there are headers, then they will automatically be printed.
        If there are headers, but a columns argument is specified, then the headers will be overwritten. These actions
        do not change the original matrix.

        :param table: Prints a table style.
        :param underline: Underlines the first row. Doesn't change the original matrix.
        :param columns: This is a list. Inserts a header if desired. Doesn't change the original matrix.
        """
        m, n = self.m, self.n
        string_matrix = [[str(self.matrix[r][c]).replace('\n', ' ').replace('\t', '    ') for c in range(n)] for r in range(m)]

        if columns:
            for _ in range(n - len(columns)):
                columns.append('')
            string_matrix.insert(0, columns)
            m += 1

        if self.headers and not columns:
            string_matrix.insert(0, self.headers)
            m += 1

        if table and not self.headers and not columns:
            string_matrix.insert(0, ['' for _ in range(n)])
            m += 1

        if underline or table:
            underlines = ['-'*len(string_matrix[0][c]) for c in range(len(string_matrix[0]))]
            string_matrix.insert(1, underlines)
            m += 1

        column_widths = [max([len(string_matrix[r][c]) for r in range(m)]) for c in range(n)]  # Column widths in characters

        for r in range(m):
            for c in range(n):
                value = string_matrix[r][c]
                space = column_widths[c] - len(value)
                if c == n - 1:
                    if table:
                        if r != 1:
                            sys.stdout.write(value)
                        else:
                            sys.stdout.write(value + '-'*space)
                    else:
                        sys.stdout.write(value)
                else:
                    if table:
                        if r != 1:
                            sys.stdout.write(value + ' '*space + ' | ')
                        else:
                            sys.stdout.write(value + '-'*space + '-+-')
                    else:
                        sys.stdout.write(value + ' '*(space + 3))
            sys.stdout.write('\n')

    def write_to_csv(self, csv_file):
        """
        Created a csv file for the matrix. Does include the headers if they exist.

        :param csv_file: The path to desired csv file
        :return:
        """
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            if self.headers:
                matrix = self.matrix[:]
                matrix.insert(0, self.headers)
            else:
                matrix = self.matrix[:]
            writer.writerows(matrix)

    def __repr__(self):
        return f'Data(m={self.m}, n={self.n}, headers={self.headers})'

    def __getitem__(self, item):
        return list(self.data_dict[item])


def print_matrix(mat, **kwargs):
    """
    Prints out a matrix in a presentable manner. If the matrix is being printed multiple times, consider printing from
    a declared object.

    :param mat: Matrix to print
    :param kwargs: See Data.print() method.
    """
    Data(mat).print(**kwargs)


def read_csv(csv_file, delimiter=',', find_headers=True, floats=True):
    """
    Reads and returns a Data object.

    :param csv_file: Path to the csv file
    :param delimiter: The delimiter
    :param find_headers: Specifies whether or not the first row should be taken to be the header.
    :param floats: choose weather or not to try to convert numerical values into floats
    :return: A Data object
    """
    with open(csv_file, 'r') as f:
        if not floats:
            return Data(list(csv.reader(f, delimiter=delimiter)), find_headers=find_headers)
        else:
            lines = list(csv.reader(f, delimiter=delimiter))
            for r in range(len(lines)):
                for c in range(len(lines[0])):
                    try:
                        lines[r][c] = float(lines[r][c])
                    except Exception:
                        pass
            return Data(lines, find_headers=find_headers)


def augment():
    # todo: this is something useful that could be added later
    pass
