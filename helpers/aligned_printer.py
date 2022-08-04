import itertools


class AlignedPrinter(object):
    """ Print Rows nicely as a table. """
    def __init__(self):
        self.rows = []
        self.maxs = []

    def append(self, *row):
        self.rows.append(row)
        self.maxs = [max(max_cur, len(row_entry))
                     for max_cur, row_entry in
                     itertools.zip_longest(self.maxs, row, fillvalue=0)]

    def print(self):
        for row in self.rows:
            for width, row_entry in zip(self.maxs, row):
                print('{row_entry:{width}}'.format(row_entry=row_entry, width=width), end='   ')
            print()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.print()
