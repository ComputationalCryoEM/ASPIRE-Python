import pandas as pd

from aspyre.utils import ensure


class StarFile:
    def __init__(self):
        self.sections = {}


    def read(self, fn):
        with open(fn, 'r') as f:
            index = self._index_star(f)

        self.sections = {}

        for (name, entry) in index.items():
            self.sections[name] = self._parse_section(fn, entry)


    def _parse_section(self, fn, entry):
        begin_line_num = entry['begin_line_num']
        end_line_num = entry['end_line_num']

        if entry['is_loop']:
            return self._parse_loop(fn, begin_line_num, end_line_num)
        else:
            return self._parse_list(fn, begin_line_num, end_line_num)


    def _parse_list(self, fn, begin_line_num, end_line_num):
        skiprows = begin_line_num - 1
        nrows = end_line_num - begin_line_num + 1
        converters = {0: lambda name: name[1:], 1: str}

        df = pd.read_csv(fn, delim_whitespace=True, skiprows=skiprows, nrows=nrows, converters=converters, index_col=0, header=None, names=['key', 'value'])
        df = df.transpose()

        return df


    def _parse_loop(self, fn, begin_line_num, end_line_num):
        names = []

        with open(fn, 'r') as f:
            for _ in range(begin_line_num - 1):
                next(f)

            for line in f:
                line = line.strip()
                if line.startswith('_'):
                    name = line.split()[0][1:]
                    names.append(name)
                else:
                    break

        skiprows = begin_line_num + len(names) - 1
        nrows = end_line_num - skiprows

        df = pd.read_csv(fn, delim_whitespace=True, skiprows=skiprows, nrows=nrows, names=names, dtype=str)

        return df


    def _index_star(self, f):
        entries = {}

        line_num = 0
        while True:
            # Skip empty lines until non-empty or EOF
            for line in f:
                line_num += 1
                if line.strip() and not line.startswith('#'):
                    break
            else:
                break

            line = line.strip()

            ensure(line.startswith('data_'), f'Invalid line in STAR file: "{line}"')

            name = line[5:]

            # Skip empty line
            f.readline()
            line_num += 1

            # Determine if it's a loop or not
            line = f.readline().strip()
            line_num += 1
            is_loop = (line == 'loop_')

            if is_loop:
                begin_line_num = line_num + 1
            else:
                begin_line_num = line_num

            # Skip empty lines until empty or EOF
            for line in f:
                line_num += 1
                if not line.strip():
                    break

            end_line_num = line_num - 1

            entry = {'is_loop': is_loop, 'begin_line_num': begin_line_num, 'end_line_num': end_line_num}

            entries[name] = entry

        return entries
