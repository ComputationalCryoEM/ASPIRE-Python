from warnings import warn
from collections import namedtuple
from itertools import tee, zip_longest, starmap


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip_longest(a, b, fillvalue='')


def read_star(filename):
    f = open(filename, 'r')

    block_name = None

    fpair = pairwise(f)

    blocks = {}

    for line, next_line in fpair:
        line = line.strip()

        if line.startswith('data_'):
            block_name = line[5:]

            if not block_name:
                block_name = '__root__'

        if line.startswith('loop_'):
            if not block_name:
                raise ValueError('Loop outside of data block')

            star_loop = read_loop(fpair)

            blocks[block_name] = star_loop

            block_name = None
        elif next_line.startswith('_'):
            if not block_name:
                raise ValueError('List outside of data block')

            star_list = read_star_list(fpair)

            blocks[block_name] = star_list

            block_name = None

    f.close()

    return blocks


def read_loop(fpair):
    labels = read_star_list(fpair)
    values = read_star_data(fpair, len(labels))

    star_loop_tuple = namedtuple('star_loop', labels._fields)

    star_loop = list(starmap(star_loop_tuple, values))

    return star_loop


def read_star_list(fpair):
    labels = []
    values = []

    for line, next_line in fpair:
        line = line.split('#')[0]
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        line_label = parts[0][1:]

        if len(parts) == 1:
            line_value = None
        else:
            line_value = parts[1]

        labels.append(line_label)
        values.append(line_value)

        if not next_line.startswith('_'):
            break

    values = list(map(auto_cast, values))

    star_list_tuple = namedtuple('star_list', labels)

    return star_list_tuple(*values)


def read_star_data(fpair, num_fields):
    values = []

    for line, next_line in fpair:
        line = line.split('#')[0]
        line = line.strip()

        if not line:
            break

        line_values = line.split()

        if len(line_values) != num_fields:
            warn('Incorrect number of fields in STAR file loop',
                RuntimeWarning)
            line_values += [''] * (num_fields - len(line_values))
            line_values = line_values[0:num_fields]

        line_values = list(map(auto_cast, line_values))

        values.append(line_values)

    return values


def auto_cast(value):
    try:
        value = float(value)
    except:
        if value == 'None':
            value = None

    return value
