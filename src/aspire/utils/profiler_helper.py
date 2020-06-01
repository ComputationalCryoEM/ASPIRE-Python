#/usr/bin/env python
"""
Utility function to demonstrate wrapping arbitrary code in a context.

Here we wrap some standard library python profiler.

Usage for a line in  something.py:

...
line = doesAthing
...

replace
...
with prof_sandwich:
    line = doesAthing

Defaults to writing output to file something.pstats.
If you provide event_name, will write to something_event_name.pstats.
Verbose=True optionally prints.

...

Note in recent version of python/cProfile,
they include a context manager, but it doesn't
let you do arbitrary things we might want.

Added nvtx for Junchao.  Defaults false.
    nvtx=True it will pass down default id_color=-1,
    nvtz=int it will pass down id_color=int

"""

import contextlib
import cProfile
import inspect
import io
import os
import pstats
import time

_prof = cProfile.Profile()

@contextlib.contextmanager
def prof_sandwich(event_name=None, verbose=False, toFile=True, suffix=None, nvtx=False):

    # Setup nvtx
    # Since some machines may not have cuda, import it here as needed:
    if nvtx is not False:
        from cupy.cuda.nvtx import RangePop
        from cupy.cuda.nvtx import RangePush
    # defaults
    nvtx_id_color = -1
    nvtx_event_name = str(event_name) or 'Default Event'
    if nvtx is not False and nvtx >= 0:
        nvtx_id_color = nvtx
        if event_name is None:
            nvtx_event_name = f'Event {nvtx}'


    # Create string buffer (memory).
    s = io.StringIO()

    if nvtx: RangePush(nvtx_event_name, nvtx_id_color)
    _prof.enable()
    yield
    _prof.disable()
    if nvtx: RangePop()

    # Create a stats object
    ps = pstats.Stats(_prof, stream=s).sort_stats('cumulative')
    # Print the stats to the buffer
    ps.print_stats()

    if verbose:
        # Note if you have a logger can swap
        print(s.getvalue())

    if toFile:
        # Figure out filename.
        caller = inspect.stack()[2][1] # 3.5+ can use inspect.stack()[2].filename
        fn = os.path.splitext(os.path.basename(caller))[0]
        if event_name is not None:
            fn = f'{fn}_{event_name}'
        if suffix is not None:
            fn = f'{fn}_{suffix}'
        fn = f'{fn}.pstats'

        # If that file exists, bump it.
        #   Let the operator decide if they want to delete it.
        if os.path.exists(fn):
            os.rename(fn, f'{fn}-{int(time.time())}')

        with open(fn, 'w') as fh:
            fh.write(s.getvalue())

