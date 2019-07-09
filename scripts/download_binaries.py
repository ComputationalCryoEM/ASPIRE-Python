import os
import sys
import requests

from aspyre.aspire.utils.helpers import yellow
from aspyre.aspire.common.config import AspireConfig


base_url = 'https://storage.googleapis.com/aspire-python'

binary_files = [
    '/bessel.npy',
    '/images.mat',
    '/images_large.mat',
    '/init_avg_image.mat',
    '/init_avg_image_large.mat',
    '/init_avg_image_large.mat',
]


if os.path.exists(AspireConfig.binaries_folder):
    choice = input("'binaries' folder exist, override existing files [N/y]?")
    if choice.lower() != 'y':
        sys.exit()

else:
    os.mkdir(AspireConfig.binaries_folder)

for bf in binary_files:
    full_url = base_url + bf
    print(f'downloading resource: {full_url}')
    res = requests.get(full_url)
    assert res.status_code == 200, f"Failed to download resource {bf}"

    with open(AspireConfig.binaries_folder + bf, 'wb') as fh:
        fh.write(res.content)

    print(f"{yellow(os.path.basename(bf))} is saved to 'binaries'.")

print('done.\n  $ ls ./binaries')
