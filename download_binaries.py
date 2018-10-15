import requests
import sys
import os


base_url = 'https://s3.amazonaws.com/aspire-python/data'

binary_files = [
    '/images.mat',
    '/images_large.mat',
    '/init_avg_image.mat',
    '/init_avg_image_large.mat',
]

dest_folder = 'binaries'

if os.path.exists(dest_folder):
    choice = input("binaries folder exist, override existing files [N/y]?")
    if choice.lower() != 'y':
        sys.exit()

else:
    os.mkdir(dest_folder)

for bf in binary_files:
    full_url = base_url + bf
    print(f'downloading resource from: {full_url}')
    res = requests.get(full_url)
    assert res.status_code == 200, f"Failed to download resource {bf}"

    with open(dest_folder + bf, 'wb') as fh:
        fh.write(res.content)

    print(f"saved {os.path.basename(bf)} to binaries.")

print('done.')
