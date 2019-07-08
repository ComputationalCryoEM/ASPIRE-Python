from aspyre.apple.apple import Apple

if __name__ == '__main__':
    apple = Apple(output_dir='apple_out', create_jpg=True)
    apple.process_folder('../tests/saved_test_data/mrc_files')
