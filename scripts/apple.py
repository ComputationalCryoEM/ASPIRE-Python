from aspyre.apple.apple import Apple
from aspyre.utils.config import ConfigArgumentParser


if __name__ == '__main__':

    parser = ConfigArgumentParser(description='Apple Picker')

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--mrc_dir", help="Path to folder containing all mrc files for particle picking")
    group.add_argument("--mrc_file", help="Path to a single mrc file for particle picking")

    parser.add_argument("--output_dir",
                        help="Path to folder to save *.star files. If unspecified, no star files are created.")
    parser.add_argument("--create_jpg", action='store_true', help="save *.jpg files for picked particles.")

    with parser.parse_args() as args:

        apple = Apple(args.output_dir)
        if args.mrc_dir:
            apple.process_folder(args.mrc_dir, create_jpg=args.create_jpg)
        elif args.mrc_file:
            apple.process_micrograph(args.mrc_file, create_jpg=args.create_jpg)
