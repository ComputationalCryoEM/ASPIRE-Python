import click
from click import UsageError

from aspire.apple.apple import Apple


@click.command()
@click.option(
    "--mrc_dir", help="Path to folder containing all mrc files for particle picking"
)
@click.option("--mrc_file", help="Path to a single mrc file for particle picking")
@click.option(
    "--output_dir",
    help="Path to folder to save *.star files. If unspecified, no star files are created.",
)
@click.option(
    "--create_jpg", is_flag=True, help="save *.jpg files for picked particles."
)
def apple(mrc_dir, mrc_file, output_dir, create_jpg):
    """Pick and save particles from one or more mrc files."""

    # Exactly one of mrc_dir/mrc_file should be specified.
    # We handle this manually here until Click supports mutually exclusive options.
    if all([mrc_dir, mrc_file]) or not any([mrc_dir, mrc_file]):
        raise UsageError("Specify one of --mrc_dir or --mrc_file.")

    picker = Apple(output_dir)
    if mrc_dir:
        picker.process_folder(mrc_dir, create_jpg=create_jpg)
    elif mrc_file:
        picker.process_micrograph(mrc_file, create_jpg=create_jpg)
