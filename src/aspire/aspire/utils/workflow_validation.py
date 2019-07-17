"""
[Converted from Matlab file cryo_workflow_preprocess_validate.m]

Validate that the workflow file has all the required parameters to exectue preprocessing.
"""

import logging
import sys

from xmltodict import parse
from aspire.aspire.common.config import necessary_workflow_fields

logger = logging.getLogger(__name__)


def cryo_workflow_preprocess_validate(workflow_xml_file):
    try:
        with open(workflow_xml_file, 'r') as xml_file_handler:
            tree_dict = parse(xml_file_handler.read())

    except FileNotFoundError:
        logger.error(f'workflow file "{workflow_xml_file}" does not exist!')
        sys.exit(1)

    # validate all necessary entries are in the file
    for field in necessary_workflow_fields:
        for sub_field in necessary_workflow_fields[field]:
            try:
                tree_dict['workflow'][field][sub_field]
            except KeyError:
                logger.error("A necessary value is missing from workflow file!\n"
                             f"workflow file: {workflow_xml_file}\nmissing field: "
                             f"<workflow:{field}:{sub_field}>")
                sys.exit(2)


if __name__ == "__main__":
    cryo_workflow_preprocess_validate(sys.argv[1])
