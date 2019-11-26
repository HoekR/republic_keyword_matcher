mport json
import os
import re
from collections import defaultdict

import copy

year = 1725
base_config = {
    "year": year,
    "base_dir": "/Users/rikhoekstra/surfdrive/rsg_share/data/hocr",
    "page_index": "republic_hocr_pages",
    "page_doc_type": "page",
    "tiny_word_width": 15, # pixel width
    "avg_char_width": 20,
    "remove_tiny_words": True,
    "remove_line_numbers": False,
}


def get_pages_info(config):
    scan_files = file_parser.get_files(config["data_dir"])
    print("Number of scan files:", len(scan_files))
    return file_parser.gather_page_columns(scan_files)

def set_config_year(base_config, year):
    config = copy.deepcopy(base_config)
    config["year"] = year
    config["data_dir"] = os.path.join(config["base_dir"], '{}'.format(year))
    return config

year_config = set_config_year(base_config, year)
pages_info = get_pages_info(year_config)

