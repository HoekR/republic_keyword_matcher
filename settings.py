import json
import os
import re
import copy
from collections import defaultdict
import republic.parser.republic_file_parser as file_parser



host = "annotation.republic-caf.diginfra.org/"
port = 443
url_prefix = "elasticsearch"

config = {
    "elastic_config": {
        "host": host,
        "port": port,
        "url_prefix": url_prefix,
        "url": f"http://{host}:{port}/" + f"{url_prefix}/" if url_prefix else ""
    },
}


# base_config = {
#     "tiny_word_width": 15, # pixel width
#     "avg_char_width": 20,
#     "remove_tiny_words": True,
#     "remove_line_numbers": False,
# }
#
#
# def get_pages_info(config):
#     scan_files = file_parser.get_files(config["data_dir"])
#     print("Number of scan files:", len(scan_files))
#     return file_parser.gather_page_columns(scan_files)
#
# def set_config_year(base_config, year):
#     config = copy.deepcopy(base_config)
#     config["year"] = year
#     config["data_dir"] = os.path.join(config["base_dir"], '{}'.format(year))
#     return config
