"""Entry point: their data_extraction_root with the chunked nurseCharting reader."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, "/work/code")
sys.path.insert(0, "/work/runner")
from data_extractor import utils           # noqa: E402
from nc_chunked_patch import read_nc_table_chunked  # noqa: E402
utils.read_nc_table = read_nc_table_chunked
import data_extraction_root                # noqa: E402
sys.argv = [sys.argv[0], "--eicu_dir", "/data", "--output_dir", "/out"]
data_extraction_root.main()
