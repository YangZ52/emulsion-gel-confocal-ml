"""Small helper for running the publication analysis from a Jupyter notebook.

Usage in a notebook:

    code_path = "/path/to/emulsion-gel-confocal-ml/src/emulsion_gel_confocal_ml_publication.py"
    with open(code_path, "r") as f:
        code = f.read()
    exec(code)

If your data are not inside this repository, set the environment variables
below before calling exec(code).
"""

import os

os.environ["EMULSION_GEL_EXCEL"] = "/path/to/emulsion gel confocal.xlsx"
os.environ["EMULSION_GEL_IMAGE_DIR"] = "/path/to/real confocal images of emulsion gels"
os.environ["EMULSION_GEL_OUTPUT_DIR"] = "/path/to/output_folder"

code_path = "/path/to/emulsion-gel-confocal-ml/src/emulsion_gel_confocal_ml_publication.py"
with open(code_path, "r") as f:
    code = f.read()
exec(code)
