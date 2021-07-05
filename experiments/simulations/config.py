import sys
import pathlib

from matplotlib import pyplot as plt
import matplotlib as mpl

script_name = pathlib.Path(sys.argv[0]).stem
FIGURES_DIR = pathlib.Path(
    __file__).parents[2] / "figures" / "simulations" / script_name
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

# mpl.rc("text", usetex=True)
# mpl.rc("font", family="serif")
# mpl.rc(
#     "text.latex",
#     preamble=r"\usepackage{mathpazo} \usepackage{eulervm} \usepackage{amssymb}"
#     r"\usepackage{amsmath} \usepackage{bm} \usepackage{DejaVuSans}",
# )
