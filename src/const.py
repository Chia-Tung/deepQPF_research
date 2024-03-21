import geopandas as gpd
import matplotlib as mpl
import yaml

from src.utils.time_util import TimeUtil

### Blacklist for datetime
three_days = TimeUtil.three_days
BLACKLIST_PATH = "./exp_config/black_list.txt"
GREYLIST = (
    three_days(2018, 5, 7)
    + three_days(2018, 5, 8)
    + three_days(2019, 7, 22)
    + three_days(2019, 8, 18)
    + three_days(2019, 9, 30)
    + three_days(2019, 10, 1)
    + three_days(2019, 12, 30)
    + three_days(2019, 12, 31)
    + three_days(2021, 6, 4)
    + three_days(2021, 10, 16)
)

### Load config
config_file = "./exp_config/transformer_config_large.yml"
with open(config_file, "r") as content:
    CONFIG = yaml.safe_load(content)

### Load Taiwan county data
COUNTY_DATA = gpd.read_file("./visualization/town_shp/COUNTY_MOI_1090820.shp")

### set colorbar
CWBRR = mpl.colors.ListedColormap(
    [
        "#FFFFFF",
        "#9CFCFF",
        "#03C8FF",
        "#059BFF",
        "#0363FF",
        "#059902",
        "#39FF03",
        "#FFFB03",
        "#FFC800",
        "#FF9500",
        "#FF0000",
        "#CC0000",
        "#990000",
        "#960099",
        "#C900CC",
        "#FB00FF",
        "#FDC9FF",
    ]
)
bounds = [0, 1, 2, 5, 10, 15, 20, 30, 40, 50, 70, 90, 110, 130, 150, 200, 300]
NORM = mpl.colors.BoundaryNorm(bounds, CWBRR.N)
