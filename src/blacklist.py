import datetime
from pathlib import Path

from src.const import BLACKLIST_PATH, GREYLIST
from src.utils.time_util import TimeUtil

# Needed when executing this file
# import sys
# sys.path.append(str(Path.cwd()))


class Blacklist:
    BLACKLIST = set()
    GREYLIST = set(GREYLIST)

    @classmethod
    def read_blacklist(cls) -> None:
        with open(BLACKLIST_PATH, "r") as f:
            text = f.readline()
            while text:
                cls.BLACKLIST.add(eval(text[:-1]))  # skip '\n'
                text = f.readline()


if __name__ == "__main__":
    import yaml
    from p_tqdm import p_map

    from src.data_loaders.rain_loader_jay import RainLoaderJay

    config_file = "./exp_config/exp2.yml"
    with open(config_file, "r") as content:
        config = yaml.safe_load(content)

    # instantiate rain loader
    data_meta_info = config["train_config"]["data_meta_info"]
    rain_meta_info = data_meta_info["rain"]
    rain_loader = RainLoaderJay(**rain_meta_info)

    # get all datetime
    all_paths = sorted(
        Path(rain_loader.path).rglob(f"**/*.{rain_loader.formatter.split('.')[-1]}")
    )
    all_time = []
    for path in all_paths:
        all_time.append(TimeUtil.parse_filename_to_time(path, rain_loader.formatter))

    # filter out insufficient datetime
    time_map = rain_loader.oup_initial_time_fn(all_time, 3, 6)

    """ 
    if there is any grid point which has an hourly rainfall larger than
    250 mm, then set this datetime in the blacklist.
    """

    def fn(dt: datetime) -> None:
        data = rain_loader.load_output_data(dt, 3, 6, [20, 27], [118, 123.5])
        if data.max() >= 250:
            with open(BLACKLIST_PATH, "a") as f:
                f.write(dt.__repr__() + "\n")

    results = p_map(fn, time_map, **{"num_cpus": 8})
