import datetime
from pathlib import Path

# Needed when executing this file
# import sys
# sys.path.append(str(Path.cwd()))

from src.utils.time_util import TimeUtil
three_days = TimeUtil.three_days

class Blacklist:
    BLACKLIST_PATH = '/wk171/handsomedong/deepQPF_research/exp_config/black_list.txt'
    BLACKLIST = set()

    GREYLIST = three_days(2018,5,7) + three_days(2018,5,8) + three_days(2019,7,22) + \
                three_days(2019,8,18) + three_days(2019,9,30) + three_days(2019,10,1) + \
                three_days(2019,12,30) + three_days(2019,12,31) + three_days(2021,6,4) + \
                three_days(2021,10,16)
    GREYLIST = set(GREYLIST)

    @classmethod
    def read_blacklist(cls) -> None:
        with open(Blacklist.BLACKLIST_PATH, 'r') as f:
            text = f.readline()
            while text:
                cls.BLACKLIST.add(eval(text[:-1])) # skip '\n'
                text = f.readline()

if __name__ == '__main__':
    import yaml
    from p_tqdm import p_map
    from src.data_loaders.rain_loader_jay import RainLoaderJay

    config_file = './exp_config/exp2.yml'
    with open(config_file, "r") as content:
        config = yaml.safe_load(content) 

    # instantiate rain loader
    data_meta_info=config['train_config']['data_meta_info']
    rain_meta_info = data_meta_info['rain']
    rain_loader = RainLoaderJay(**rain_meta_info)

    # get all datetime
    all_paths = sorted(
        Path(rain_loader.path).rglob(f"**/*.{rain_loader.formatter.split('.')[-1]}"))
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
            with open('/wk171/handsomedong/deepQPF_research/exp_config/black_list.txt', 'a') as f:
                f.write(dt.__repr__() + '\n')

    results = p_map(fn, time_map, **{"num_cpus": 8})