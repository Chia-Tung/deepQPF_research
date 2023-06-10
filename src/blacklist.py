from src.utils.time_util import TimeUtil

three_days = TimeUtil.three_days

class Blacklist:
    BLACKLIST_PATH = '/wk171/handsomedong/deepQPF_research/exp_config/black_list.txt'

    GREYLIST = three_days(2018,5,7) + three_days(2018,5,8) + three_days(2019,7,22) + three_days(2019,8,18) + \
                three_days(2019,9,30) + three_days(2019,10,1) + three_days(2019,12,30) + three_days(2019,12,31) + \
                three_days(2021,6,4) + three_days(2021,10,16)
    GREYLIST = sorted(list(set(GREYLIST)))