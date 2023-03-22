import os
import numpy as np
import netCDF4 as nc
import metpy.calc as mpcalc
from metpy.units import units
from torch.utils.data import Dataset
from datetime import datetime, timedelta
from typing import List

from core.radar_data_aggregator import CompressedAggregatedRadarData
from core.compressed_rain_data import CompressedRainData
from core.constants import (RADAR_Q95, RAIN_Q95, TIME_GRANULARITY_MIN,
                            TERRAIN_FILE, ERA_DIR)
from core.dataset import load_data
from core.enum import DataType
from DLRA_prep.utils_data_collect.terrain_slope import load_shp, mapping

class BasicDataLoader(Dataset):
    def __init__(
        self,
        start_time: datetime,
        end_time: datetime,
        input_len: int,
        output_len: int,
        output_interval: int,
        threshold: float,
        data_type: List[str],
        hourly_data=False,
        img_size=None,
        sampling_rate=None,
        is_train: bool = False,
        is_valid: bool = False,
        is_test: bool = False,
    ):
        super().__init__()
        self._start_time = start_time
        self._end_time = end_time
        self._ilen = input_len
        self._olen = output_len
        self._output_interval = output_interval
        self._threshold = threshold
        self._dtype = data_type
        self._hourly_data = hourly_data
        self._img_size = img_size
        self._sampling_rate = sampling_rate
        
        if self._sampling_rate is None:
            self._sampling_rate = self._ilen

        self._index_map = []
        # There is an issue in python 3.6.10 with multiprocessing. workers should therefore be set to 0.
        self._dataset = load_data(
            self._s,
            self._e,
            is_validation=is_validation,
            is_test=is_test,
            is_train=is_train,
            workers=0,
        )
        self._ccrop = CenterCropNumpy(self._img_size)
        self._time = list(self._dataset['radar'].keys())
        self._raw_altitude = Altitude_data(['高程', '坡度', '坡向'])
        self._slope_x, self._slope_y = cal_slope(self._raw_altitude[0])
        self._blur_altitude = blurness(self._raw_altitude[0], k_size=5)
        # whether to also give last 5 hours hour averaged rain rate
        self._hourly_data = hourly_data
        self._hetero_data = hetero_data
        self._set_index_map()

        DataType.print(self._dtype, prefix=self.__class__.__name__)
        print(f'[{self.__class__.__name__}] {self._s}<->{self._e} ILen:{self._ilen} TLen:{self._tlen} '
              f'Toff:{self._toffset} TAvgLen:{self._tavg_len} Residual:{int(self._residual)} Hrly:{int(hourly_data)} '
              f'Sampl:{self._sampling_rate} RandStd:{self._random_std} Th:{self._threshold}')

    def _set_index_map(self):
        raw_idx = 0
        skip_counter = 0
        target_offset = self._ilen + self._toffset + self._tavg_len * self._tlen
        while raw_idx < len(self._time):
            if raw_idx + target_offset >= len(self._time):
                break

            if self._time[raw_idx + target_offset] - self._time[raw_idx] != timedelta(seconds=TIME_GRANULARITY_MIN *
                                                                                     target_offset * 60):
                skip_counter += 1
            else:
                self._index_map.append(raw_idx)

            raw_idx += 1
            
        print(f'[{self.__class__.__name__}] Size:{len(self._index_map)} Skipped:{skip_counter}')
    
    def six_multiplication(self, input_data, factor=2000):
        # add a dim of 6 at the very beginning
        output_map = [input_data for _ in range(self._ilen)]
        output_map = np.stack(output_map, axis=0)/factor # size [6, 120, 120]
        return output_map

    def _get_raw_radar_data(self, index):
        key = self._time[index]
        raw_data = self._dataset['radar'][key]
        # 2D radar
        return CompressedAggregatedRadarData.load_from_raw(raw_data)
        # 3D radar for 21lv
        #return CompressedRadarData.load_from_raw(raw_data).transpose(2,0,1) # [NZ, NX, NY]
        # 3D radar for 5lv
        #return CompressedRadarData.load_from_raw(raw_data).transpose(2,0,1)[0:10:2] # [NZ, NX, NY]

    def _input_end(self, index):
        return index + self._ilen

    def _input_range(self, index):
        return range(index, self._input_end(index))

    def _get_radar_data(self, index):
        return self._get_radar_data_from_range(self._input_range(index))

    def _get_radar_data_from_range(self, index_range):
        radar = [self._ccrop(self._get_raw_radar_data(idx))[None, ...] for idx in index_range]
        radar = np.concatenate(radar, axis=0) # [6, 120, 120]
        radar[radar < 0] = 0
        radar = radar / RADAR_Q95
        return radar

    def _get_raw_rain_data(self, index):
        key = self._time[index]
        raw_data = self._dataset['rain'][key]
        data = CompressedRainData.load_from_raw(raw_data)
        return data

    def _get_past_hourly_data(self, data_from_range_function, index):
        end_idx = self._input_end(index)
        output = []
        period = 6
        last_hour_data = None
        for _ in range(self._ilen):
            if end_idx <= 0:
                output.append(last_hour_data)
                continue

            start_idx = max(0, end_idx - period)
            data = data_from_range_function(range(start_idx, end_idx))
            last_hour_data = np.mean(data, axis=0, keepdims=True)
            output.append(last_hour_data)
            end_idx -= period

        return np.concatenate(output, axis=0)

    def _get_rain_data_from_range(self, index_range):
        rain = [self._ccrop(self._get_raw_rain_data(idx))[None, ...] for idx in index_range]
        rain = np.concatenate(rain, axis=0)
        rain[rain < 0] = 0
        rain = rain / RAIN_Q95
        return rain

    def _get_rain_data(self, index):
        # NOTE: average of last 5 frames is the rain data.
        return self._get_rain_data_from_range(self._input_range(index))

    def _get_most_recent_target(self, index, tavg_len=None):
        """
        Returns the averge rainfall which has happened in last self._tlen*10 minutes.
        """
        if tavg_len is None:
            tavg_len = self._tavg_len

        target_end_idx = self._input_end(index)
        target_start_idx = target_end_idx - tavg_len
        target_start_idx = max(0, target_start_idx)

        temp_data = [
            self._ccrop(self._get_raw_rain_data(idx))[None, ...] for idx in range(target_start_idx, target_end_idx)
        ]
        # print('Recent Target', list(range(target_start_idx, target_end_idx)))
        target = np.concatenate(temp_data, axis=0).astype(np.float32)
        target[target < 0] = 0
        assert target.shape[0] == tavg_len or target_start_idx == 0
        return target.mean(axis=0, keepdims=True)

    def _get_avg_target(self, index):
        target_start_idx = self._input_end(index) + self._toffset
        target_end_idx = target_start_idx + self._tlen * self._tavg_len
        temp_data = [
            self._ccrop(self._get_raw_rain_data(idx))[None, ...] for idx in range(target_start_idx, target_end_idx)
        ]
        # print('Target', list(range(target_start_idx, target_end_idx)))
        temp_data = np.concatenate(temp_data, axis=0)
        ''' no need average
        temp_data[temp_data < 0] = 0
        temp_data = temp_data / RAIN_Q95
        return temp_data # [18, 120, 120]
        '''
        ''' need to average'''
        target = []
        for i in range(self._tavg_len - 1, len(temp_data), self._tavg_len):
            target.append(temp_data[i - (self._tavg_len - 1):i + 1].mean(axis=0, keepdims=True))
        assert len(target) == self._tlen
        target = np.concatenate(target, axis=0)
        target[target < 0] = 0
        return target
        # return target/RAIN_Q95
        

    def __len__(self):
        return len(self._index_map) // self._sampling_rate

    def _get_internal_index(self, input_index):
        # If total we have 500 entries. Then with _ilen being 5, input index will vary in [0,100]
        input_index = input_index * self._sampling_rate
        index = self._index_map[input_index]
        return index

    def _get_past_hourly_rain_data(self, index):
        return self._get_past_hourly_data(self._get_rain_data_from_range, index)

    def _get_past_hourly_radar_data(self, index):
        return self._get_past_hourly_data(self._get_radar_data_from_range, index)

    def _random_perturbation(self, target):
        assert self._train is True

        def _rhs_idx(eps, N):
            return (eps, N) if eps > 0 else (0, N + eps)

        def _lhs_idx(eps, N):
            lidx = abs(eps) // 2
            ridx = N - (abs(eps) - lidx)
            return (lidx, ridx)

        Nx, Ny = target.shape[-2:]
        eps_x = int(np.random.normal(scale=self._random_std))
        eps_y = int(np.random.normal(scale=self._random_std))
        d_lx, d_rx = _rhs_idx(eps_x, Nx)
        d_ly, d_ry = _rhs_idx(eps_y, Ny)

        lx, rx = _lhs_idx(eps_x, Nx)
        ly, ry = _lhs_idx(eps_y, Ny)

        target[:, lx:rx, ly:ry] = target[:, d_lx:d_rx, d_ly:d_ry]
        target[:, :lx] = 0
        target[:, rx:] = 0
        target[:, :, :ly] = 0
        target[:, :, ry:] = 0
        return target

    def get_target_dt_and_season(self, index):
        index = range(index, index+6)
        target_dt = [self._time[i] for i in index] # 6 datetime
        dt_matrix = map(lambda x: self.periodization(x), target_dt)
        return np.array(list(dt_matrix), dtype = np.float32) # [6, 2]

    def periodization(self, inp_t: datetime):
        year = inp_t.year
        month = inp_t.month
        day = inp_t.day
        hour = inp_t.hour
        minute = inp_t.minute
        # arc_time = np.round(2 * np.pi * (hour + minute / 60) / 24, 10)
        arc_seas = np.round(2 * np.pi * 
        int(datetime(year, month, day).strftime("%j")) / int(datetime(year, 12, 31).strftime("%j")),
        10)
        # time_matrix = np.array([(np.sin(arc_seas), np.cos(arc_seas)),
        #                         (np.sin(arc_time), np.cos(arc_time))],)
        time_matrix = np.array([np.sin(arc_seas), np.cos(arc_seas)])
        return time_matrix
    
    def resize_as_target_input(self, dt_matrix: np.array, target:tuple) -> (np.array):
        '''
        transform into a uniform matrix
        dt_matrix shape = [N, season(2), time(2)]
        target = (H, W)
        '''
        new_matrix = np.ones([dt_matrix.shape[0], np.size(dt_matrix[0]), target[0], target[1]], dtype = np.float32)
        for t in range(dt_matrix.shape[0]):
                tmp = dt_matrix[t].reshape(-1)
                for j in range(len(tmp)):
                    new_matrix[t, j] = np.ones((target[0], target[1]), dtype=np.float32) * tmp[j]
        return new_matrix

    def initial_time(self, index):
        index = self._get_internal_index(index)
        index = index + 5 + self._toffset
        return self._time[index]

    def get_index_from_target_ts(self, ts):
        if ts in self._time:
            internal_index = self._time.index(ts)
            internal_index -= (self._toffset + self._ilen)
            index = self._index_map.index(internal_index)
            assert index % self._sampling_rate == 0
            return index // self._sampling_rate

        return None
    
    def get_era5_data(self, ec_dir, index, level, keys = ['u', 'v']):
        # load ERA5
        dt = self._time[index + 5] # initial datetime
        filepath = os.path.join(ec_dir, str(dt.year), str(dt.year)+'{:02d}'.format(dt.month), 
                                f'era5_{dt.year}{dt.month:02}{dt.day:02}.nc'
                                )
        data = Dataset(filepath, 'r')

        # northern TW region
        latStart = 20; latEnd = 27;
        lonStart = 118; lonEnd = 123.5;
        lat = np.linspace(latStart,latEnd,561)[325:445]
        lon = np.linspace(lonStart,lonEnd,441)[215:335]

        def clear_mask_fn(var):
            var = var.astype(np.float32) # np.nan is a float
            var[np.where(var.mask!=0)] = np.nan
            return np.array(var)

        def within_dege_fn(var, edge):
            return np.where((var >= edge[0]) & (var < edge[-1]))[0]

        longitude = clear_mask_fn(data.variables['longitude'][:])
        latitude = clear_mask_fn(data.variables['latitude'][:])
        levels = clear_mask_fn(data.variables['level'][:])

        # get avg value
        avg_var = []
        for key in keys:
            var = clear_mask_fn(data.variables[key][:]) # masked array [24, 20, lat, lon]
            var = var[
                :, 
                np.where(levels == level)[0][0], 
                within_dege_fn(latitude, lat)[0]:within_dege_fn(latitude, lat)[-1]+1, 
                within_dege_fn(longitude, lon)[0]:within_dege_fn(longitude, lon)[-1]+1]
            var = np.nanmean(var, axis=(-1, -2))
            avg_var.append(var[dt.hour])
        return avg_var

    def get_era5_theta_e(self, pressure, temperature, specific_humidity):
        dewpoint = mpcalc.dewpoint_from_specific_humidity(
            pressure * units('hPa'), 
            temperature * units('K'), 
            specific_humidity * units('kg/kg')
        )
        theta_e = mpcalc.equivalent_potential_temperature(
            pressure * units('hPa'), 
            temperature * units('K'), 
            dewpoint
        )
        return np.array(theta_e)


    def dotProduct(self, ec_dir, slope_x, slope_y, index):
        avg_wind = self.get_era5_data(ec_dir, index, 850)
        # dot
        result = slope_x * avg_wind[0] + slope_y * avg_wind[1]
        result = result.astype(np.float32)
        return result

    def get_info_for_model(self):
        return {'input_shape': self[0][0].shape[2:]}

    def __getitem__(self, input_index):
        index = self._get_internal_index(input_index)
        input_data = []

        if self._dtype & DataType.RADAR:
            #input_data.append(self._get_radar_data(index)) # numpy array [6, 21, 120, 120]
            input_data.append(self._get_radar_data(index)[:, None, ...]) # numpy array [6, 1, 120, 120]

        if self._dtype & DataType.ELEVATION:
            input_data.append(self.six_multiplication(self._blur_altitude)[:, None]) # [6, 1, 120, 120]

        if self._dtype & DataType.WTDOT:
            result = self.dotProduct(ERA_DIR, self._slope_x, self._slope_y, index) # [120, 120]
            result[result >= 600] = 600
            # input_data.append(self.six_multiplication(result, factor=2000)[:, None]) # [6, 1, 120, 120]
            input_data.append(self.six_multiplication(result, factor=600)[:, None]) # [6, 1, 120, 120]


        if self._hourly_data:
            input_data.append(self._get_past_hourly_rain_data(index)[:, None, ...])
            input_data.append(self._get_past_hourly_radar_data(index)[:, None, ...])
            
        if self._dtype & DataType.MONTH:
            _time_data = self.get_target_dt_and_season(index) # [6, 2]
            input_data.append(self.resize_as_target_input(_time_data, self._img_size))# [6, 2, 120, 120]

        if self._dtype & DataType.WIND:
            avg_wind = self.get_era5_data(ERA_DIR, index, 850)
            input_data.append(
                self.resize_as_target_input(
                    self.six_multiplication(avg_wind, factor = 10), # [6, 2]
                    self._img_size
                ) # [6, 2, 120, 120]
            )
        
        # ERA5 850hpa 
        if self._dtype & DataType.THETAE:
            temp, q = self.get_era5_data(ERA_DIR, index, 850, ['t', 'q'])
            theta_e = self.get_era5_theta_e(850, temp, q)
            input_data.append(
                self.resize_as_target_input(
                    self.six_multiplication(theta_e, factor = 300), # [6, 1]
                    self._img_size
                ) # [6, 1, 120, 120]
            )

        # rain data must always be the last one
        if self._dtype & DataType.RAIN:
            input_data.append(self._get_rain_data(index)[:, None, ...])

        if len(input_data) > 1:
            inp = np.concatenate(input_data, axis=1)
        else:
            inp = input_data[0]

        target = self._get_avg_target(index) 
        if self._train and self._random_std > 0:
            target = self._random_perturbation(target)

        # NOTE: mask needs to be created before tackling the residual option. We wouldn't know which entries are relevant
        # in the residual space.
        mask = np.zeros_like(target)
        mask[target > self._threshold] = 1

        assert target.max() < 500
        # NOTE: There can be a situation where previous data is absent when self._tlen + self._tavg_len -1 > self._ilen
        if self._residual:
            assert self._random_std == 0
            recent_target = self._get_most_recent_target(index)
            target -= recent_target
            return inp, target, mask, recent_target
        
        return inp, target, mask

def Altitude_data(keys):
        alt_loader = load_shp(TERRAIN_FILE)
        container = []
        for key in keys:
            target_map, latList, lonList = alt_loader.getMap(key)
            new_map, new_lat, new_lon = mapping(latList, lonList, target_map, (120, 120))
            container.append(new_map[None])
        return np.concatenate(container, axis=0) # size [3, 120, 120]
    
def cal_slope(altitude):
    # For dim=1, idx 0 is south; idx -1 is north
    ns_shift = np.zeros([altitude.shape[0], altitude.shape[1]])
    ew_shift = np.zeros([altitude.shape[0], altitude.shape[1]])
    ns_shift[:-1] = altitude[1:]
    ew_shift[:, :-1] = altitude[:, 1:]
    ns_slope = -(altitude - ns_shift) # north - south
    ew_slope = -(altitude - ew_shift) # east - west
    
    ns_slope = blurness(ns_slope, k_size=5)
    ew_slope = blurness(ew_slope, k_size=5)
    return ew_slope, ns_slope
    
def blurness(data, k_size=3):
    # Moving Average
    # data shape = [H, W]
    pd = k_size//2
    tmp = np.pad(data, ((pd,pd), (pd,pd)), 'constant')
    tmpp = np.copy(tmp)
    for i in range(pd, tmp.shape[0]-pd):
        for j in range(pd, tmp.shape[1]-pd):
            tmpp[i, j] = tmp[i-pd:i+pd+1, j-pd:j+pd+1].mean()
    return tmpp[pd:-pd, pd:-pd]

class CenterCropNumpy:
    def __init__(self, crop_shape):
        self._cropx, self._cropy = crop_shape

    def __call__(self, img):
        #x, y = img.shape[-2:]
        #startx = x // 2 - self._cropx // 2
        #starty = y // 2 - self._cropy // 2
        #return img[..., startx:startx + self._cropx, starty:starty + self._cropy] #[21, 540, 420]
        #return img[...,325:445, 215:335] #[21,120,120]
        return img

def moving_average(a, n=6):
    output = np.zeros_like(a)
    for i in range(a.shape[0]):
        output[i:i + n] += a[i:i + 1]

    output[n:] = output[n:] / n
    output[:n] = output[:n] / np.arange(1, n + 1).reshape(-1, 1, 1)
    return output