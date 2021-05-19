# -*- coding: utf-8 -*-
"""
Created on Wed May 19 13:42:28 2021

@author: freeridingeo
"""

import collections
import datetime as dt
import logging
from itertools import repeat

import numpy as np

from sentinelhub import (DataCollection, MimeType, SHConfig, SentinelHubCatalog, 
                         SentinelHubDownloadClient, SentinelHubRequest, 
                         bbox_to_dimensions, filter_times, parse_time_interval)
from sentinelhub.data_collections import handle_deprecated_data_source

import sys
sys.path.append("D:/Code/eotopia/repo_core")
from eodata import EOPatch
from eotask import EOTask
from constants import FeatureType, FeatureTypeSet

LOGGER = logging.getLogger(__name__)


def get_available_timestamps(bbox, config, data_collection, time_difference, 
                             time_interval=None, maxcc=None):
    """
    Helper function to search for all available timestamps, based on query parameters
    
    :param bbox: Bounding box
    :type bbox: BBox
    :param time_interval: Time interval to query available satellite data from
        type time_interval: different input formats available (e.g. (str, str), or (datetime, datetime)
    :param data_collection: Source of requested satellite data.
    :type data_collection: DataCollection
    :param maxcc: Maximum cloud coverage, in ratio [0, 1], default is None
    :type maxcc: float
    :param time_difference: Minimum allowed time difference, used when filtering dates, None by default.
    :type time_difference: datetime.timedelta
    :param config: Sentinel Hub Config
    :type config: SHConfig
    :return: list of datetimes with available observations
    """

    query = None
    if maxcc and data_collection.has_cloud_coverage:
        if isinstance(maxcc, (int, float)) and (maxcc < 0 or maxcc > 1):
            raise ValueError('Maximum cloud coverage "maxcc" parameter\
                             should be a float on an interval [0, 1]')
        query = {'eo:cloud_cover': {'lte': int(maxcc * 100)}}

    fields = {'include': ['properties.datetime'], 'exclude': []}

    catalog = SentinelHubCatalog(base_url=data_collection.service_url, config=config)
    search_iterator = catalog.search(collection=data_collection, bbox=bbox, 
                                     time=time_interval,
                                     query=query, fields=fields)

    all_timestamps = search_iterator.get_timestamps()
    filtered_timestamps = filter_times(all_timestamps, time_difference)

    if len(filtered_timestamps) == 0:
        raise ValueError("No available images for requested time range: {}"\
                         .format(time_interval))
    return filtered_timestamps

class SentinelHubInputBase(EOTask):
    """ 
    Base class for Processing API input tasks
    """

    def __init__(self, data_collection, size=None, resolution=None, 
                 cache_folder=None, config=None, max_threads=None,
                 data_source=None):
        """
        :param data_collection: A collection of requested satellite data.
        :type data_collection: DataCollection
        :param size: Number of pixels in x and y dimension.
        :type size: tuple(int, int)
        :type resolution: Resolution in meters, passed as a single number or a tuple of two numbers -
            resolution in horizontal and resolution in vertical direction.
        :type resolution: float or (float, float)
        :param cache_folder: Path to cache_folder. If set to None (default) 
            requests will not be cached.
        :type cache_folder: str
        :param config: An instance of SHConfig defining the service
        :type config: SHConfig or None
        :param max_threads: Maximum threads to be used when downloading data.
        :type max_threads: int
        :param data_source: A deprecated alternative to data_collection
        :type data_source: DataCollection
        """
        if (size is None) == (resolution is None):
            raise ValueError("Exactly one of the parameters 'size' and\
                             'resolution' should be given.")

        self.size = size
        self.resolution = resolution
        self.config = config or SHConfig()
        self.max_threads = max_threads
        self.data_collection = DataCollection(handle_deprecated_data_source(data_collection, data_source))
        self.cache_folder = cache_folder

    def execute(self, eopatch=None, bbox=None, time_interval=None):
        """ Main execute method for the Process API tasks
        """

        eopatch = eopatch or EOPatch()

        self._check_and_set_eopatch_bbox(bbox, eopatch)
        size_x, size_y = self._get_size(eopatch)

        if time_interval:
            time_interval = parse_time_interval(time_interval)
            timestamp = self._get_timestamp(time_interval, eopatch.bbox)
        else:
            timestamp = eopatch.timestamp

        if eopatch.timestamp and timestamp:
            self.check_timestamp_difference(timestamp, eopatch.timestamp)
        elif timestamp:
            eopatch.timestamp = timestamp

        requests = self._build_requests(eopatch.bbox, size_x, size_y, 
                                        timestamp, time_interval)
        requests = [request.download_list[0] for request in requests]

        LOGGER.debug('Downloading %d requests of type %s', len(requests), 
                     str(self.data_collection))
        client = SentinelHubDownloadClient(config=self.config)
        responses = client.download(requests, max_threads=self.max_threads)
        LOGGER.debug('Downloads complete')

        temporal_dim = len(timestamp) if timestamp else 1
        shape = temporal_dim, size_y, size_x
        self._extract_data(eopatch, responses, shape)

        eopatch.meta_info['size_x'] = size_x
        eopatch.meta_info['size_y'] = size_y
        if timestamp:  # do not overwrite time interval in case of timeless features
            eopatch.meta_info['time_interval'] = time_interval

        self._add_meta_info(eopatch)
        return eopatch

    def _get_size(self, eopatch):
        """
        Get the size (width, height) for the request either from inputs, 
        or from the (existing) eopatch"""
        if self.size is not None:
            return self.size

        if self.resolution is not None:
            return bbox_to_dimensions(eopatch.bbox, self.resolution)

        if eopatch.meta_info and eopatch.meta_info.get('size_x')\
                and eopatch.meta_info.get('size_y'):
            return eopatch.meta_info.get('size_x'), eopatch.meta_info.get('size_y')

        raise ValueError('Size or resolution for the requests should be provided!')

    def _add_meta_info(self, eopatch):
        """
        Add information to eopatch metadata
        """
        if self.maxcc:
            eopatch.meta_info['maxcc'] = self.maxcc
        if self.time_difference:
            eopatch.meta_info['time_difference'] = self.time_difference

    @staticmethod
    def _check_and_set_eopatch_bbox(bbox, eopatch):
        if eopatch.bbox is None:
            if bbox is None:
                raise ValueError('Either the eopatch or the task must\
                                 provide valid bbox.')
            eopatch.bbox = bbox
            return

        if bbox is None or eopatch.bbox == bbox:
            return
        raise ValueError('Either the eopatch or the task must provide bbox,\
                         or they must be the same.')

    @staticmethod
    def check_timestamp_difference(timestamp1, timestamp2):
        """ 
        Raises an error if the two timestamps are not the same
        """
        error_msg = "Trying to write data to an existing eopatch with a different timestamp."
        if len(timestamp1) != len(timestamp2):
            raise ValueError(error_msg)

        for ts1, ts2 in zip(timestamp1, timestamp2):
            if ts1 != ts2:
                raise ValueError(error_msg)

    def _extract_data(self, eopatch, images, shape):
        """ 
        Extract data from the received images and assign them to eopatch features
        """
        raise NotImplementedError("The _extract_data method should be\
                                  implemented by the subclass.")

    def _build_requests(self, bbox, size_x, size_y, timestamp, time_interval):
        """ 
        Build requests
        """
        raise NotImplementedError("The _build_requests method should be\
                                  implemented by the subclass.")

    def _get_timestamp(self, time_interval, bbox):
        """ 
        Get the timestamp array needed as a parameter for downloading the images
        """

ProcApiType = collections.namedtuple('ProcApiType', 
                                     'id unit sample_type np_dtype feature_type')

class SentinelHubEvalscriptTask(SentinelHubInputBase):
    """ 
    Process API task to download data using evalscript
    """
    def __init__(self, features=None, evalscript=None, data_collection=None, 
                 size=None, resolution=None, maxcc=None, time_difference=None, 
                 cache_folder=None, max_threads=None, config=None, 
                 mosaicking_order=None, aux_request_args=None):
        """
        :param features: Features to construct from the evalscript.
        :param evalscript: Evascript for the request. Beware that all outputs 
            from SentinelHub services should be named
            and should have the same name as corresponding feature
        :type evalscript: str
        :param data_collection: Source of requested satellite data.
        :type data_collection: DataCollection
        :param size: Number of pixels in x and y dimension.
        :type size: tuple(int, int)
        :type resolution: Resolution in meters, passed as a single number or a 
            tuple of two numbers -
            resolution in horizontal and resolution in vertical direction.
        :type resolution: float or (float, float)
        :param maxcc: Maximum cloud coverage, a float in interval [0, 1]
        :type maxcc: float
        :param time_difference: Minimum allowed time difference, used when 
            filtering dates, None by default.
        :type time_difference: datetime.timedelta
        :param cache_folder: Path to cache_folder. If set to None (default) 
            requests will not be cached.
        :type cache_folder: str
        :param config: An instance of SHConfig defining the service
        :type config: SHConfig or None
        :param max_threads: Maximum threads to be used when downloading data.
        :type max_threads: int
        :param mosaicking_order: Mosaicking order, which has to be either 
            'mostRecent', 'leastRecent' or 'leastCC'.
        :type mosaicking_order: str
        :param aux_request_args: a dictionary with auxiliary information for 
            the input_data part of the SH request
        :type aux_request_args: dict
        """
        super().__init__(data_collection=data_collection, size=size, 
                         resolution=resolution, cache_folder=cache_folder,
                         config=config, max_threads=max_threads)

        self.features = self._parse_and_validate_features(features)
        self.responses = self._create_response_objects()

        if not evalscript:
            raise ValueError('evalscript parameter must not be missing/empty')
        self.evalscript = evalscript

        if maxcc and isinstance(maxcc, (int, float)) and (maxcc < 0 or maxcc > 1):
            raise ValueError('maxcc should be a float on an interval [0, 1]')

        self.maxcc = maxcc
        self.time_difference = time_difference or dt.timedelta(seconds=1)
        self.mosaicking_order = mosaicking_order
        self.aux_request_args = aux_request_args

    def _parse_and_validate_features(self, features):
        if not features:
            raise ValueError('features must be defined')

        allowed_features = FeatureTypeSet.RASTER_TYPES.union({FeatureType.META_INFO})
        _features = list(self._parse_features(features, 
                                              allowed_feature_types=allowed_features, 
                                              new_names=True)())

        ftr_data_types = set(ft for ft, _, _ in _features if not ft.is_meta())
        if all(ft.is_timeless() for ft in ftr_data_types) or\
            all(ft.is_time_dependent() for ft in ftr_data_types):
            return _features
        raise ValueError('Cannot mix time dependent and timeless requests!')

    def _create_response_objects(self):
        """ 
        Construct SentinelHubRequest output_responses from features
        """
        responses = []
        for feat_type, feat_name, _ in self.features:
            if feat_type.is_raster():
                responses.append(SentinelHubRequest.output_response(feat_name, 
                                                                    MimeType.TIFF))
            elif feat_type.is_meta():
                responses.append(SentinelHubRequest.output_response('userdata', 
                                                                    MimeType.JSON))
            else:
                # should not happen as features have already been validated
                raise ValueError(f'{feat_type} not supported!')
        return responses

    def _get_timestamp(self, time_interval, bbox):
        """ 
        Get the timestamp array needed as a parameter for downloading the images
        """
        if any(feat_type.is_timeless() for feat_type, _, _ in self.features if feat_type.is_raster()):
            return None

        return get_available_timestamps(bbox=bbox, time_interval=time_interval,
                                        data_collection=self.data_collection,
                                        maxcc=self.maxcc, 
                                        time_difference=self.time_difference, 
                                        config=self.config)

    def _build_requests(self, bbox, size_x, size_y, timestamp, time_interval):
        """ 
        Build requests
        """
        if timestamp:
            dates = [(date - self.time_difference, date + self.time_difference)\
                     for date in timestamp]
        else:
            dates = [parse_time_interval(time_interval, allow_undefined=True)]\
                if time_interval else [None]
        return [self._create_sh_request(date, bbox, size_x, size_y) for date in dates]

    def _create_sh_request(self, time_interval, bbox, size_x, size_y):
        """ 
        Create an instance of SentinelHubRequest
        """
        return SentinelHubRequest(
            evalscript=self.evalscript,
            input_data=[SentinelHubRequest.input_data(
                data_collection=self.data_collection,
                mosaicking_order=self.mosaicking_order,
                time_interval=time_interval,
                maxcc=self.maxcc,
                other_args=self.aux_request_args
            )],
            responses=self.responses,
            bbox=bbox,
            size=(size_x, size_y),
            data_folder=self.cache_folder,
            config=self.config
        )

    def _extract_data(self, eopatch, data_responses, shape):
        """ 
        Extract data from the received images and assign them to eopatch features
        """
        if len(self.features) == 1:
            ftype, fname, _ = self.features[0]
            extension = 'json' if ftype.is_meta() else 'tif'
            data_responses = [{f'{fname}.{extension}': data} for data in data_responses]

        for ftype, fname, new_fname in self.features:
            if ftype.is_meta():
                data = [data['userdata.json'] for data in data_responses]

            elif ftype.is_time_dependent():
                data = np.asarray([data[f"{fname}.tif"] for data in data_responses])
                data = data[..., np.newaxis] if data.ndim == 3 else data

            else:
                data = np.asarray(data_responses[0][f"{fname}.tif"])[..., np.newaxis]

            eopatch[ftype][new_fname] = data
        return eopatch



