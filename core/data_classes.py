# -*- coding: utf-8 -*-
"""

@author: freeridingeo
"""

import re
import os
import warnings

from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio

class RasterData(object):
    """ 
    """
    def __init__(self, path, list_separate=True, timestamps=None):
        """
        Constructs a raster image.
        """

        if isinstance(path, str):
            self.path = path if os.path.isabs(path)\
                else os.path.join(os.getcwd(), path)
            path = self.__prependVSIdirective(path)
            with rasterio.open(path, 'r') as src:
                self.src = src
                self.raster = src.read()

        elif isinstance(path, Path):
            print("")
            with rasterio.open(path, 'r') as src:
                self.src = src
                self.raster = src.read()

        # TODO!
        elif isinstance(path, list):
            path = self.__prependVSIdirective(path)
            # self.path = tempfile.NamedTemporaryFile(suffix='.vrt').name
            # self.raster = gdalbuildvrt(src=filename,
            #                            dst=self.filename,
            #                            options={'separate': list_separate},
            #                            void=False)

        self.__data = [None] * self.bands

        # TODO!
        # if self.format == 'ENVI':
        #     with HDRobject(self.filename + '.hdr') as hdr:
        #         if hasattr(hdr, 'band_names'):
        #             self.bandnames = hdr.band_names
        #         else:
        #             self.bandnames = ['band{}'.format(x) for x in range(1, self.bands + 1)]
        # elif self.format == 'VRT':
        #     vrt_tilenames = [os.path.splitext(os.path.basename(x))[0] for x in self.files[1:]]
        #     if len(vrt_tilenames) == self.bands:
        #         self.bandnames = vrt_tilenames
        #     elif self.bands == 1:
        #         self.bandnames = ['mosaic']
        # else:
        #     self.bandnames = ['band{}'.format(x) for x in range(1, self.bands + 1)]
        self.bandnames = ['band{}'.format(x) for x in range(1, self.bands + 1)]

        if timestamps is not None:
            if len(timestamps) != len(self.bandnames):
                raise RuntimeError('the number of time stamps is different to the number of bands')
        self.timestamps = timestamps
    
    def __enter__(self):
        return self

    def __str__(self):
        vals = dict()
        vals['rows'], vals['cols'], vals['bands'] = self.src.dim
        # TOD!
#        vals.update(self.geo)
        vals['crs'] = self.src.crs
#        vals['epsg'] = self.epsg
        vals['filename'] = self.filename if self.filename is not None else 'memory'
        if self.timestamps is not None:
            t0 = min(self.timestamps)
            t1 = max(self.timestamps)
            vals['time'] = 'time range : {0} .. {1}\n'.format(t0, t1)
        else:
            vals['time'] = ''
        
        info = 'class      : Raster object\n' \
               'dimensions : {rows}, {cols}, {bands} (rows, cols, bands)\n' \
               'resolution : {xres}, {yres} (x, y)\n' \
               '{time}' \
               'extent     : {xmin}, {xmax}, {ymin}, {ymax} (xmin, xmax, ymin, ymax)\n' \
               'coord. ref.: {crs} (EPSG:{epsg})\n' \
               'data source: {filename}'.format(**vals)
        
        return info

    def __getitem__(self, index):
        """
        subset the object by slices or vector geometry. 
        If slices are provided, one slice for each raster dimension
        needs to be defined. 
        I.e., if the raster object contains several image bands, three slices 
        are necessary.
        Integer slices are treated as pixel coordinates and float slices 
        as map coordinates.
        If a :class:`~spatialist.vector.Vector` geometry is defined, it is 
        internally projected to the raster CRS if necessary, its extent derived, 
        and the extent converted to raster pixel slices, which are then used for subsetting.
        
        index: :obj:`tuple` of :obj:`slice` or :obj:`~spatialist.vector.Vector`
            the subsetting indices to be used
        
        returns Raster
            a new raster object referenced through an in-memory GDAL VRT file
        
        Examples
        --------
            filename = 'test'
            with RasterData(filename) as Raster:
                print(Raster)
        
            xmin = 0
            xmax = 100
            ymin = 4068985.595
            ymax = 4088985.595
            with RasterData(filename)[ymin:ymax, xmin:xmax, :] as Raster:
                print(Raster)

            ext = {'xmin': 713315.198, 'xmax': 715315.198, 'ymin': ymin, 'ymax': ymax}
            with bbox(ext, crs=32629) as vec:
                with RasterData(filename)[vec] as Raster:
                    print(Raster)
        """
        # TODO!
        # subsetting via Vector object
        # if isinstance(index, Vector):
        #     geomtypes = list(set(index.geomTypes))
        #     if len(geomtypes) != 1:
        #         raise RuntimeError('Raster subsetting is only supported for Vector objects with one type of geometry')
        #     geomtype = geomtypes[0]
        #     if geomtype == 'POLYGON':
        #         # intersect bounding boxes of vector and raster object
        #         inter = intersect(index.bbox(), self.bbox())
        #         if inter is None:
        #             raise RuntimeError('no intersection between Raster and Vector object')
        #     else:
        #         raise RuntimeError('Raster subsetting is only supported for POLYGON geometries')
        #     # get raster indexing slices from intersect bounding box extent
        #     sl = self.__extent2slice(inter.extent)
        #     # subset raster object with slices
        #     with self[sl] as sub:
        #         # mask subsetted raster object with vector geometries
        #         masked = sub.__maskbyvector(inter)
        #     inter = None
        #     return masked

        ######################################################################
        # subsetting via slices        

        if isinstance(index, tuple):
            ras_dim = 2 if self.src.count == 1 else 3
            if ras_dim != len(index):
                raise IndexError(
                    'mismatch of index length ({0}) and raster dimensions ({1})'\
                        .format(len(index), ras_dim))
            for i in [0, 1]:
                if hasattr(index[i], 'step') and index[i].step is not None:
                    raise IndexError('step slicing of {} is not allowed'\
                                     .format(['rows', 'cols'][i]))
            index = list(index) 

        # treat float indices as map coordinates and convert them to image coordinates
        yi = index[0]
        if isinstance(yi, float):
            # TODO!
            yi = self.coord_map2img(y=yi)
        if isinstance(yi, slice):
            if isinstance(yi.start, float) or isinstance(yi.stop, float):
                start = None if yi.stop is None else self.coord_map2img(y=yi.stop)
                stop = None if yi.start is None else self.coord_map2img(y=yi.start)
                yi = slice(start, stop)
        
        xi = index[1]
        if isinstance(xi, float):
            # TODO!
            xi = self.coord_map2img(x=xi)
        if isinstance(xi, slice):
            if isinstance(xi.start, float) or isinstance(xi.stop, float):
                start = None if xi.start is None else self.coord_map2img(x=xi.start)
                stop = None if xi.stop is None else self.coord_map2img(x=xi.stop)
                xi = slice(start, stop)

        # create index lists from subset slices
        subset = dict()
        subset['rows'] = list(range(0, self.rows))[yi]
        subset['cols'] = list(range(0, self.cols))[xi]
        for key in ['rows', 'cols']:
            if not isinstance(subset[key], list):
                subset[key] = [subset[key]]
        if len(index) > 2:
            bi = index[2]
            if isinstance(bi, str):
                bi = self.bandnames.index(bi)
            elif isinstance(bi, datetime):
                bi = self.timestamps.index(bi)
            elif isinstance(bi, int):
                pass
            elif isinstance(bi, slice):
                if isinstance(bi.start, int):
                    start = bi.start
                elif isinstance(bi.start, str):
                    start = self.bandnames.index(bi.start)
                elif isinstance(bi.start, datetime):
                    larger = [x for x in self.timestamps if x > bi.start]
                    tdiff = [x - bi.start for x in larger]
                    closest = larger[tdiff.index(min(tdiff))]
                    start = self.timestamps.index(closest)
                elif bi.start is None:
                    start = None
                else:
                    raise TypeError('band indices must be either int or str')
                if isinstance(bi.stop, int):
                    stop = bi.stop
                elif isinstance(bi.stop, str):
                    stop = self.bandnames.index(bi.stop)
                elif isinstance(bi.stop, datetime):
                    smaller = [x for x in self.timestamps if x < bi.stop]
                    tdiff = [bi.start - x for x in smaller]
                    closest = smaller[tdiff.index(min(tdiff))]
                    stop = self.timestamps.index(closest)
                elif bi.stop is None:
                    stop = None
                else:
                    raise TypeError('band indices must be either int or str')
                bi = slice(start, stop)
            else:
                raise TypeError('band indices must be either int or str')
            index[2] = bi
            subset['bands'] = list(range(0, self.bands))[index[2]]
            if not isinstance(subset['bands'], list):
                subset['bands'] = [subset['bands']]
        else:
            subset['bands'] = [0]

        if len(subset['rows']) == 0 or len(subset['cols']) == 0 or len(subset['bands']) == 0:
            raise RuntimeError('no suitable subset for defined slice:\n  {}'.format(index))
        
        # TODO!
        # update geo dimensions from subset list indices
        geo = self.geo
        geo['xmin'] = self.coord_img2map(x=min(subset['cols']))
        geo['ymax'] = self.coord_img2map(y=min(subset['rows']))
        
        geo['xmax'] = self.coord_img2map(x=max(subset['cols']) + 1)
        geo['ymin'] = self.coord_img2map(y=max(subset['rows']) + 1)

        # TODO!
        # # options for creating a GDAL VRT data set
        # opts = dict()
        # opts['xRes'], opts['yRes'] = self.res
        # opts['outputSRS'] = self.projection
        # opts['srcNodata'] = self.nodata
        # opts['VRTNodata'] = self.nodata
        # opts['bandList'] = [x + 1 for x in subset['bands']]
        # opts['outputBounds'] = (geo['xmin'], geo['ymin'], geo['xmax'], geo['ymax'])

        # create an in-memory VRT file and return the output raster dataset as new Raster object
        # outname = os.path.join('/vsimem/', os.path.basename(tempfile.mktemp()))
#        outname = tempfile.NamedTemporaryFile(suffix='.vrt').name
#        out_ds = gdalbuildvrt(src=self.filename, dst=outname, options=opts, void=False)

        timestamps = self.timestamps
        if len(index) > 2:
            bandnames = self.bandnames[index[2]]
            if timestamps is not None:
                timestamps = timestamps[index[2]]
        else:
            bandnames = self.bandnames
        if not isinstance(bandnames, list):
            bandnames = [bandnames]
        if timestamps is not None and not isinstance(timestamps, list):
            timestamps = [timestamps]
 
        # TODO!
#        out = RasterData(out_ds)
#        out.bandnames = bandnames
#        out.timestamps = timestamps
#        return out

    # TODO!
    # def __extent2slice(self, extent):
    #     extent_bbox = bbox(extent, self.proj4)
    #     inter = intersect(self.bbox(), extent_bbox)
    #     extent_bbox.close()
    #     if inter:
    #         ext_inter = inter.extent
    #         ext_ras = self.geo
    #         xres, yres = self.res
    #         tolerance_x = xres * subset_tolerance / 100
    #         tolerance_y = yres * subset_tolerance / 100
    #         colmin = int(floor((ext_inter['xmin'] - ext_ras['xmin'] + tolerance_x) / xres))
    #         colmax = int(ceil((ext_inter['xmax'] - ext_ras['xmin'] - tolerance_x) / xres))
    #         rowmin = int(floor((ext_ras['ymax'] - ext_inter['ymax'] + tolerance_y) / yres))
    #         rowmax = int(ceil((ext_ras['ymax'] - ext_inter['ymin'] - tolerance_y) / yres))
    #         inter.close()
    #         if self.bands == 1:
    #             return slice(rowmin, rowmax), slice(colmin, colmax)
    #         else:
    #             return slice(rowmin, rowmax), slice(colmin, colmax), slice(0, self.bands)
    #     else:
    #         raise RuntimeError('extent does not overlap with raster object')

    def __maskbyvector(self, vec, outname=None, format='GTiff', nodata=0):
        
        if outname is not None:
            driver_name = format
        else:
            driver_name = 'MEM'

        # TODO!
#        with rasterize(vec, self) as vecmask:
#            mask = vecmask.matrix()

        driver = self.src.driver
        meta = self.src.meta
        
        if not outname:
            outname = 'vectormasked_raster'
        
        # TODO!
        with rasterio.open(outname, 'w', **meta) as dst:
            dst.write()

    def __prependVSIdirective(self, filename):
        """
        prepend one of /vsizip/ or /vsitar/ to the file name if a zip of tar archive.
        
        filename: str or list
            the file name, e.g. archive.tar.gz/filename
        """
        if isinstance(filename, str):
            if re.search(r'\.zip', filename):
                filename = '/vsizip/' + filename
            if re.search(r'\.tar', filename):
                filename = '/vsitar/' + filename
        elif isinstance(filename, list):
            filename = [self.__prependVSIdirective(x) for x in filename]
        return filename

    # TODO
    def allstats(self, approximate=False):
        """
        Compute some basic raster statistics

        approximate: bool
            approximate statistics from overviews or a subset of all tiles?

        returns list of dicts
            a list with a dictionary of statistics for each band. Keys: `min`, `max`, `mean`, `sdev`.
            See :osgeo:meth:`gdal.Band.ComputeStatistics`.
        """
        # statcollect = []
        # for x in self.layers():
        #     try:
        #         stats = x.ComputeStatistics(approximate)
        #     except RuntimeError:
        #         stats = None
        #     stats = dict(zip(['min', 'max', 'mean', 'sdev'], stats))
        #     statcollect.append(stats)
        # return statcollect
    
    def array(self):
        """
        read all raster bands into a numpy ndarray

        returns    numpy.ndarray
            the array containing all raster data
        """
        if self.bands == 1:
            return self.matrix()
        else:
            if isinstance(self.nodata, list):
                for i in range(0, self.bands):
                    self.raster[:, :, i][self.raster[:, :, i] == self.nodata[i]] = np.nan
            else:
                self.raster[self.raster == self.nodata] = np.nan
            return np.squeeze(self.raster)

    def assign(self, array, band):
        """
        assign an array to an existing Raster object

        array: numpy.ndarray
            the array to be assigned to the Raster object
        band: int
            the index of the band to assign to
        """
        self.__data[band] = array
    
    @property
    def bands(self):
        """
        Returns the number of image bands
        """
        return self.src.count

    @property
    def bandnames(self):
        return self.__bandnames

    @bandnames.setter
    def bandnames(self, names):
        """
        set the names of the raster bands
        
        names: list of str
            the names to be set; must be of same length as the number of bands
        """
        if not isinstance(names, list):
            raise TypeError('the names to be set must be of type list')
        if len(names) != self.bands:
            raise ValueError(
                'length mismatch of names to be set ({}) and number of bands ({})'\
                    .format(len(names), self.bands))
        self.__bandnames = names


    def _apply_crs_and_affine(self, params):
        """
        Applies any CRS and affine parameters to a raster.
        """
    
    def _apply_selection_and_scale(self, params, dimensions_consumed):
        """
        Applies region selection and scaling parameters to a raster.
        """

    def _apply_spatial_transformations(self, params):
        """
        Applies spatial transformation and clipping.
        """
    def _apply_visualization(self, params):
        """
        Applies visualization parameters to an image.
        """

