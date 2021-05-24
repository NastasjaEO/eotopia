# -*- coding: utf-8 -*-
"""
See
https://github.com/johntruckenbrodt/spatialist/blob/master/spatialist/raster.py

@author: freeridingeo
"""

import re
import os
import warnings

from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import (Point, MultiPoint, 
                              LineString, MultiLineString,
                              Polygon, MultiPolygon)

import sys
sys.path.append("D:/Code/eotopia/utils")
from raster_utils import rasterize
from list_utils import dissolve
from string_utils import parse_literal
sys.path.append("D:/Code/eotopia/multiprocessing")
from multiprocessing_utils import multicore, parallel_apply_along_axis


class RasterData(object):

    def __init__(self, path, list_separate=True, timestamps=None):

        if isinstance(path, str):
            self.path = path if os.path.isabs(path)\
                else os.path.join(os.getcwd(), path)
            path = self.__prependVSIdirective(path)
            with rasterio.open(path, 'r') as src:
                self.src = src
                self.trafo = src.transform

        elif isinstance(path, Path):
            print("")
            with rasterio.open(path, 'r') as src:
                self.src = src
                self.trafo = src.transform

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
        vals.update(self.geo)
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
            yi = self.coord_map2img(y=yi)
        if isinstance(yi, slice):
            if isinstance(yi.start, float) or isinstance(yi.stop, float):
                start = None if yi.stop is None else self.coord_map2img(y=yi.stop)
                stop = None if yi.start is None else self.coord_map2img(y=yi.start)
                yi = slice(start, stop)
        
        xi = index[1]
        if isinstance(xi, float):
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
        geo = self.trafo
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
    #         ext_ras = self.src.transform
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

    def bbox(self, outname=None, driver='ESRI Shapefile', overwrite=True, 
             source='image'):
        """
        Parameters
        ----------
        outname: str or None
            the name of the file to write; If `None`, the bounding box is returned
            as :class:`~spatialist.vector.Vector` object
        driver: str
            The file format to write
        overwrite: bool
            overwrite an already existing file?
        source: {'image', 'gcp'}
            get the bounding box of either the image or the ground control points
        Returns
        -------
        Vector or None
            the bounding box vector object
        """
        bbox = self.src.bounds
        crs = self.src.crs
        extent = self.extent
        if outname is None:
            return bbox(coordinates=extent, crs=crs)
        else:
            bbox(coordinates=extent, crs=crs, outname=outname,
                 driver=driver, overwrite=overwrite)

    @property
    def cols(self):
        """
        the number of image columns
        """
        return self.src.cols

    @property
    def rows(self):
        """
        the number of image rows
        """
        return self.src.rows

    def coord_map2img(self, x=None, y=None):
        """
        convert map coordinates in the raster CRS to image pixel coordinates.
        Either x, y or both must be defined.
        
        Parameters
        ----------
        x: int or float
            the x coordinate
        y: int or float
            the y coordinate
        Returns
        -------
        int or tuple
            the converted coordinate for either x, y or both
        """
        if x is None and y is None:
            raise TypeError("both 'x' and 'y' cannot be None")
        out = []
        if x is not None:
            out.append(int((x - self.geo['xmin']) / self.geo['xres']))
        if y is not None:
            out.append(int((self.geo['ymax'] - y) / abs(self.geo['yres'])))
        return tuple(out) if len(out) > 1 else out[0]

    def coord_img2map(self, x=None, y=None):
        """
        convert image pixel coordinates to map coordinates in the raster CRS.
        Either x, y or both must be defined.
        
        Parameters
        ----------
        x: int or float
            the x coordinate
        y: int or float
            the y coordinate
        Returns
        -------
        float or tuple
            the converted coordinate for either x, y or both
        """
        if x is None and y is None:
            raise TypeError("both 'x' and 'y' cannot be None")
        out = []
        if x is not None:
            out.append(self.geo['xmin'] + self.geo['xres'] * x)
        if y is not None:
            out.append(self.geo['ymax'] - abs(self.geo['yres']) * y)
        return tuple(out) if len(out) > 1 else out[0]

    @property
    def dim(self):
        """
        tuple: (rows, columns, bands)
        """
        return (self.rows, self.cols, self.bands)

    @property
    def driver(self):
        return self.src.driver

    @property
    def dtype(self):
        return self.src.dtype

    @property
    def crs(self):
        return self.src.crs

    @property
    def extent(self):
        return {key: self.geo[key] for key in ['xmin', 'xmax', 'ymin', 'ymax']}

    def extract_weighted_average(self, px, py, radius=1, nodata=None):
        """
        extract weighted average of pixels intersecting with a 
        defined radius to a point.
        Parameters
        ----------
        px: int or float
            the x coordinate in units of the Raster SRS
        py: int or float
            the y coordinate in units of the Raster SRS
        radius: int or float
            the radius around the point to extract pixel values from; defined as multiples of the pixel resolution
        nodata: int
            a value to ignore from the computations; If `None`, the nodata value of the Raster object is used
        Returns
        -------
        int or float
            the the weighted average of all pixels within the defined radius
        """
        if not self.geo['xmin'] <= px <= self.geo['xmax']:
            raise RuntimeError('px is out of bounds')
        if not self.geo['ymin'] <= py <= self.geo['ymax']:
            raise RuntimeError('py is out of bounds')

        if nodata is None:
            nodata = self.src.nodata
        
        xres, yres = self.res
        hx = xres / 2.0
        hy = yres / 2.0
        
        xlim = float(xres * radius)
        ylim = float(yres * radius)

        # compute minimum x and y pixel coordinates
        xmin = int(np.floor((px - self.geo['xmin'] - xlim) / xres))
        ymin = int(np.floor((self.geo['ymax'] - py - ylim) / yres))
        xmin = xmin if xmin >= 0 else 0
        ymin = ymin if ymin >= 0 else 0
        
        # compute maximum x and y pixel coordinates
        xmax = int(np.ceil((px - self.geo['xmin'] + xlim) / xres))
        ymax = int(np.ceil((self.geo['ymax'] - py + ylim) / yres))
        xmax = xmax if xmax <= self.cols else self.cols
        ymax = ymax if ymax <= self.rows else self.rows

        if self.__data[0] is not None:
            array = self.__data[0][ymin:ymax, xmin:xmax]
        else:
            # TODO!
            array = self.raster.GetRasterBand(1).ReadAsArray(xmin, ymin, xmax - xmin, ymax - ymin)
        
        sum = 0
        counter = 0
        weightsum = 0
        for x in range(xmin, xmax):
            for y in range(ymin, ymax):
                # check whether point is a valid image index
                val = array[y - ymin, x - xmin]
                if val != nodata:
                    # compute distances of pixel center coordinate to requested point
                    
                    xc = x * xres + hx + self.geo['xmin']
                    yc = self.geo['ymax'] - y * yres + hy                    
                    dx = abs(xc - px)
                    dy = abs(yc - py)
                    
                    # check whether point lies within ellipse: 
                        # if ((dx ** 2) / xlim ** 2) + ((dy ** 2) / ylim ** 2) <= 1
                    weight = np.sqrt(dx ** 2 + dy ** 2)
                    sum += val * weight
                    weightsum += weight
                    counter += 1
        array = None
        if counter > 0:
            return sum / weightsum
        else:
            return nodata
    
    # TODO!
    @property
    def files(self):
        """
        list of all absolute names of files associated with this raster data set
        """
        # fl = self.raster.GetFileList()
        # if fl is not None:
        #     return [os.path.abspath(x) for x in fl]

    @property
    def format(self):
        return self.src.driver#.ShortName

    @property
    def geo(self):
        """
        General image geo information.
        Returns
        -------
        dict
            a dictionary with keys `xmin`, `xmax`, `xres`, `rotation_x`, 
            `ymin`, `ymax`, `yres`, `rotation_y`
        """
        out = dict(zip(['xmin', 'xres', 'rotation_x', 'ymax', 'rotation_y', 'yres'],
                       self.src.transform))
        
        # note: yres is negative!
        out['xmax'] = out['xmin'] + out['xres'] * self.cols
        out['ymin'] = out['ymax'] + out['yres'] * self.rows
        return out
    
    def is_valid(self):
        """
        Check image integrity.
        Tries to compute the checksum for each raster layer and returns False if this fails.
        See this forum entry:
        `How to check if image is valid? <https://lists.osgeo.org/pipermail/gdal-dev/2013-November/037520.html>`_.
        Returns
        -------
        bool
            is the file valid?
        """
        checksum = 0
        for i in range(self.src.count):
            try:
                checksum += self.src.read(i + 1)#.Checksum()
            except RuntimeError:
                return False
        return True

    def load(self):
        """
        load all raster data to internal memory arrays.
        This shortens the read time of other methods like :meth:`matrix`.
        """
        for i in range(1, self.bands + 1):
            self.__data[i - 1] = self.matrix(i)

    def matrix(self, band=1, mask_nan=True):
        """
        read a raster band (subset) into a numpy ndarray
        Parameters
        ----------
        band: int
            the band to read the matrix from; 1-based indexing
        mask_nan: bool
            convert nodata values to :obj:`numpy.nan`? As :obj:`numpy.nan` requires at least float values, any integer array is cast
            to float32.
        Returns
        -------
        numpy.ndarray
            the matrix (subset) of the selected band
        """
        mat = self.__data[band - 1]
        if mat is None:
            mat = self.src.read(band)
            if mask_nan:
                if isinstance(self.nodata, list):
                    nodata = self.nodata[band - 1]
                else:
                    nodata = self.nodata
                mat[mat == nodata] = np.nan
        return mat

    @property
    def nodata(self):
        """
        float or list ofthe raster nodata value(s)
        """
        # nodatas = [self.raster.GetRasterBand(i).GetNoDataValue()
        #            for i in range(1, self.bands + 1)]
        # if len(list(set(nodatas))) == 1:
        #     return nodatas[0]
        # else:
        #     return nodatas

    @property
    def res(self):
        """
        the raster resolution in x and y direction
        """
        return (abs(float(self.geo['xres'])), abs(float(self.geo['yres'])))

    def rescale(self, fun):
        """
        perform raster computations with custom functions and assign them to the 
        existing raster object in memory
        
        Parameters
        ----------
        fun: function
            the custom function to compute on the data
        
        Examples
        --------
        >>> with RasterData('filename') as Raster:
        >>>     Raster.rescale(lambda x: 10 * x)
        """
        if self.bands != 1:
            raise ValueError('only single band images are currently supported')

        mat = self.matrix()
        scaled = fun(mat)
        self.assign(scaled, band=0)

    def write(self, outname, dtype='default', format='ENVI', nodata='default', compress_tif=False, overwrite=False,
              cmap=None, update=False, xoff=0, yoff=0, array=None):
        """
        write the raster object to a file.
        Parameters
        ----------
        outname: str
            the file to be written
        dtype: str
            the data type of the written file;
            data type notations of GDAL (e.g. `Float32`) and numpy (e.g. `int8`) are supported.
        format: str
            the file format; e.g. 'GTiff'
        nodata: int or float
            the nodata value to write to the file
        compress_tif: bool
            if the format is GeoTiff, compress the written file?
        overwrite: bool
            overwrite an already existing file? Only applies if `update` is `False`.
        cmap: :osgeo:class:`gdal.ColorTable`
            a color map to apply to each band.
            Can for example be created with function :func:`~spatialist.auxil.cmap_mpl2gdal`.
        update: bool
            open the output file fpr update or only for writing?
        xoff: int
            the x/column offset
        yoff: int
            the y/row offset
        array: numpy.ndarray
            write different data than that associated with the Raster object
        Returns
        -------
        """
        update_existing = update and os.path.isfile(outname)
        dtype = self.src.dtype
        if not update_existing:
            if os.path.isfile(outname) and not overwrite:
                raise RuntimeError('target file already exists')
            
            if format == 'GTiff' and not re.search(r'\.tif[f]*$', outname):
                outname += '.tif'
            
            # TODO!
            ## update meta
            options = []
            if format == 'GTiff' and compress_tif:
                options += ['COMPRESS=DEFLATE', 'PREDICTOR=2']
            
            # TODO!
            with rasterio.open(outname, "w", **self.src.meta) as dst:                
                dst.write()

    def _apply_visualization(self, params):
        """
        Applies visualization parameters to an image.
        """

def rasterize(vectorobject, reference, outname=None, burn_values=1, 
              expressions=None, nodata=0, append=False):
    """
    rasterize a vector object
    Parameters
    ----------
    vectorobject: Vector
        the vector object to be rasterized
    reference: Raster
        a reference Raster object to retrieve geo information and extent from
    outname: str or None
        the name of the GeoTiff output file; if None, an in-memory object of type :class:`Raster` is returned and
        parameter outname is ignored
    burn_values: int or list
        the values to be written to the raster file
    expressions: list
        SQL expressions to filter the vector object by attributes
    nodata: int
        the nodata value of the target raster file
    append: bool
        if the output file already exists, update this file with new rasterized values?
        If True and the output file exists, parameters `reference` and `nodata` are ignored.
    Returns
    -------
    Raster or None
        if outname is `None`, a raster object pointing to an in-memory dataset else `None`
    Example
    -------
    >>> outname1 = 'target1.tif'
    >>> outname2 = 'target2.tif'
    >>> with Vector('source.shp') as vec:
    >>>     with Raster('reference.tif') as ref:
    >>>         burn_values = [1, 2]
    >>>         expressions = ['ATTRIBUTE=1', 'ATTRIBUTE=2']
    >>>         rasterize(vec, reference, outname1, burn_values, expressions)
    >>>         expressions = ["ATTRIBUTE2='a'", "ATTRIBUTE2='b'"]
    >>>         rasterize(vec, reference, outname2, burn_values, expressions)
    """
    if expressions is None:
        expressions = ['']
    if isinstance(burn_values, (int, float)):
        burn_values = [burn_values]
    if len(expressions) != len(burn_values):
        raise RuntimeError('expressions and burn_values of different length')

    failed = []
    for exp in expressions:
        try:
            vectorobject.layer.SetAttributeFilter(exp)
        except RuntimeError:
            failed.append(exp)
    if len(failed) > 0:
        raise RuntimeError('failed to set the following attribute' 
                           'filter(s): ["{}"]'.format('", '.join(failed)))

    if append and outname is not None and os.path.isfile(outname):
        target_ds = rasterio.open(outname)
    # TODO!
    # else:
    #     if outname is not None:
    #         target_ds = gdal.GetDriverByName('GTiff').Create(outname, reference.cols, reference.rows, 1, gdal.GDT_Byte)
    #     else:
    #         target_ds = gdal.GetDriverByName('MEM').Create('', reference.cols, reference.rows, 1, gdal.GDT_Byte)
    #     target_ds.SetGeoTransform(reference.raster.GetGeoTransform())
    #     target_ds.SetProjection(reference.raster.GetProjection())
    #     band = target_ds.GetRasterBand(1)
    #     band.SetNoDataValue(nodata)
    for expression, value in zip(expressions, burn_values):
        vectorobject.layer.SetAttributeFilter(expression)
    #     gdal.RasterizeLayer(target_ds, [1], vectorobject.layer, burn_values=[value])
    # vectorobject.layer.SetAttributeFilter('')
    # if outname is None:
    #     return Raster(target_ds)
    # else:
    #     target_ds = None


def reproject(rasterobject, reference, outname, targetres=None, 
              resampling='bilinear', format='GTiff'):
    """
    reproject a raster file
    Parameters
    ----------
    rasterobject: Raster or str
        the raster image to be reprojected
    reference: Raster, Vector, str, int or osr.SpatialReference
        either a projection string or a spatial object with an attribute 'projection'
    outname: str
        the name of the output file
    targetres: tuple
        the output resolution in the target SRS; a two-entry tuple is required: (xres, yres)
    resampling: str
        the resampling algorithm to be used
    format: str
        the output file format
    Returns
    -------
    """
    if isinstance(rasterobject, str):
        rasterobject = RasterData(rasterobject)

    if not isinstance(rasterobject, RasterData):
        raise RuntimeError('rasterobject must be of type Raster or str')
    # TODO!
    # if isinstance(reference, (Raster, Vector)):
    #     projection = reference.projection
    #     if targetres is not None:
    #         xres, yres = targetres
    #     elif hasattr(reference, 'res'):
    #         xres, yres = reference.res
    #     else:
    #         raise RuntimeError('parameter targetres is missing and cannot be read from the reference')
    # elif isinstance(reference, (int, str, osr.SpatialReference)):
    #     try:
    #         projection = crsConvert(reference, 'proj4')
    #     except TypeError:
    #         raise RuntimeError('reference projection cannot be read')
    #     if targetres is None:
    #         raise RuntimeError('parameter targetres is missing and cannot be read from the reference')
    #     else:
    #         xres, yres = targetres
    else:
        raise TypeError('reference must be of type Raster, Vector, osr.SpatialReference, str or int')
    # TODO
    # options = {'format': format,
    #            'resampleAlg': resampling,
    #            'xRes': xres,
    #            'yRes': yres,
    #            'srcNodata': rasterobject.nodata,
    #            'dstNodata': rasterobject.nodata,
    #            'dstSRS': projection}
    # gdalwarp(rasterobject.raster, outname, options)

def stack(srcfiles, dstfile, resampling, targetres, dstnodata, srcnodata=None, shapefile=None, layernames=None,
          sortfun=None, separate=False, overwrite=False, compress=True, cores=4, pbar=False):
    """
    function for mosaicking, resampling and stacking of multiple raster files
    Parameters
    ----------
    srcfiles: list
        a list of file names or a list of lists; each sub-list is treated as a task to mosaic its containing files
    dstfile: str
        the destination file or a directory (if `separate` is True)
    resampling: {near, bilinear, cubic, cubicspline, lanczos, average, mode, max, min, med, Q1, Q3}
        the resampling method; see `documentation of gdalwarp <https://www.gdal.org/gdalwarp.html>`_.
    targetres: tuple or list
        two entries for x and y spatial resolution in units of the source CRS
    srcnodata: int, float or None
        the nodata value of the source files; if left at the default (None), the nodata values are read from the files
    dstnodata: int or float
        the nodata value of the destination file(s)
    shapefile: str, Vector or None
        a shapefile for defining the spatial extent of the destination files
    layernames: list
        the names of the output layers; if `None`, the basenames of the input files are used; overrides sortfun
    sortfun: function
        a function for sorting the input files; not used if layernames is not None.
        This is first used for sorting the items in each sub-list of srcfiles;
        the basename of the first item in a sub-list will then be used as the name for the mosaic of this group.
        After mosaicing, the function is again used for sorting the names in the final output
        (only relevant if `separate` is False)
    separate: bool
        should the files be written to a single raster stack (ENVI format) or separate files (GTiff format)?
    overwrite: bool
        overwrite the file if it already exists?
    compress: bool
        compress the geotiff files?
    cores: int
        the number of CPU threads to use
    pbar: bool
        add a progressbar? This is currently only used if `separate==False`
    Returns
    -------
    
    Raises
    ------
    RuntimeError
    Notes
    -----
    This function does not reproject any raster files. Thus, the CRS must be the same for all input raster files.
    This is checked prior to executing gdalwarp. In case a shapefile is defined, it is internally reprojected to the
    raster CRS prior to retrieving its extent.
    Examples
    --------
    .. code-block:: python
        from pyroSAR.ancillary import groupbyTime, find_datasets, seconds
        from spatialist.raster import stack
        # find pyroSAR files by metadata attributes
        archive_s1 = '/.../sentinel1/GRD/processed'
        scenes_s1 = find_datasets(archive_s1, sensor=('S1A', 'S1B'), acquisition_mode='IW')
        # group images by acquisition time
        groups = groupbyTime(images=scenes_s1, function=seconds, time=30)
        # mosaic individual groups and stack the mosaics to a single ENVI file
        # only files overlapping with the shapefile are selected and resampled to its extent
        stack(srcfiles=groups, dstfile='stack', resampling='bilinear', targetres=(20, 20),
              srcnodata=-99, dstnodata=-99, shapefile='site.shp', separate=False)
    """
    srcfiles = srcfiles.copy()

    if len(dissolve(srcfiles)) == 0:
        raise RuntimeError('no input files provided to function raster.stack')
    
    if layernames is not None:
        if len(layernames) != len(srcfiles):
            raise RuntimeError('mismatch between number of source file' 
                               'groups and layernames')

    if not isinstance(targetres, (list, tuple)) or len(targetres) != 2:
        raise RuntimeError('targetres must be a list or tuple with two entries' 
                           'for x and y resolution')
    
    if len(srcfiles) == 1 and not isinstance(srcfiles[0], list):
        raise RuntimeError('only one file specified; nothing to be done')
    
    if resampling not in ['near', 'bilinear', 'cubic', 'cubicspline', 'lanczos',
                          'average', 'mode', 'max', 'min', 'med', 'Q1', 'Q3']:
        raise RuntimeError('resampling method not supported')
    
    if os.path.isfile(dstfile) and not separate and not overwrite:
        raise RuntimeError('the output file already exists')
    
    if not separate and os.path.isdir(dstfile):
        raise RuntimeError("dstfile is an existing directory, cannot write stack with same name")

    ##########################################################################################
    # check if the projection can be read from all images and whether all share the same projection
    crslist = list()
    for x in dissolve(srcfiles):
        try:
            crs = RasterData(x).crs
        except RuntimeError as e:
            print('cannot read file: {}'.format(x))
            raise e
        crslist.append(crs)

    projections = list(set(crslist))
    if len(projections) > 1:
        raise RuntimeError('raster projection mismatch')
    elif projections[0] == '':
        raise RuntimeError('could not retrieve the projection from' 
                           'any of the {} input images'.format(len(srcfiles)))
    else:
        srs = projections[0]
    
    # TODO!
    ##########################################################################################
    # read shapefile bounding coordinates and reduce list of images to those overlapping with the shapefile
    
    # if shapefile is not None:
    #     shp = shapefile.clone() if isinstance(shapefile, Vector) else Vector(shapefile)
    #     shp.reproject(srs)
    #     ext = shp.extent
    #     arg_ext = (ext['xmin'], ext['ymin'], ext['xmax'], ext['ymax'])
    #     for i, item in enumerate(srcfiles):
    #         group = item if isinstance(item, list) else [item]
    #         if layernames is None and sortfun is not None:
    #             group = sorted(group, key=sortfun)
    #         group = [x for x in group if intersect(shp, Raster(x).bbox())]
    #         if len(group) > 1:
    #             srcfiles[i] = group
    #         elif len(group) == 1:
    #             srcfiles[i] = group[0]
    #         else:
    #             srcfiles[i] = None
    #     shp.close()
    #     srcfiles = list(filter(None, srcfiles))
    #     log.debug('number of scenes after spatial filtering: {}'.format(len(srcfiles)))

#    else:
#        arg_ext = None
    ##########################################################################################
    # set general options and parametrization
    
    dst_base = os.path.splitext(dstfile)[0]
    
    # options_warp = {'options': ['-q'],
    #                 'format': 'GTiff' if separate else 'ENVI',
    #                 'outputBounds': arg_ext, 'multithread': True,
    #                 'dstNodata': dstnodata,
    #                 'xRes': targetres[0], 'yRes': targetres[1],
    #                 'resampleAlg': resampling}
    
    # if overwrite:
    #     options_warp['options'] += ['-overwrite']

    # if separate and compress:
    #     options_warp['options'] += ['-co', 'COMPRESS=DEFLATE', '-co', 'PREDICTOR=2']
    
    # options_buildvrt = {'outputBounds': arg_ext}
    
    # if srcnodata is not None:
    #     options_warp['srcNodata'] = srcnodata
    #     options_buildvrt['srcNodata'] = srcnodata
    ##########################################################################################
    # create VRT files for mosaicing
    # the resulting list srcfiles will contain either a single image or a newly created VRT file
    # and thus each list item is one time step in the final stack

    for i, group in enumerate(srcfiles):
        if isinstance(group, list):
            if len(group) > 1:
                base = group[0]
                # in-memory VRT files cannot be shared between multiple processes on Windows
                # this has to do with different process forking behaviour
                # see function spatialist.ancillary.multicore and this link:
                # https://stackoverflow.com/questions/38236211/why-multiprocessing-process-behave-differently-on-windows-and-linux-for-global-o
                vrt_base = os.path.splitext(os.path.basename(base))[0] + '.vrt'
#                if platform.system() == 'Windows':
#                    vrt = os.path.join(tempfile.gettempdir(), vrt_base)
#                else:
#                    vrt = '/vsimem/' + vrt_base
#                gdalbuildvrt(group, vrt, options_buildvrt)
 #               srcfiles[i] = vrt
            else:
                srcfiles[i] = group[0]
        else:
            srcfiles[i] = group

    ##########################################################################################
    # define the output band names
    
    # if no specific layernames are defined, sort files by custom function
    if layernames is None and sortfun is not None:
        srcfiles = sorted(srcfiles, key=sortfun)
    
    # use the file basenames without extension as band names if none are defined
    bandnames = [os.path.splitext(os.path.basename(x))[0]\
                 for x in srcfiles] if layernames is None else layernames
    
    if len(list(set(bandnames))) != len(bandnames):
        raise RuntimeError('output bandnames are not unique')
    ##########################################################################################
    # create the actual image files

    if separate:
        if not os.path.isdir(dstfile):
            os.makedirs(dstfile)
        dstfiles = [os.path.join(dstfile, x) + '.tif' for x in bandnames]
        jobs = [x for x in zip(srcfiles, dstfiles)]
        if not overwrite:
            jobs = [x for x in jobs if not os.path.isfile(x[1])]
            if len(jobs) == 0:
                print('all target tiff files already exist, nothing to be done')
                return
            # TODO!
    #     srcfiles, dstfiles = map(list, zip(*jobs))
    #     log.debug('creating {} separate file(s):\n  {}'.format(len(dstfiles), '\n  '.join(dstfiles)))
    #     multicore(gdalwarp, cores=cores, options=options_warp,
    #               multiargs={'src': srcfiles, 'dst': dstfiles})
    # else:
    #     if len(srcfiles) == 1:
    #         options_warp['format'] = 'GTiff'
    #         if not dstfile.endswith('.tif'):
    #             dstfile = os.path.splitext(dstfile)[0] + '.tif'
    #         log.debug('creating file: {}'.format(dstfile))
    #         gdalwarp(srcfiles[0], dstfile, options_warp)
    #     else:
    #         # create VRT for stacking
    #         vrt = '/vsimem/' + os.path.basename(dst_base) + '.vrt'
    #         options_buildvrt['options'] = ['-separate']
    #         gdalbuildvrt(srcfiles, vrt, options_buildvrt)
            
    #         # increase the number of threads for gdalwarp computations
    #         options_warp['options'].extend(['-wo', 'NUM_THREADS={}'.format(cores)])
            
    #         log.debug('creating file: {}'.format(dstfile))
    #         gdalwarp(vrt, dstfile, options_warp, pbar=pbar)
            
    #         # edit ENVI HDR files to contain specific layer names
    #         hdrfile = os.path.splitext(dstfile)[0] + '.hdr'
    #         with envi.HDRobject(hdrfile) as hdr:
    #             hdr.band_names = bandnames
    #             hdr.write()
    ##########################################################################################


def apply_along_time(src, dst, func1d, nodata, format, cmap=None, maxlines=None, cores=8, *args, **kwargs):
    """
    Apply a time series computation to a 3D raster stack using multiple CPUs.
    The stack is read in chunks of maxlines x columns x time steps, 
    for which the result is computed and stored in a 2D output array.
    After finishing the computation for all chunks, the output array is 
    written to the specified file.
    
    Notes
    -----
    It is intended to directly write the computation result of each chunk to the output file respectively so that no
    unnecessary memory is used for storing the complete result.
    This however first requires some restructuring of the method :meth:`spatialist.Raster.write`.
    Parameters
    ----------
    src: Raster
        the source raster data
    dst: str
        the output file in GeoTiff format
    func1d: function
        the function to be applied over a single time series 1D array
    nodata: int
        the nodata value to write to the output file
    format: str
        the output file format, e.g. 'GTiff'
    cmap: gdal.ColorTable
        a color table to write to the resulting file; see :func:`spatialist.auxil.cmap_mpl2gdal` for creation options.
    maxlines: int
        the maximum number of lines to read at once. Controls the amount of memory used.
    cores: int
        the number of parallel cores
    args: any
        Additional arguments to `func1d`.
    kwargs:any
        Additional named arguments to `func1d`.
    See Also
    --------
    :func:`spatialist.ancillary.parallel_apply_along_axis`
    
    Returns
    -------
    """
    rows, cols, bands = src.dim
    
    if maxlines is None or maxlines > rows:
        maxlines = rows
    start = 0
    stop = start + maxlines
    
    while start < rows:
        print('processing lines {0}:{1}'.format(start, stop))
        with src[start:stop, :, :] as sub:
            arr = sub.array()
        out = parallel_apply_along_axis(func1d=func1d, axis=2,
                                        arr=arr, cores=cores,
                                        *args, **kwargs)
        
        with src[:, :, 0] as ref:
            ref.write(outname=dst, nodata=nodata, dtype='float32',
                      format=format, cmap=cmap, yoff=start,
                      array=out, update=True)
        
        start += maxlines
        stop += maxlines
        if stop > rows:
            stop = rows

###############################################################################

class VectorData(object):
    def __init__(self, path):
        if isinstance(path, str):
            self.path = path if os.path.isabs(path)\
                else os.path.join(os.getcwd(), path)
            vector = gpd.read_file(path)
            self.vector = vector
        elif isinstance(path, Path):
            vector = gpd.read_file(path)
            self.vector = vector

        # TODO!
        elif isinstance(path, list):
            print()

        self.path = path
        self.driver = self.getDriverByName(path)

        # TODO!
        # self.vector = self.driver.CreateDataSource('out') if driver == 'Memory' else self.driver.Open(filename)
        # nlayers = self.vector.GetLayerCount()
        # if nlayers > 1:
        #     raise RuntimeError('multiple layers are currently not supported')
        # elif nlayers == 1:
        #     self.init_layer()

    def getDriverByName(self, path):
        if isinstance(path, str):
            vector_path = Path(path)
        elif isinstance(path, Path):
            vector_path = path

        ext = vector_path.suffix
        if "ext" == ".shp":
            driver = "ESRI Shapefile"
        elif "ext" == ".geojson":
            driver = "GeoJSON"
        return driver
    
    def __getitem__(self, expression):
        """
        subset the vector object by index or attribute.
        Parameters
        ----------
        expression: int or str
            the key or expression to be used for subsetting.
            See :osgeo:meth:`ogr.Layer.SetAttributeFilter` for details on the expression syntax.
        Returns
        -------
        Vector
            a vector object matching the specified criteria
        
        Examples
        --------
        Assuming we have a shapefile called `testsites.shp`, which has an attribute `sitename`,
        we can subset individual sites and write them to new files like so:
        
        >>> from spatialist import Vector
        >>> filename = 'testsites.shp'
        >>> with Vector(filename)["sitename='site1'"] as site1:
        >>>     site1.write('site1.shp')
        """
        if not isinstance(expression, (int, str)):
            raise RuntimeError('expression must be of type int or str')

        expression = parse_literal(expression) if isinstance(expression, str) else expression

        # TODO!
        # if isinstance(expression, int):
        #     feat = self.getFeatureByIndex(expression)
        # else:
        #     self.layer.SetAttributeFilter(expression)
        #     feat = self.getfeatures()
        #     feat = feat if len(feat) > 0 else None
        #     self.layer.SetAttributeFilter('')
        # if feat is None:
        #     return None
#        else:
#            return feature2vector(feat, ref=self)
    
    def __enter__(self):
        return self

    def __str__(self):
        vals = dict()
        # TODO!
        #vals['proj4'] = self.proj4
        #vals.update(self.extent)
        vals['filename'] = self.filename if self.filename is not None else 'memory'
        vals['geomtype'] = ', '.join(list(set(self.geomTypes)))
        
        info = 'class         : spatialist Vector object\n' \
               'geometry type : {geomtype}\n' \
               'extent        : {xmin:.3f}, {xmax:.3f}, {ymin:.3f}, {ymax:.3f} (xmin, xmax, ymin, ymax)\n' \
               'coord. ref.   : {proj4}\n' \
               'data source   : {filename}'.format(**vals)
        return info

    def addfeature(self, geometry, fields=None):
        """
        add a feature to the vector object from a geometry
        Parameters
        ----------
        geometry: :osgeo:class:`ogr.Geometry`
            the geometry to add as a feature
        fields: dict or None
            the field names and values to assign to the new feature
        Returns
        -------
        """
        # TODO!
        # feature = ogr.Feature(self.layerdef)
        # feature.SetGeometry(geometry)
        
        # if fields is not None:
        #     for fieldname, value in fields.items():
        #         if fieldname not in self.fieldnames:
        #             raise IOError('field "{}" is missing'.format(fieldname))
        #         try:
        #             feature.SetField(fieldname, value)
        #         except NotImplementedError as e:
        #             fieldindex = self.fieldnames.index(fieldname)
        #             fieldtype = feature.GetFieldDefnRef(fieldindex).GetTypeName()
        #             message = str(e) + '\ntrying to set field {} (type {}) to value {} (type {})'
        #             message = message.format(fieldname, fieldtype, value, type(value))
        #             raise (NotImplementedError(message))
        
        # self.layer.CreateFeature(feature)
        # feature = None
        # self.init_features()
    
    def addfield(self, name, type, width=10):
        """
        add a field to the vector layer
        Parameters
        ----------
        name: str
            the field name
        type: int
            the OGR Field Type (OFT), e.g. ogr.OFTString.
            See `Module ogr <https://gdal.org/python/osgeo.ogr-module.html>`_.
        width: int
            the width of the new field (only for ogr.OFTString fields)
        Returns
        -------
        """
        # TODO!
        # fieldDefn = ogr.FieldDefn(name, type)
        # if type == ogr.OFTString:
        #     fieldDefn.SetWidth(width)
        # self.layer.CreateField(fieldDefn)
    
    def addlayer(self, name, srs, geomType):
        """
        add a layer to the vector layer
        Parameters
        ----------
        name: str
            the layer name
        srs: int, str or :osgeo:class:`osr.SpatialReference`
            the spatial reference system. See :func:`spatialist.auxil.crsConvert` for options.
        geomType: int
            an OGR well-known binary data type.
            See `Module ogr <https://gdal.org/python/osgeo.ogr-module.html>`_.
        Returns
        -------
        """
        # TODO!
#        self.vector.CreateLayer(name, srs, geomType)
#        self.init_layer()

    def addvector(self, vec):
        """
        add a vector object to the layer of the current Vector object
        Parameters
        ----------
        vec: Vector
            the vector object to add
        merge: bool
            merge overlapping polygons?
        Returns
        -------
        """
        # TODO!
        # vec.layer.ResetReading()
        # for feature in vec.layer:
        #     self.layer.CreateFeature(feature)
        # self.init_features()
        # vec.layer.ResetReading()

    def bbox(self, outname=None, driver=None, overwrite=True):
        """
        create a bounding box from the extent of the Vector object
        Parameters
        ----------
        outname: str or None
            the name of the vector file to be written; if None, a Vector object is returned
        driver: str
            the name of the file format to write
        overwrite: bool
            overwrite an already existing file?
        Returns
        -------
        Vector or None
            if outname is None, the bounding box Vector object
        """
        # TODO!
#        if outname is None:
#            return bbox(self.extent, self.srs)
#        else:
#            bbox(self.extent, self.srs, outname=outname, driver=driver, overwrite=overwrite)

    # TODO!
#    def clone(self):
#        return feature2vector(self.getfeatures(), ref=self)

    # TODO!
    def convert2wkt(self, set3D=True):
        """
        export the geometry of each feature as a wkt string
        Parameters
        ----------
        set3D: bool
            keep the third (height) dimension?
        Returns
        -------
        """
        features = self.getfeatures()
        for feature in features:
            try:
                feature.geometry().Set3D(set3D)
            except AttributeError:
                dim = 3 if set3D else 2
                feature.geometry().SetCoordinateDimension(dim)
        
        return [feature.geometry().ExportToWkt() for feature in features]

    @property
    def extent(self):
        """
        the extent of the vector object
        Returns
        -------
        dict
            a dictionary with keys `xmin`, `xmax`, `ymin`, `ymax`
        """
        # TODO!
#        return dict(zip(['xmin', 'xmax', 'ymin', 'ymax'], self.layer.GetExtent()))
    
    @property
    def fieldDefs(self):
        """
        the field definition for each field of the Vector object
        """
        # TODO
#        return [self.layerdef.GetFieldDefn(x) for x in range(0, self.nfields)]
    
    @property
    def fieldnames(self):
        """
        the names of the fields
        """
        # TODO
#       return sorted([field.GetName() for field in self.fieldDefs])
    
    @property
    def geomType(self):
        """
        the layer geometry type
        """
        # TODO
#       return self.layerdef.GetGeomType()
    
    @property
    def geomTypes(self):
        """
        the geometry type of each feature
        """
        return [feat.GetGeometryRef().GetGeometryName() for feat in self.getfeatures()]

    def getArea(self):
        """
        the area of the vector geometries
        """
        # TODO
#        return sum([x.GetGeometryRef().GetArea() for x in self.getfeatures()])
    
    def getFeatureByAttribute(self, fieldname, attribute):
        """
        get features by field attribute
        Parameters
        ----------
        fieldname: str
            the name of the queried field
        attribute: int or str
            the field value of interest
        Returns
        -------
        list of :osgeo:class:`ogr.Feature` or :osgeo:class:`ogr.Feature`
            the feature(s) matching the search query
        """
        # TODO1
        attr = attribute.strip() if isinstance(attribute, str) else attribute
        if fieldname not in self.fieldnames:
            raise KeyError('invalid field name')
        out = []
#        self.layer.ResetReading()
#        for feature in self.layer:
#            field = feature.GetField(fieldname)
#            field = field.strip() if isinstance(field, str) else field
#            if field == attr:
#                out.append(feature.Clone())
#        self.layer.ResetReading()
#        if len(out) == 0:
#            return None
#        elif len(out) == 1:
#            return out[0]
#        else:
#            return out
    
    def getFeatureByIndex(self, index):
        """
        get features by numerical (positional) index
        Parameters
        ----------
        index: int
            the queried index
        Returns
        -------
        :osgeo:class:`ogr.Feature`
            the requested feature
        """
        # TODO
        # feature = self.layer[index]
        # if feature is None:
        #     feature = self.getfeatures()[index]
        # return feature
    
    def getfeatures(self):
        """
        list of cloned features
        """
        # TODO
#        self.layer.ResetReading()
#        features = [x.Clone() for x in self.layer]
#        self.layer.ResetReading()
#        return features
    
    def getProjection(self, type):
        """
        get the CRS of the Vector object. See :func:`spatialist.auxil.crsConvert`.
        Parameters
        ----------
        type: str
            the type of projection required.
        Returns
        -------
        int, str or :osgeo:class:`osr.SpatialReference`
            the output CRS
        """
        # TODO
#        return crsConvert(self.layer.GetSpatialRef(), type)
    
    def getUniqueAttributes(self, fieldname):
        """
        the unique attributes of the field
        """
        # TODO!
        # self.layer.ResetReading()
        # attributes = list(set([x.GetField(fieldname) for x in self.layer]))
        # self.layer.ResetReading()
        # return sorted(attributes)
    
    def init_features(self):
        """
        delete all in-memory features
        """
        # TODO!
#        del self.__features
#        self.__features = [None] * self.nfeatures
    
    def init_layer(self):
        """
        initialize a layer object
        """
        # TODO!
#        self.layer = self.vector.GetLayer()
#        self.__features = [None] * self.nfeatures
    
    @property
    def layerdef(self):
        """
        the layer's feature definition
        """
        # TODO
#      return self.layer.GetLayerDefn()
    
    @property
    def layername(self):
        """
        the name of the layer
        """
        # TODO
#        return self.layer.GetName()
    
    def load(self):
        """
        load all feature into memory
        """
        # TODO!
        # self.layer.ResetReading()
        # for i in range(self.nfeatures):
        #     if self.__features[i] is None:
        #         self.__features[i] = self.layer[i]
    
    @property
    def nfeatures(self):
        """
        the number of features
        """
        # TODO!
#        return len(self.layer)
    
    @property
    def nfields(self):
        """
        the number of fields
        """
        # TODO!
#        return self.layerdef.GetFieldCount()
    
    @property
    def nlayers(self):
        """
        the number of layers
        """
        # TODO!
#        return self.vector.GetLayerCount()
    
    @property
    def proj4(self):
        """
        the CRS in PRO4 format
        """
        # TODO!
#        return self.vector.GetLayerCount()
        return self.srs.ExportToProj4().strip()
    
    def reproject(self, projection):
        """
        in-memory reprojection
        """
        # TODO!
#        srs_out = crsConvert(projection, 'osr')
        
        # the following check was found to not work in GDAL 3.0.1; likely a bug
        # if self.srs.IsSame(srs_out) == 0:
#        if self.getProjection('epsg') != crsConvert(projection, 'epsg'):
            
            # create the CoordinateTransformation
#            coordTrans = osr.CoordinateTransformation(self.srs, srs_out)
            
 #           layername = self.layername
 #           geomType = self.geomType
 #           features = self.getfeatures()
 #           feat_def = features[0].GetDefnRef()
 #           fields = [feat_def.GetFieldDefn(x) for x in range(0, feat_def.GetFieldCount())]
            
 #           self.__init__()
 #           self.addlayer(layername, srs_out, geomType)
 #           self.layer.CreateFields(fields)
            
#            for feature in features:
#                geom = feature.GetGeometryRef()
#                geom.Transform(coordTrans)
#                newfeature = feature.Clone()
#                newfeature.SetGeometry(geom)
#                self.layer.CreateFeature(newfeature)
#                newfeature = None
#            self.init_features()
    
    def setCRS(self, crs):
        """
        directly reset the spatial reference system of the vector object.
        This is not going to reproject the Vector object, see :meth:`reproject` instead.
 
        Example
        -------
        >>> site = Vector('shape.shp')
        >>> site.setCRS('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs ')
        """
        # TODO
        # try to convert the input crs to osr.SpatialReference
        # srs_out = crsConvert(crs, 'osr')
        
        # # save all relevant info from the existing vector object
        # layername = self.layername
        # geomType = self.geomType
        # layer_definition = ogr.Feature(self.layer.GetLayerDefn())
        # fields = [layer_definition.GetFieldDefnRef(x) for x in range(layer_definition.GetFieldCount())]
        # features = self.getfeatures()
        
        # # initialize a new vector object and create a layer
        # self.__init__()
        # self.addlayer(layername, srs_out, geomType)
        
        # # add the fields to new layer
        # self.layer.CreateFields(fields)
        
        # # add the features to the newly created layer
        # for feat in features:
        #     self.layer.CreateFeature(feat)
        # self.init_features()
    
    @property
    def srs(self):
        """
        the geometry's spatial reference system
        """
        # TODO!
#        return self.layer.GetSpatialRef()
    
    def write(self, outfile, driver=None, overwrite=True):
        """
        write the Vector object to a file
        Parameters
        ----------
        outfile:
            the name of the file to write; the following extensions are automatically detected
            for determining the format driver:
            
            .. list_drivers:: vector
            
        driver: str
            the output file format; needs to be defined if the format cannot
            be auto-detected from the filename extension
        overwrite: bool
            overwrite an already existing file?
        Returns
        -------
        """
        # TODO!
        # if driver is None:
        #     driver = self.__driver_autodetect(outfile)
        
        # driver = ogr.GetDriverByName(driver)
        
        # if os.path.exists(outfile):
        #     if overwrite:
        #         driver.DeleteDataSource(outfile)
        #     else:
        #         raise RuntimeError('target file already exists')
        
        # outdataset = driver.CreateDataSource(outfile)
        # outlayer = outdataset.CreateLayer(name=self.layername,
        #                                   srs=self.srs,
        #                                   geom_type=self.geomType)
        # outlayerdef = outlayer.GetLayerDefn()
        
        # for fieldDef in self.fieldDefs:
        #     outlayer.CreateField(fieldDef)
        
        # self.layer.ResetReading()
        # for feature in self.layer:
        #     outFeature = ogr.Feature(outlayerdef)
        #     outFeature.SetGeometry(feature.GetGeometryRef())
        #     for name in self.fieldnames:
        #         outFeature.SetField(name, feature.GetField(name))
        #     # add the feature to the shapefile
        #     outlayer.CreateFeature(outFeature)
        #     outFeature = None
        # self.layer.ResetReading()
        # outdataset = None
    
################################################################################

def bbox(coordinates, crs, outname=None, driver=None, overwrite=True):
    """
    create a bounding box vector object or shapefile from coordinates and 
    coordinate reference system.
    The CRS can be in either WKT, EPSG or PROJ4 format
    
    Parameters
    ----------
    coordinates: dict
        a dictionary containing numerical variables with keys `xmin`, `xmax`, `ymin` and `ymax`
    crs: int, str, :osgeo:class:`osr.SpatialReference`
        the CRS of the `coordinates`. See :func:`~spatialist.auxil.crsConvert` for options.
    outname: str
        the file to write to. If `None`, the bounding box is returned as 
        :class:`~spatialist.vector.Vector` object
    driver: str
        the output file format; needs to be defined if the format cannot
            be auto-detected from the filename extension
    overwrite: bool
        overwrite an existing file?
    
    Returns
    -------
    Vector or None
        the bounding box Vector object
    """
    # TODO!
    # srs = crsConvert(crs, 'osr')
    # ring = ogr.Geometry(ogr.wkbLinearRing)
    
    # ring.AddPoint(coordinates['xmin'], coordinates['ymin'])
    # ring.AddPoint(coordinates['xmin'], coordinates['ymax'])
    # ring.AddPoint(coordinates['xmax'], coordinates['ymax'])
    # ring.AddPoint(coordinates['xmax'], coordinates['ymin'])
    # ring.CloseRings()
    
    # geom = ogr.Geometry(ogr.wkbPolygon)
    # geom.AddGeometry(ring)
    
    # geom.FlattenTo2D()
    
    # bbox = Vector(driver='Memory')
    # bbox.addlayer('bbox', srs, geom.GetGeometryType())
    # bbox.addfield('area', ogr.OFTReal)
    # bbox.addfeature(geom, fields={'area': geom.Area()})
    # geom = None
    # if outname is None:
    #     return bbox
    # else:
    #     bbox.write(outfile=outname, driver=driver, overwrite=overwrite)

def centerdist(obj1, obj2):
    if not isinstance(obj1, VectorData) or isinstance(obj2, VectorData):
        raise IOError('both objects must be of type Vector')
    
    # TODO
    # feature1 = obj1.getFeatureByIndex(0)
    # geometry1 = feature1.GetGeometryRef()
    # center1 = geometry1.Centroid()
    
    # feature2 = obj2.getFeatureByIndex(0)
    # geometry2 = feature2.GetGeometryRef()
    # center2 = geometry2.Centroid()
    
    # return center1.Distance(center2)

def dissolve_polygon(infile, outfile, field, layername=None):
    """
    dissolve the polygons of a vector file by an attribute field
    Parameters
    ----------
    infile: str
        the input vector file
    outfile: str
        the output shapefile
    field: str
        the field name to merge the polygons by
    layername: str
        the name of the output vector layer;
        If set to None the layername will be the basename of infile without extension
    Returns
    -------
    """
    with VectorData(infile) as vec:
        srs = vec.srs
        feat = vec.layer[0]
        d = feat.GetFieldDefnRef(field)
        width = d.width
        type = d.type
        feat = None
    
    layername = layername if layername is not None\
        else os.path.splitext(os.path.basename(infile))[0]
    # TODO!
    # the following can be used if GDAL was compiled with the spatialite extension
    # not tested, might need some additional/different lines
    # with Vector(infile) as vec:
    #     vec.vector.ExecuteSQL('SELECT ST_Union(geometry), {0} FROM {1} GROUP BY {0}'.format(field, vec.layername),
    #                          dialect='SQLITE')
    #     vec.write(outfile)
    # conn.execute('CREATE VIRTUAL TABLE merge USING VirtualOGR("{}");'.format(infile))
    # select = conn.execute('SELECT {0},asText(ST_Union(geometry)) as geometry FROM merge GROUP BY {0};'.format(field))
    # fetch = select.fetchall()
    # with Vector(driver='Memory') as merge:
    #     merge.addlayer(layername, srs, ogr.wkbPolygon)
    #     merge.addfield(field, type=type, width=width)
    #     for i in range(len(fetch)):
    #         merge.addfeature(ogr.CreateGeometryFromWkt(fetch[i][1]), {field: fetch[i][0]})
    #     merge.write(outfile)
    # conn.close()

def feature2vector(feature, ref, layername=None):
    """
    create a Vector object from ogr features
    Parameters
    ----------
    feature: list of :osgeo:class:`ogr.Feature` or :osgeo:class:`ogr.Feature`
        a single feature or a list of features
    ref: Vector
        a reference Vector object to retrieve geo information from
    layername: str or None
        the name of the output layer; retrieved from `ref` if `None`
    Returns
    -------
    Vector
        the new Vector object
    """
    # TODO
    features = feature if isinstance(feature, list) else [feature]
    # layername = layername if layername is not None else ref.layername
    # vec = Vector(driver='Memory')
    # vec.addlayer(layername, ref.srs, ref.geomType)
    # feat_def = features[0].GetDefnRef()
    # fields = [feat_def.GetFieldDefn(x) for x in range(0, feat_def.GetFieldCount())]
    # vec.layer.CreateFields(fields)
    # for feat in features:
    #     vec.layer.CreateFeature(feat)
    # vec.init_features()
    # return vec

def intersect(obj1, obj2):
    """
    intersect two Vector objects
    """
    # TODO!
    # if not isinstance(obj1, Vector) or not isinstance(obj2, Vector):
    #     raise RuntimeError('both objects must be of type Vector')
    
    # obj1 = obj1.clone()
    # obj2 = obj2.clone()
    
    # obj1.reproject(obj2.srs)
    
    # #######################################################
    # # create basic overlap
    # union1 = ogr.Geometry(ogr.wkbMultiPolygon)
    # # union all the geometrical features of layer 1
    # for feat in obj1.layer:
    #     union1.AddGeometry(feat.GetGeometryRef())
    # obj1.layer.ResetReading()
    # union1.Simplify(0)
    # # same for layer2
    # union2 = ogr.Geometry(ogr.wkbMultiPolygon)
    # for feat in obj2.layer:
    #     union2.AddGeometry(feat.GetGeometryRef())
    # obj2.layer.ResetReading()
    # union2.Simplify(0)
    # # intersection
    # intersect_base = union1.Intersection(union2)
    # union1 = None
    # union2 = None
    # #######################################################
    # # compute detailed per-geometry overlaps
    # if intersect_base.GetArea() > 0:
    #     intersection = Vector(driver='Memory')
    #     intersection.addlayer('intersect', obj1.srs, ogr.wkbPolygon)
    #     fieldmap = []
    #     for index, fielddef in enumerate([obj1.fieldDefs, obj2.fieldDefs]):
    #         for field in fielddef:
    #             name = field.GetName()
    #             i = 2
    #             while name in intersection.fieldnames:
    #                 name = '{}_{}'.format(field.GetName(), i)
    #                 i += 1
    #             fieldmap.append((index, field.GetName(), name))
    #             intersection.addfield(name, type=field.GetType(), width=field.GetWidth())
        
    #     for feature1 in obj1.layer:
    #         geom1 = feature1.GetGeometryRef()
    #         if geom1.Intersects(intersect_base):
    #             for feature2 in obj2.layer:
    #                 geom2 = feature2.GetGeometryRef()
    #                 # select only the intersections
    #                 if geom2.Intersects(intersect_base):
    #                     intersect = geom2.Intersection(geom1)
    #                     fields = {}
    #                     for item in fieldmap:
    #                         if item[0] == 0:
    #                             fields[item[2]] = feature1.GetField(item[1])
    #                         else:
    #                             fields[item[2]] = feature2.GetField(item[1])
    #                     intersection.addfeature(intersect, fields)
    #     intersect_base = None
    #     return intersection

def wkt2vector(wkt, srs, layername='wkt'):
    """
    convert a well-known text string geometry to a Vector object
    Examples
    --------
    >>> from spatialist.vector import wkt2vector
    >>> wkt = 'POLYGON ((0. 0., 0. 1., 1. 1., 1. 0., 0. 0.))'
    >>> with wkt2vector(wkt, srs=4326) as vec:
    >>>     print(vec.getArea())
    1.0
    """
    # TODO!
    # geom = ogr.CreateGeometryFromWkt(wkt)
    # geom.FlattenTo2D()
    
    # srs = crsConvert(srs, 'osr')
    
    # vec = Vector(driver='Memory')
    # vec.addlayer(layername, srs, geom.GetGeometryType())
    # if geom.GetGeometryName() != 'POINT':
    #     vec.addfield('area', ogr.OFTReal)
    #     fields = {'area': geom.Area()}
    # else:
    #     fields = None
    # vec.addfeature(geom, fields=fields)
    # geom = None
    # return vec
