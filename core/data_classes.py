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

import sys
sys.path.append("D:/Code/eotopia/utils")
from raster_utils import rasterize


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

