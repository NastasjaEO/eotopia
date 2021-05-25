# -*- coding: utf-8 -*-
"""
Created on Tue May 25 14:21:07 2021

@author: nasta
"""

import os
import rasterio

import sys
sys.path.append("D:/Code/eotopia/utils")
from list_utils import dissolve


def simple_rasterstack(filelist, dstfile):
    with rasterio.open(filelist[0]) as src0:
        meta = src0.meta
    meta.update(count = len(filelist))
    with rasterio.open(dstfile, 'w', **meta) as dst:
        for id, layer in enumerate(filelist, start=1):
            with rasterio.open(layer) as src1:
                dst.write_band(id, src1.read(1))


def stack(srcfiles, dstfile, resampling, targetres, dstnodata, 
          srcnodata=None, shapefile=None, layernames=None,
          sortfun=None, separate=False, 
          overwrite=False, compress=True, cores=4, pbar=False):
    """
    function for mosaicking, resampling and stacking of multiple raster files

    Parameters
    ----------
    srcfiles: list
        a list of file names or a list of lists; each sub-list is treated as a 
        task to mosaic its containing files
    dstfile: str
        the destination file or a directory (if `separate` is True)
    resampling: {near, bilinear, cubic, cubicspline, lanczos, average, mode, 
                 max, min, med, Q1, Q3}
        the resampling method; 
        see `documentation of gdalwarp <https://www.gdal.org/gdalwarp.html>`_.
    targetres: tuple or list
        two entries for x and y spatial resolution in units of the source CRS
    srcnodata: int, float or None
        the nodata value of the source files; if left at the default (None), 
        the nodata values are read from the files
    dstnodata: int or float
        the nodata value of the destination file(s)
    shapefile: str, Vector or None
        a shapefile for defining the spatial extent of the destination files
    layernames: list
        the names of the output layers; if `None`, the basenames of the 
        input files are used; overrides sortfun
    sortfun: function
        a function for sorting the input files; not used if layernames is not None.
        This is first used for sorting the items in each sub-list of srcfiles;
        the basename of the first item in a sub-list will then be used as the 
        name for the mosaic of this group.
        After mosaicing, the function is again used for sorting the names in 
        the final output
        (only relevant if `separate` is False)
    separate: bool
        should the files be written to a single raster stack (ENVI format) or 
        separate files (GTiff format)?
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
    
    Notes
    -----
    This function does not reproject any raster files. Thus, the CRS must be 
    the same for all input raster files.
    This is checked prior to executing gdalwarp. In case a shapefile is defined, 
    it is internally reprojected to the raster CRS prior to retrieving its extent.

    Examples
    --------
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
    #     shp = shapefile.clone() if isinstance(shapefile, VectorData)\
    #           else VectorData(shapefile)
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
