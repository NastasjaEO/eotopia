# -*- coding: utf-8 -*-
"""
Created on Thu May 13 12:17:55 2021

@author: freeridingeo
"""

from xml.etree import ElementTree as ET
from lxml import etree
import datetime

def metadata_from_safefile(safefile, namespaces):
    tree = ET.fromstring(safefile)
        
    meta = dict()
    meta['acquisition_mode'] = tree.find('.//s1sarl1:mode', namespaces).text
    meta['acquisition_time'] = dict(
            [(x, tree.find('.//safe:{}Time'.format(x), namespaces).text)\
             for x in ['start', 'stop']])
#    meta['start'], meta['stop'] = (parse_date(meta['acquisition_time'][x])\
#                                   for x in ['start', 'stop'])
    meta['coordinates'] = [tuple([float(y) for y in x.split(',')]) for x in
                               tree.find('.//gml:coordinates', namespaces).text.split()]
    meta['orbit'] = tree.find('.//s1:pass', namespaces).text[0]
        
    meta['orbitNumber_abs'] = int(tree.find('.//safe:orbitNumber[@type="start"]', 
                                            namespaces).text)
    meta['orbitNumber_rel'] = int(tree.find('.//safe:relativeOrbitNumber[@type="start"]', 
                                            namespaces).text)
    meta['cycleNumber'] = int(tree.find('.//safe:cycleNumber', namespaces).text)
    meta['frameNumber'] = int(tree.find('.//s1sarl1:missionDataTakeID', namespaces).text)
    
    meta['orbitNumbers_abs'] = dict(
            [(x, int(tree.find('.//safe:orbitNumber[@type="{0}"]'.format(x), 
                               namespaces).text)) for x in
             ['start', 'stop']])
    meta['orbitNumbers_rel'] = dict(
            [(x, int(tree.find('.//safe:relativeOrbitNumber[@type="{0}"]'.format(x), 
                               namespaces).text)) for x in
             ['start', 'stop']])
    meta['polarizations'] = [x.text\
                             for x in\
                                 tree.findall('.//s1sarl1:transmitterReceiverPolarisation', namespaces)]
    meta['product'] = tree.find('.//s1sarl1:productType', namespaces).text
    meta['category'] = tree.find('.//s1sarl1:productClass', namespaces).text
    meta['sensor'] = tree.find('.//safe:familyName', 
                               namespaces).text.replace('ENTINEL-', '') + tree.find(
            './/safe:number', namespaces).text
    meta['IPF_version'] = float(tree.find('.//safe:software', 
                                          namespaces).attrib['version'])
    meta['sliceNumber'] = int(tree.find('.//s1sarl1:sliceNumber', namespaces).text)
    meta['totalSlices'] = int(tree.find('.//s1sarl1:totalSlices', namespaces).text)
        
    meta['spacing'] = tuple([float(tree.find('.//{}PixelSpacing'.format(dim)).text)
                                 for dim in ['range', 'azimuth']])
    meta['samples'] = int(tree.find('.//imageAnnotation/imageInformation/numberOfSamples').text)
    meta['lines'] = int(tree.find('.//imageAnnotation/imageInformation/numberOfLines').text)
    heading = float(tree.find('.//platformHeading').text)
    meta['heading'] = heading if heading > 0 else heading + 360
    meta['incidence'] = float(tree.find('.//incidenceAngleMidSwath').text)
    meta['image_geometry'] = tree.find('.//projection').text.replace(' ', '_').upper()
        
    return meta

def subswath_metadata(xmlfile):
    '''
    Parse S1 subswath xml file and assign metadata parameters
    This function will parse the xml file and assign the values needed
    to a python dictionary
    '''
    xmldoc = ET.parse(xmlfile).getroot()
    meta_dict = {}
    meta_dict['missionId'] = xmldoc.find('.//adsHeader/missionId').text
    meta_dict['productType'] = xmldoc.find('.//adsHeader/productType').text
    meta_dict['polarisation'] = xmldoc.find('.//adsHeader/polarisation').text
    meta_dict['mode'] = xmldoc.find('.//adsHeader/mode').text
    meta_dict['swath'] = xmldoc.find('.//adsHeader/swath').text
    meta_dict['numberOfLooks'] = xmldoc.find('.//imageAnnotation/processingInformation/swathProcParamsList/swathProcParams/rangeProcessing/numberOfLooks').text
    meta_dict['numberOfSamples'] = xmldoc.find('.//imageAnnotation/imageInformation/numberOfSamples').text
    meta_dict['numberOfLines'] = xmldoc.find('.//imageAnnotation/imageInformation/numberOfLines').text
    meta_dict['firstValidSample'] = xmldoc.find('.//swathTiming/burstList/burst/firstValidSample').text#    meta_dict['re'] = xmldoc.find('.//.../re').text
    meta_dict['linesPerBurst'] = xmldoc.find('.//swathTiming/linesPerBurst').text#    meta_dict['re'] = xmldoc.find('.//.../re').text
    meta_dict['meanBitRate'] = xmldoc.find('.//generalAnnotation/downlinkInformationList/downlinkInformation/downlinkValues/dataFormat/meanBitRate').text
    meta_dict['pixelValue'] = xmldoc.find('.//imageAnnotation/imageInformation/pixelValue').text
    meta_dict['startTime'] =  datetime.datetime.strptime(xmldoc.find('.//adsHeader/startTime').text,"%Y-%m-%dT%H:%M:%S.%f")
    meta_dict['stopTime'] =  datetime.datetime.strptime(xmldoc.find('.//adsHeader/stopTime').text,"%Y-%m-%dT%H:%M:%S.%f")
    meta_dict['productFirstLineUtcTime'] =  datetime.datetime.strptime(xmldoc.find('.//imageAnnotation/imageInformation/productFirstLineUtcTime').text,"%Y-%m-%dT%H:%M:%S.%f")
    meta_dict['clockstart'] = xmldoc.find('.//imageAnnotation/imageInformation/productFirstLineUtcTime')#.text,"%Y-%m-%dT%H:%M:%S.%f")
    meta_dict['absoluteOrbitNumber'] = xmldoc.find('.//adsHeader/absoluteOrbitNumber').text
    meta_dict['flight_direction'] = xmldoc.find('.//generalAnnotation/productInformation/pass').text
    meta_dict['frequency'] = float(xmldoc.find('.//generalAnnotation/productInformation/radarFrequency').text)
    meta_dict['rangeSampleRate'] = float(xmldoc.find('.//generalAnnotation/productInformation/rangeSamplingRate').text)
    meta_dict['rangePixelSize'] = 299792458./(2.0*meta_dict['rangeSampleRate'])
    meta_dict['azimuthFrequency'] = float(xmldoc.find('.//imageAnnotation/imageInformation/azimuthFrequency').text)
    meta_dict['azimuthPixelSize'] = float(xmldoc.find('.//imageAnnotation/imageInformation/azimuthPixelSpacing').text)
    meta_dict['azimuthTimeInterval'] = float(xmldoc.find('.//imageAnnotation/imageInformation/azimuthTimeInterval').text)
    meta_dict['azimuthAnxTime'] = float(xmldoc.find('.//swathTiming/burstList/burst/azimuthAnxTime').text)      # ANX: Ascending Node Crossing
    meta_dict['lookBandwidth'] = float(xmldoc.find('.//imageAnnotation/processingInformation/swathProcParamsList/swathProcParams/rangeProcessing/lookBandwidth').text)    
    meta_dict['lines'] = int(xmldoc.find('.//swathTiming/linesPerBurst').text)
    meta_dict['samples'] = int(xmldoc.find('.//swathTiming/samplesPerBurst').text)
    meta_dict['startingRange'] = float(xmldoc.find('.//imageAnnotation/imageInformation/slantRangeTime').text)*299792458./2.0
    meta_dict['incidenceAnFe'] = float(xmldoc.find('.//imageAnnotation/imageInformation/incidenceAngleMidSwath').text)
    meta_dict['slantRangeTime'] = float(xmldoc.find('.//imageAnnotation/imageInformation/slantRangeTime').text)
    meta_dict['prf'] = (xmldoc.find('.//generalAnnotation/downlinkInformationList/downlinkInformation/prf').text)
    meta_dict['txPulseLength'] = float(xmldoc.find('.//generalAnnotation/downlinkInformationList/downlinkInformation/downlinkValues/txPulseLength').text)
    meta_dict['platformHeading'] = float(xmldoc.find('.//platformHeading').text)
    meta_dict['incidenceAngleMidSwath'] = float(xmldoc.find('.//incidenceAngleMidSwath').text)
    meta_dict['terrainHeight'] = float(xmldoc.find('.//generalAnnotation/terrainHeightList/terrainHeight/value').text)
    return meta_dict



def xml_platformHeading(path):
    """
    path: Path to XML file of the specific swath
    """
    tree = etree.parse(path)
    root = tree.getroot()
    platformHeading = root.xpath('.//platformHeading')
    return float(platformHeading[0].text)

def xml_incangle(path):
    """
    path: Path to XML file of the specific swath
    """
    tree = etree.parse(path)
    root = tree.getroot()
    incangle = root.xpath('.//incidenceAngleMidSwath')
    return float(incangle[0].text)

def xml_gridpoint_data(path):
    tree = etree.parse(path)
    root = tree.getroot()
#    geolocationGridPoints = root.xpath('.//geolocationGridPoint')
    incidenceAngles = root.xpath('.//incidenceAngle')
    elevationAngles = root.xpath('.//elevationAngle')
    latitudes = root.xpath('.//latitude')
    longitudes = root.xpath('.//longitude')
    hghts = root.xpath('.//height')
 
    incA = []
    for i in range(len(incidenceAngles)):
        incA.append(incidenceAngles[i].text)
    elevA = []
    for i in range(len(elevationAngles)):
        elevA.append(elevationAngles[i].text)
    lats = []
    for i in range(len(latitudes)):
        lats.append(latitudes[i].text)
    lons = []
    for i in range(len(longitudes)):
        lons.append(longitudes[i].text)
    heights = []
    for i in range(len(hghts)):
        heights.append(hghts[i].text)

    length = len(incA)
    lim = len(lats)
    start = length - lim
    incA = incA[start:]
    elevA = elevA[start:]
    return incA, elevA, heights, lats, lons

