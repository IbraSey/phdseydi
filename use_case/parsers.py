#!/usr/bin/env python3
# coding: utf-8
#--------------------------------------------------------------------------
# Filename: parsers.py
#  Purpose: Parse input files
# Creation: 2020-05-23
#  Authors: M. Keller, C. Duverger
#--------------------------------------------------------------------------

"""
Parse input files in a specifc format for phebus.
"""

import os
import pyproj
import numpy as np
import pandas as pd
import shapely.geometry as sg
from shapely.validation import explain_validity
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
import xml.etree.ElementTree as ET

from .sourcemodel import SourceModel, AreaSource, GRTruncated, Rupture


def parse_coastlines(FILE):
    """Read geographical coastlines coordinates.

    Parameters
    ----------
    FILE: character string
        Pathname to file containing coastlines in lon/lat coordinates
    proj: pyproj.Proj object
        Project lon/lat into km ground coordinates

    Returns
    -------
    coastlines: list of arrays
        List of longitudes and latitudes coordinates

    Notes
    -----
    The coastlines file should contain one pair of coordinates per line
    nan's indicate a separation between disconnected islands.

    We should modifiy this function to read and return pandas.Dataframe.
    """

    coastlines = []
    nlines = 0

    with open(FILE) as f:
        lines = f.read().splitlines()

    for line in lines:
        if np.isnan(float(line.split()[0])):
            if nlines > 0:
                coastlines.append(np.array([lons, lats]))
            nlines += 1
            lons = []
            lats = []
        else:
            lons.append(float(line.split()[0]))
            lats.append(float(line.split()[1]))

    coastlines.append(np.array([lons, lats]))

    return coastlines


def proj_coastlines(coastlines, p_to=pyproj.CRS("epsg:2154")):
    """Project geographical coordinates coastlines into ground coordinates.

    Parameters
    ----------
    coastlines : list of numpy.array
        Coastlines in longitude/latitude coordinates
    proj: pyproj.Proj
        Project geographical (longitude, latitude) coordinates into ground (kilometric) coordinates

    Returns
    -------
    coastlines_xy: list of numpy.array
        Coastlines in projected ground kilometric coordinates
    """
    p = pyproj.CRS("epsg:4326")
    t = pyproj.Transformer.from_crs(p, p_to, always_xy=True)
    coastlines_xy = []
    for array in coastlines:
        lons = array[0]
        lats = array[1]
        projarray = t.transform(lons, lats)
        coastlines_xy.append(np.array(projarray)*1e-3)

    return coastlines_xy


def parse_catalog(FILE_CATALOG):
    """Read catalogue of earthquakes.

    Parameters
    ----------
    FILE_CATALOG: character string
        Pathname to file containing earthquakes

    Returns
    -------
    catalog: pandas.Dataframe
    """
    catalog = pd.read_csv(FILE_CATALOG)
    return catalog


def proj_catalog(catalog, p_to=pyproj.CRS("epsg:2154")):
    """Project geographical coordinates into ground coordinates.

    This function computes the ground kilometric coordinates of earthquakes
    from the geographical ones and return an updated catalogue with X and Y
    columns.

    Parameters
    ----------
    catalog : pandas.Dataframe
        Catalog of earthquakes in longitude/latitude coordinates.
    proj: pyproj.Proj object
        Projection in ground (kilometric) coordinates.
        Default : pyproj.CRS("epsg:2154").

    Returns
    -------
    catalog : pandas.Dataframe
        Catalog with X, Y columns, projected in ground kilometric coordinates.
    """

    p = pyproj.CRS("epsg:4326")  # input projection in geographical coordinates
    t = pyproj.Transformer.from_crs(p, p_to, always_xy=True)
    X,Y = np.zeros((2, len(catalog)))
    for i in range(len(catalog)):
        earthquake = catalog.iloc[i]
        X[i],Y[i] = np.array( t.transform(earthquake['longitude'], earthquake['latitude']) )*1e-3
    catalog['X'] = X
    catalog['Y'] = Y
    return catalog


def plot_polygon(ax, poly, **kwargs):
    path = Path.make_compound_path(
        Path(np.asarray(poly.exterior.coords)[:, :2]),
        *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])

    patch = PathPatch(path, **kwargs)
    collection = PatchCollection([patch], **kwargs)

    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection


def parse_polygons( FILE_POLYGONS ):
    """Create a list of AreaSource objects.

    Parameters
    ----------
    FILE_POLYGONS : str
        Path to file containing source model polygon file.

    Returns
    -------
    zones : list
        List of AreaSource objects created from names and coordinates in
        polygon file.

    Notes
    -----
    See test scripts test_source_model and test_source_model_tree for usage
    examples and file formatting.
    """

    import matplotlib.patches as mp

    with open(FILE_POLYGONS) as f:
        lines = f.read().splitlines()

    ###############################################################
    # fill-in zone names and boundary coordinates, stack in zones #
    ###############################################################

    zones = []
    zone_name = None
    coords = []

    for line in lines:
        if line.startswith('>'): #end of polygon for one zone
            polygon = sg.Polygon(coords)

            if not polygon.is_valid :
                print(explain_validity(polygon))

                # # figure
                # fig,ax = plt.subplots()
                # plot_polygon(ax, polygon, facecolor=(1,0,0,0.2), edgecolor=(1,0,0,1), lw=1.5)
                # plt.title("Polygon %s - Invalid" % zone_name)
                # plt.show()

                polygon = polygon.buffer(0)
                # fig,ax = plt.subplots()
                # plot_polygon(ax, polygon, facecolor=(0,0.5,0,0.2), edgecolor=(0,0.5,0,1), lw=1.5)
                # plt.title("Polygon %s - Valid" % zone_name)
                # plt.show()
                print("%s corrected!" % zone_name)

            zone = AreaSource(name=zone_name, polygon=polygon)
            zones.append(zone)
            print('added zone %s' % zone_name)
            coords = []
            zone_name = None
        else:
            pt = (float(line.split()[0]), float(line.split()[1]))
            coords.append(pt)
            if not zone_name:
                zone_name = str(line.split()[2])

    return zones


def parse_seismic_parameters(zones, FILE_PARAMETERS):
    """Parse seismic parameters.

    Update AreaSource objects by adding GRTruncated attribute given in
    separate file.

    Parameters
    ----------
    FILE_PARAMETERS : str
        Pathname to seismic parameters file.
    zones : list
        List of pybus.sourcemodel.AreaSource objects.

    Returns
    -------
    zones : list
        List of pybus.sourcemodel.AreaSource objects, with mfd attributes
        (pybus.recurrencelaw.GRTruncated objects).
    """

    seismic_parameters_file_header = ['name', 'extended_name', 'date', 'mmax_obs', 'mmax_min', 'mmax_max', 'source_depth_min', 'source_depth_max', 'upper_seismogenic_depth', 'lower_seismogenic_depth', 'rupture1_weight', 'strike1_min', 'strike1_max', 'dip1_min', 'dip1_max', 'fault_type1', 'rupture2_weight', 'strike2_min', 'strike2_max', 'dip2_min', 'dip2_max', 'fault_type2', 'rupture3_weight', 'strike3_min', 'strike3_max', 'dip3_min', 'dip3_max', 'fault_type3', 'surface_km2', 'unknown1', 'unknown2']

    # read zone activity parameters file
    seismic_parameters = pd.read_csv(FILE_PARAMETERS, header=None, names=seismic_parameters_file_header, sep='\s+', dtype={'name':str, 'extended_name':str, 'fault_type1':str, 'fault_type2':str, 'fault_type3':str})
    # print(seismic_parameters)

    ########################
    # add mfd and ruptures #
    ########################

    new_zones = []

    for i in range(len(seismic_parameters)):
        parameters = seismic_parameters.iloc[i]
        # print(parameters)
        try:
            index = [zone.name for zone in zones].index(parameters['name'])
        except:
            print('AreaSource %s not in model.'%(parameters['name']))
            continue

        zone = zones[index]

        zone.upper_seismogenic_depth = parameters['upper_seismogenic_depth']
        zone.lower_seismogenic_depth = parameters['lower_seismogenic_depth']

        zone.source_depth_min = parameters['source_depth_min']
        zone.source_depth_max = parameters['source_depth_max']

        zone.mmax_max = parameters['mmax_max']
        zone.mmax_min = parameters['mmax_min']

        zone.mfd = GRTruncated()
        zone.mfd.mmin = 4.5
        zone.mfd.mbin = 0.1
        zone.mfd.mmax = (parameters['mmax_min'] + parameters['mmax_max'])/2.

        typeF1 = parameters['fault_type1']
        weight1 = parameters['rupture1_weight']
        strike1 = (parameters['strike1_min'] + parameters['strike1_max'])/2.
        dip1 = (parameters['dip1_min'] + parameters['dip1_max'])/2.
        rup1 = Rupture(typeF1, weight1, strike1, dip1)

        typeF2 = parameters['fault_type2']
        weight2 = parameters['rupture2_weight']
        strike2 = (parameters['strike2_min'] + parameters['strike2_max'])/2.
        dip2 = (parameters['dip2_min'] + parameters['dip2_max'])/2.
        rup2 = Rupture(typeF2, weight2, strike2, dip2)

        typeF3 = parameters['fault_type3']
        weight3 = parameters['rupture3_weight']
        strike3 = (parameters['strike3_min'] + parameters['strike3_max'])/2.
        dip3 = (parameters['dip3_min'] + parameters['dip3_max'])/2.
        rup3 = Rupture(typeF3, weight3, strike3, dip3)

        zone.ruptures = [rup1, rup2, rup3]

        new_zones.append(zone)

    return new_zones


def parse_seismic_activity_rates(zones, FILE_RATES):
    """Read seismic activity rates file.

    Parameters
    ----------
    FILE_RATES: str
        Pathname to seismic activity rates file.

    Returns
    -------
    zones : list
        List of pybus.sourcemodel.AreaSource object.
    """

    seismic_activity_rates = pd.read_csv(FILE_RATES, sep='\s+', dtype={'name':str})

    ###########
    # add mfd #
    ###########

    new_zones = []

    for i in range(len(seismic_activity_rates)):
        rates = seismic_activity_rates.iloc[i]
        try:
            index = [zone.name for zone in zones].index(rates['name'])
        except:
            print('AreaSource %s not in model.'%(rates['name']))
            break

        zone = zones[index]

        zone.mfd.mfit = rates['Mfit']
        zone.mfd.a_mean = rates['a']
        zone.mfd.b_mean = rates['b']
        zone.mfd.a_err = rates['err_a']
        zone.mfd.b_err = rates['err_b']
        try:
            zone.mfd.rho = rates['rho_a_b']
            zone.mfd.rho_err = rates['err_rho']
        except:
            zone.mfd.rho = None
            zone.mfd.rho_err = None

        new_zones.append(zone)

    return new_zones


def parse_zones_agregated(FILE_AGREGATION):
    """Parse zones agregated.

    Parameters
    ----------
    FILE_AGREGATION : str
        Pathname to agregated zones file.

    Returns
    -------
    zones_agregated : list
        List of lists of area names to agregate.
    """

    zones_agregated = []

    try:
        with open(FILE_AGREGATION) as f:
            for line in f:
                zones_agregated.append( line.split() )
    except IOError:
        print("WARNING: File is not accessible: %s" % FILE_AGREGATION)
        print("WARNING: No agregated zones for model %s.\n" % FILE_AGREGATION.split('/')[-1].split('.')[0].split('_')[-1])

    return zones_agregated


def parse_source_model_text(name, FILE_POLYGONS, FILE_PARAMETERS, FILE_RATES, FILE_AGREGATION=None):
    """Read text input files to construct the original source model.

    Parameters
    ----------
    name : str
        Name of the source model

    FILE_POLYGONS : str
        Pathname to polygons file

    FILE_PARAMETERS : str
        Pathname to seismic parameters file

    FILE_RATES : str
        Pathname to seismic activity rates file

    FILE_AGREGATION : str, optional
        Pathname to agregated zones file

    Returns
    -------
    sm : pybus.sourcemodel.SourceModel
        Source model of area sources.
    """

    zones = parse_polygons(FILE_POLYGONS)
    zones = parse_seismic_parameters(zones, FILE_PARAMETERS)
    zones = parse_seismic_activity_rates(zones, FILE_RATES)
    zones_agregated = parse_zones_agregated(FILE_AGREGATION)

    sm = SourceModel(name, zones, zones_agregated)

    return sm


def parse_source_model_xml(FILE_SM_XML):
    """ Read xml source model input files to construct the original source model.

    Parameters
    ----------
    FILE_SM_XML : str
        Pathname to xml source model file

    Returns
    -------
    sm : pybus.sourcemodel.SourceModel
        Source model of area sources.
    """

    # -------------
    # # Solution 1: using the OpenQuake library
    # import openquake.hazardlib.nrml as ohn
    #
    # tree = ohn.read(FILE_SM_XML)
    #
    # sm_name = tree.sourceModel['name']
    #
    # sm_areas = []
    # for areaSource in tree.sourceModel.sourceGroup:
    #     as_name = areaSource['id']
    #     as_extended_name = areaSource['name']
    #
    #     text = areaSource.areaGeometry.Polygon.exterior.LinearRing.posList.text
    #     coords = np.array([(float(lon), float(lat)) for lon, lat in zip(text[0::2], text[1::2])])
    #     as_polygon = sg.Polygon(coords)
    #
    #     as_upper_depth = float(areaSource.areaGeometry.upperSeismoDepth.text)
    #     as_lower_depth = float(areaSource.areaGeometry.lowerSeismoDepth.text)
    #
    #     a = float(areaSource.truncGutenbergRichterMFD['aValue'])
    #     b = float(areaSource.truncGutenbergRichterMFD['bValue'])
    #     mmin = float(areaSource.truncGutenbergRichterMFD['minMag'])
    #     mmax = float(areaSource.truncGutenbergRichterMFD['maxMag'])
    #     as_mfd = GRTruncated(amean=a, bmean=b, mmin=mmin, mmax=mmax)
    #
    #     as_ruptures = []
    #     for nodal in areaSource.nodalPlaneDist:
    #         rup = Rupture(strike=nodal['strike'], dip=nodal['dip'], rake=nodal['rake'], weight=nodal['probability'])
    #         as_ruptures.append(rup)
    #
    #     as = AreaSource(name=as_name, extended_name=as_extended_name,\
    #                     polygon=as_polygon, mfd=as_mfd, ruptures=as_ruptures,\
    #                     upper_seismogenic_depth=as_upper_depth,\
    #                     lower_seismogenic_depth=as_lower_depth)
    #
    #     sm_areas.append(as)
    #
    # sm = SourceModel(name=sm_name, zones=sm_areas)
    #
    # return sm

    # ----------
    # Solution 2: being independant from OpenQuake and using the classical ElementTree object from xml python library
    # Cette méthode n'est valable qu'avec un format impliquant un ordre precis dans attributs...
    # Amélioration possible en recherchant les attributs clés nécessaires.
    tree = ET.parse(FILE_SM_XML)
    root = tree.getroot()

    sourceModel = root[0]  # only 1 source model in a file
    sm_name = sourceModel.attrib.get('name')

    sm_areas = []
    for areaSource in sourceModel:
        as_name = areaSource.attrib.get('id')
        as_extended_name = areaSource.attrib.get('name')

        text = areaSource[0][0][0][0][0].text.split()
        coords = [(float(lon), float(lat)) for lon, lat in zip(text[0::2], text[1::2])]
        as_polygon = sg.Polygon(coords)

        as_upper_depth = float(areaSource[0][1].text)
        as_lower_depth = float(areaSource[0][2].text)

        a = float(areaSource[3].attrib.get('aValue'))
        b = float(areaSource[3].attrib.get('bValue'))
        mmin = float(areaSource[3].attrib.get('minMag'))
        mmax = float(areaSource[3].attrib.get('maxMag'))
        as_mfd = GRTruncated(a_mean=a, b_mean=b, mmin=mmin, mmax=mmax)

        as_ruptures = []
        for nodal in areaSource[4]:
            rup = Rupture(strike=float(nodal.attrib.get('strike')),\
                          dip=float(nodal.attrib.get('dip')),\
                          rake=float(nodal.attrib.get('rake')),\
                          weight=float(nodal.attrib.get('probability')))
            as_ruptures.append(rup)

        aso = AreaSource(name=as_name, extended_name=as_extended_name,\
                        polygon=as_polygon, mfd=as_mfd, ruptures=as_ruptures,\
                        upper_seismogenic_depth=as_upper_depth,\
                        lower_seismogenic_depth=as_lower_depth)

        sm_areas.append(aso)

    sm = SourceModel(name=sm_name, zones=sm_areas)

    return sm
