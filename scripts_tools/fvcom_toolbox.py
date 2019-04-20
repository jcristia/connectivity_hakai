"""
    Created on Sat Jun 21 12:35:15 2014
    Update log:
            Version 1.0
            Version 2.0
            Version 3.0 Massively updated and cleaned up
    @author: Pramod;

    Copyright (C) <2018>  <Pramod Thupaki: Pramod.Thupaki@hakai.org>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>
"""
import numpy as np
from netCDF4 import Dataset as ncdata
import datetime
import time
import sys
from matplotlib.dates import date2num
import pyproj
#
# Flatten a list of lists to get a list
#
flatten = lambda l: [item for sublist in l for item in sublist]

WGS84 = pyproj.Proj("+init=EPSG:4326")
UTM9N = pyproj.Proj("+init=EPSG:26909")


def convert_lonlat2xy(lon, lat):
    # convert from lon/lat to UTM9N
    # x = np.zeros((len(lon), 1), dtype=float)
    # y = np.zeros((len(lon), 1), dtype=float)
    x, y = pyproj.transform(WGS84, UTM9N, lon, lat)
    return x, y


def convert_xy2lonlat(ndxy):
    # convert from UTM9N to lon/lat
    lon, lat = pyproj.transform(UTM9N, WGS84, ndxy[:, 0], ndxy[:, 1])
    return lon, lat


def interp_data(v, zlev, zintp):
    #
    # read the raw values and interpolate to standard zlevels
    #
    # v, zlev have to be ascending order.
    # zlev[0], zintp[0] is value closest to surface; +ve upwards from surface
    from scipy import interpolate
    if zintp[0] > zlev[0]:
        v = np.append(v[0], v)
        zlev = np.append(zintp[0], zlev)
    if zintp[-1] < zlev[-1]:
        v = np.append(v, v[-1])
        zlev = np.append(zlev, zintp[-1])

    f = interpolate.interp1d(zlev, v, kind='linear')
    vintp = f(zintp)
    return vintp


def vertAvg(v, z, Ha, Hb):
    # ~ calculate the vertical averaged velocity for top H(m) given v(siglay,t) and d(siglev)
    vavg = 0.
    dzChk = 0.
    for i in range(len(z)):
        if Ha >= z[i + 1] and Ha < z[i]:
            vavg = vavg + v[i] * (Ha - z[i + 1])
            dzChk = dzChk + (Ha - z[i + 1])
            # ~ print 'a',i, v[i],z[i], (Ha - z[i+1])
        if Ha > z[i] and Hb < z[i + 1]:
            vavg = vavg + v[i] * (z[i] - z[i + 1])
            dzChk = dzChk + (z[i] - z[i + 1])
            # ~ print 'b',i, v[i],z[i], (z[i]-z[i+1])
        if Hb < z[i] and Hb >= z[i + 1]:
            vavg = vavg + v[i] * (z[i] - Hb)
            dzChk = dzChk + (z[i] - Hb)
            # ~ print 'c',i,v[i],z[i], (z[i] - Hb)
            break
    if abs(dzChk - (Ha - Hb)) > 0.1:
        print ("Failed dzCheck")
        sys.exit()
    vavg = vavg / (Ha - Hb)
    return vavg


def mjd2date(mjd, opt=1):
    from datetime import datetime, timedelta
    t0 = datetime(1858, 11, 17, 0, 0, 0)
    timeObj = t0 + timedelta(days=mjd)
    if opt == 0:
        dtstr = timeObj
    elif opt == 1:
        dtstr = timeObj.strftime('%Y-%m-%d %H:%M:%S')  # 2012-09-01 13:00:00
    elif opt == 2:
        dtstr = timeObj.strftime('%d-%b-%Y %H:%M:%S')  # 01-Sep-2012 13:00:00
    elif opt == 3:
        dtstr = timeObj.strftime('%Y-%m-%d')  # 2012-09-01
    # the general datetime object is datetime.fromtimestamp(unxTime)
    # which is a useful variable to keep on hand !
    return dtstr


def mjd2datetime(mjd):
    # return python datetime object corresponding to MJD
    from datetime import datetime, timedelta
    t0 = datetime(1858, 11, 17, 0, 0, 0)
    time = t0 + timedelta(days=mjd)
    # the general datetime object is datetime.fromtimestamp(unxTime)
    # which is a useful variable to keep on hand !
    return time


def unix2date(unxTime, opt):
    from datetime import datetime, timedelta
    unixOrigin = datetime(1970, 1, 1, 0, 0, 0)
    timeObj = unixOrigin + timedelta(seconds=unxTime)
    if opt == 0:
        dtstr = timeObj
    elif opt == 1:
        dtstr = timeObj.strftime('%Y-%m-%d %H:%M:%S')  # 2012-09-01 13:00:00
    elif opt == 2:
        dtstr = timeObj.strftime('%d-%b-%Y %H:%M:%S')  # 01-Sep-2012 13:00:00
    elif opt == 3:
        dtstr = timeObj.strftime('%Y-%m-%d')  # 2012-09-01
    # the general datetime object is datetime.fromtimestamp(unxTime)
    # which is a useful variable to keep on hand !
    return dtstr


def matlab_to_python_datetime(matlab_datenum):
    from datetime import datetime, timedelta
    return datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum%1) - timedelta(days=366)


def convertNodal2ElemVals(nv, v):
    if nv.shape[0] == 3:
        print 'converting nv shape', nv.shape
        nv = nv.transpose()
    nvm1 = nv - 1
    vc = (v[nvm1[:, 0]] + v[nvm1[:, 1]] + v[nvm1[:, 2]]) / 3.
    return vc


def gen_elev_timeseries(obc_amp, obc_phs, tidalPeriod, dt, ntime):
    # create elevation timeseries from phase and amplitude information in the file
    # dt should be in hours (time step for timeseries output)
    # obc_phs in Deg, obc_amp in meters,
    # tidalPeriod in hours (S2 = 12.00 hours)
    # ntime is number of time steps for output time series

    pi = np.pi
    deg2rad = 2. * pi / 360.
    freq = 1.0 / np.asarray(tidalPeriod).astype(float)
    nobc = obc_amp.shape[1]
    ncon = obc_amp.shape[0]
    print 'ncon, nobc', ncon, nobc, np.shape(obc_amp)
    elev = np.zeros((ntime, nobc), dtype=float)
    times = np.arange(0, ntime)*dt
    for i in range(ncon):
        for e in range(nobc):
            elev[:, e] = elev[:, e] + obc_amp[i, e] * np.cos(2.0 * pi * times * freq[i] - deg2rad * (obc_phs[i, e]))
    return times, elev


def make_julian_elev_forcing(casename, obclist, elev, date0, dt):
    # -----------------------------------------------------------------------
    #  make julian forcing file
    # -----------------------------------------------------------------------
    nobc = len(obclist)
    ntimes = np.shape(elev)[0]
    # print ('ntimes =', ntimes)
    if nobc != np.shape(elev)[1]:
        print ("elev matrix not correct!")
        sys.exit()
    if casename[-3:] == '.nc':
        fout = casename
    else:
        fout = '{:s}_{:s}_julian_obc.nc'.format(casename, date0[0:10])
    # print ('fout=', fout)
    # open netcdf file
    ncfile = ncdata(fout, mode='w', format='NETCDF3_CLASSIC')
    # write global attributes
    setattr(ncfile, 'type', 'FVCOM TIME SERIES ELEVATION FORCING FILE')
    setattr(ncfile, 'title', 'JULIAN FVCOM TIDAL FORCING DATA CREATED FROM OLD FILE TYPE')
    setattr(ncfile, 'source', 'fvcom grid (unstructured) surface forcing')
    setattr(ncfile, 'history', 'created by Python function: makeJulianElevForcing')
    # dimensions
    ncfile.createDimension('time', None)
    ncfile.createDimension('nobc', nobc)
    ncfile.createDimension('DateStrLen', 26)
    # write the time variables
    t = time.strptime(date0, '%Y-%m-%d %H:%M:%S')
    t = datetime.datetime(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min)
    L = [t + n * datetime.timedelta(hours=dt) for n in range(0, ntimes)]
    dtStr = [i.strftime('%Y-%m-%d %H:%M:%S') for i in L]
    addTimeVariables(ncfile, dtStr)
    # write attributes of variables
    obc_nodes = ncfile.createVariable('obc_nodes', 'int32', ('nobc',))
    setattr(obc_nodes, 'long_name', 'Open Boundary Node Number')
    setattr(obc_nodes, 'grid', 'obc_grid')
    obc_nodes[:] = obclist
    # write attributes of variables
    elevation = ncfile.createVariable('elevation', 'float32', ('time', 'nobc',))
    setattr(elevation, 'long_name', 'Open Boundary Elevation')
    setattr(elevation, 'units', 'meters')
    elevation[:] = elev

    ncfile.close()
    return fout


def readN(fin, N):
    # -----------------------------------------------------------------------
    # new faster way of reading the N floats
    # -----------------------------------------------------------------------
    i = 0
    L = ''
    while i < N:
        l = fin.readline()
        i = i + len(l.split())
        L = L + l
    val = np.asarray(L.split()).astype(float)
    return val


def find_closest(X, Y, x, y):
    # -------------------------------------------------------------------------
    # finds closest points without any conversion from latLon to geo
    # ------------------------------------------------------------------------
    N = len(x) #number of points
    pos = []
    ds = []
    for i in np.arange(0, N):
        DS = np.sqrt(np.square(X - x[i]) + np.square(Y - y[i]))
        pos.append(np.argmin(DS))
        ds.append(np.min(DS))
    pos = np.asarray(pos)
    ds = np.asarray(ds)
    return pos, ds


def find_closest_node(trimesh, x, y, latlon=True):
    if latlon:
        x, y = pyproj.transform(WGS84, UTM9N, x, y)
    pos, ds = find_closest(trimesh.x, trimesh.y, [x], [y])
    # convert array position to node number!
    return pos+1


def find_closest_element(trimesh, x, y, latlon=True):
    xc = convertNodal2ElemVals(trimesh.triangles+1, trimesh.x)
    yc = convertNodal2ElemVals(trimesh.triangles+1, trimesh.y)
    if latlon:
        x, y = pyproj.transform(WGS84, UTM9N, x, y)
    pos, ds = find_closest(xc, yc, [x], [y])
    # convert array position to element number!
    return pos+1


def addTimeVariables(ncfile, dtStr):
    # -----------------------------------------------------------------------
    # function to add the time variables in correct format
    #   assumes the necessary dimensions (time, DateStrLen) are already declared
    # -----------------------------------------------------------------------
    mjd = mjDate(dtStr)
    # Create and write the Time string
    data = ncfile.createVariable('Times', 'S1', ('time', 'DateStrLen',))
    setattr(data, 'format', 'modified julian day (MJD)')
    setattr(data, 'time_zone', 'UTC')
    for i in range(len(dtStr)):
        for j in range(len(dtStr[i])):
            data[i, j] = dtStr[i][j]
        # Create and write the time variable
    data = ncfile.createVariable('time', 'float64', ('time',))
    setattr(data, 'long_name', 'Time')
    setattr(data, 'units', 'days since 1858-11-17 00:00:00')
    setattr(data, 'time_zone', 'UTC')
    data[:] = mjd
    itime = np.floor(mjd).astype(int)
    itime2 = []
    for s in dtStr:
        t = time.strptime(s, '%Y-%m-%d %H:%M:%S')
        itime2.append(t.tm_hour * 3600 * 1000 + t.tm_min * 60 * 1000)
    itime2 = np.asarray(itime2).astype(int)
    # write Itime
    data = ncfile.createVariable('Itime', 'int32', ('time',))
    setattr(data, 'long_name', 'Time')
    setattr(data, 'units', 'days since 1858-11-17 00:00:00')
    setattr(data, 'time_zone', 'UTC')
    data[:] = itime
    # write Itime2
    data = ncfile.createVariable('Itime2', 'int32', ('time',))
    setattr(data, 'long_name', 'Time')
    setattr(data, 'units', 'msec since 00:00:00')
    setattr(data, 'time_zone', 'UTC')
    data[:] = itime2


def mjDate(dtStr, te=[1858, 11, 17, 0, 0, 0]):
    # ------------------------------------------------------------------------
    # converts dt string in format '%Y-%m-%d %H:%M:%S' to modified julian date
    # input has to be as list ['','',''] EVEN if single time is given as input!
    # ------------------------------------------------------------------------
    # ~ t0 = datetime.datetime(1858,11,17,0,0,0)
    t0 = datetime.datetime(te[0], te[1], te[2], te[3], te[4])
    mjd = []
    for s in dtStr:
        t = time.strptime(s, '%Y-%m-%d %H:%M:%S')
        t = datetime.datetime(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min)
        n = date2num(t) - date2num(t0)
        mjd.append(n)
    return np.asarray(mjd)


def readBathy(fBathy):
    # ------------------------------------------------------------------------
    # reads the bathymetry file in FVCOM 3 format. first line is ignored (header)
    # depth data is expected in third column (first two can be node x and y)
    # and +ve is downward from WSE
    # ------------------------------------------------------------------------
    if '_dep.dat' not in fBathy:
        fBathy = fBathy + '_dep.dat'
    bathy = np.genfromtxt(fBathy, dtype='float32', usecols=(2), skip_header=1)
    return bathy


def read_its_V2(fits):
    # ------------------------------------------------------------------------
    # read the observed its in fvcom 2 format.
    # ------------------------------------------------------------------------
    its = np.genfromtxt(fits, dtype='float32', skip_header=2)
    nn, nzlev = np.shape(its)
    T = its[0:nn:2, :]
    S = its[1:nn:2, :]
    T = np.transpose(T)
    S = np.transpose(S)
    return T, S


def readObcList(fObcList):
    # ------------------------------------------------------------------------
    # reads the list of nodes located on open boundaries.
    # no header single column. integer node numbers. other columns if present
    # will be ignored
    # ------------------------------------------------------------------------
    obcNodes = np.genfromtxt(fObcList, dtype='int32', usecols=(0))
    return obcNodes


def readObcFile_V3(fObc):
    # ------------------------------------------------------------------------
    #  read the obc file (fvcom version 3.X format)
    # ------------------------------------------------------------------------
    if '_obc.dat' not in fObc:
        fObc = fObc + '_obc.dat'
    obcNodes = np.genfromtxt(fObc, dtype='int32', usecols=(1), skip_header=1)
    return obcNodes


def readNgh(fngh):
    # ------------------------------------------------------------------------
    # reads the ngh format for converting with node lat lon
    # node connectivity and resulting element list are not generated
    # ------------------------------------------------------------------------
    "Usage: [lon, lat]=readFvcomNgh('grdFiles/kit4')"
    if not '_ngh.dat' in fngh:
        fngh = fngh + '_ngh.dat'
    a = np.genfromtxt(fngh, dtype='float32', usecols=(1, 2), skip_header=3)
    lon = a[:, 0]
    lat = a[:, 1]
    return lon, lat


def readMesh_V2(fMesh):
    # ------------------------------------------------------------------------
    # reads unstructured (triangle) mesh in fvcom-2.X format
    # returns the element connectivity matrix nv and node locations
    # ------------------------------------------------------------------------
    fin = open(fMesh, 'r')
    C = fin.readlines()
    last = 0
    nnode = 0
    nelem = 0
    # calculate number of nodes and elements in mesh
    for i in range(len(C)):
        n = int(C[i].split()[0])
        if n > last:
            last = n
        else:
            if nelem == 0:
                nelem = last
                last = 0
    nnode = n
    # find mesh connectivity
    nv = []
    ndxy = []
    for i in range(nelem):
        nv.append(C[i].split()[1:4])
    nv = np.asarray(nv)
    # read the node locations
    for i in range(nnode):
        ndxy.append(C[i + nelem].split()[1:3])
    ndxy = np.asarray(ndxy)

    nv = nv.astype(int)
    ndxy = ndxy.astype(float)

    return nv, ndxy


def readMesh_V3(fMesh):
    # ------------------------------------------------------------------------
    # reads unstructured (triangle) mesh in fvcom-3.X format
    # returns the element connectivity matrix nv and node locations
    # ------------------------------------------------------------------------
    if '_grd.dat' not in fMesh:
        fMesh = fMesh.strip() + '_grd.dat'
    with open(fMesh, 'r') as fid:
        nnode = int(fid.readline().split('=')[1])
        nelem = int(fid.readline().split('=')[1])
        elems = np.zeros((nelem, 3), dtype=int)
        nodes = np.zeros((nnode, 2), dtype=float)
        for i in range(nelem):
            elems[i, :] = np.asarray(fid.readline().split()[1:4], dtype=int)
        for i in range(nnode):
            nodes[i, :] = np.asarray(fid.readline().split()[1:3], dtype=float)
    return elems, nodes


def do_geom(nsiglev, nsiglay, spow):
    # ------------------------------------------------------------------------
    # Calculates the GEOMETRIC sigma levels and layers.
    # author : Jason Chaffey
    # ------------------------------------------------------------------------
    siglev = np.empty([nsiglev], dtype=np.float32)
    siglay = np.empty([nsiglay], dtype=np.float32)
    for nn in range((nsiglev + 1) / 2):
        siglev[nn] = -1.0 * np.power(float(nn) / float((nsiglev + 1) / 2 - 1), spow) / 2.0
    nn = (nsiglev + 1) / 2
    while nn < nsiglev:
        siglev[nn] = np.power(float(nsiglev - nn - 1) / float((nsiglev + 1) / 2 - 1), spow) / 2.0 - 1.0
        nn = nn + 1
    for nn in range(nsiglev - 1):
        siglay[nn] = (siglev[nn] + siglev[nn + 1]) / 2.0

    return siglev, siglay


def do_uniform(nsiglev, nsiglay):
    # ------------------------------------------------------------------------
    # Calculates the UNIFORM sigma levels and layers.
    # author : Jason Chaffey
    # ------------------------------------------------------------------------
    siglev = np.empty([nsiglev], dtype=np.float32)
    siglay = np.empty([nsiglay], dtype=np.float32)
    siginc = 1.0 / float(nsiglay)
    for nn in range(nsiglev):
        siglev[nn] = -1.0 * float(nn) * siginc
    for nn in range(nsiglev - 1):
        siglay[nn] = (siglev[nn] + siglev[nn + 1]) / 2.0

    return siglev, siglay


def do_tanh(nsiglev, du2, dl2):
    # -----------------------------------------------------------------------
    # calculates the TANH sigma coordinate levels and layers
    # -----------------------------------------------------------------------
    nsiglay = nsiglev - 1
    siglev = np.zeros((nsiglev, 1), 'float')
    siglay = np.zeros((nsiglay, 1), 'float')
    for k in range(nsiglay):
        x1 = dl2 + du2
        x1 = x1 * (nsiglay - k) / nsiglay
        x1 = x1 - dl2
        x1 = np.tanh(x1)
        x2 = np.tanh(dl2)
        x3 = x2 + np.tanh(du2)
        siglev[k] = (x1 + x2) / x3 - 1.0e0
    siglev[nsiglev - 1] = -1.0e0

    for i in range(nsiglay):
        siglay[i] = (siglev[i] + siglev[i + 1]) / 2.0e0

    return siglev, siglay


def do_generalized(nsiglev, du2, dl2, hmin, fdep):
    # -----------------------------------------------------------------------
    # TPP - Debug this !!
    # calculates the GENERALIZED sigma coordinate levels and layers
    # -----------------------------------------------------------------------
    dep = readBathy(fdep)
    nsiglay = nsiglev - 1
    siglev = np.zeros((nsiglev, 1), 'float')
    siglay = np.zeros((nsiglay, 1), 'float')
    for k in range(nsiglay):
        x1 = dl2 + du2
        x1 = x1 * (nsiglay - k) / nsiglay
        x1 = x1 - dl2
        x1 = np.tanh(x1)
        x2 = np.tanh(dl2)
        x3 = x2 + np.tanh(du2)
        siglev[k] = (x1 + x2) / x3 - 1.0e0
    siglev[nsiglev - 1] = -1.0e0
    for i in range(nsiglay):
        siglay[i] = (siglev[i] + siglev[i + 1]) / 2.0e0
    return siglev, siglay


def read_sigma(fname):
    # ------------------------------------------------------------------------
    # Gets the type of sigma coordinates and the number and values of
    #   layers and levels.
    # Currently supports: 1) Uniform
    #                     2) Geometric
    #                     3) Tanh  -  Pramod Thupaki
    #                     4) Generalized - Pramod Thupaki
    # author : Jason Chaffey; Pramod Thupaki
    # ------------------------------------------------------------------------
    if '_sigma.dat' not in fname:
        fname = fname + '_sigma.dat'

    fn = open(fname, "r")
    nsiglev = 0
    for line in fn:
        line.strip()
        fields = line.split("=")
        dummy = fields[0]
        if dummy[0:22] == "NUMBER OF SIGMA LEVELS":
            nsiglev = int(fields[1])
            nsiglay = nsiglev - 1
            print ("Number of sigma levels is ", nsiglev)
    if nsiglev == 0:
        print ("Error! Could not find NUMBER OF SIGMA LEVELS line!")
        sys.exit()
    fn.seek(0)
    sname = ""
    for line in fn:
        line.strip()
        fields = line.split("=")
        dummy = fields[0]
        if dummy[0:21] == "SIGMA COORDINATE TYPE":
            sname = fields[1]
    if len(sname) == 0:
        print ("Error! Could not find SIGMA COORDINATE TYPE line!")
        sys.exit()
    fn.seek(0)
    spow = 0.0
    #  if sname[0:9] == "GEOMETRIC":
    if "UNIFORM" in sname:
        spow = 1.0
        siglev, siglay = do_uniform(nsiglev, nsiglay)
    #    print siglev
    elif "GEOMETRIC" in sname:
        for line in fn:
            line.strip()
            fields = line.split("=")
            dummy = fields[0]
            if dummy[0:11] == "SIGMA POWER":
                spow = float(fields[1])
        print ("GEOMETRIC Sigma power is", spow)

        if spow == 0:
            print ("Error! Could not find SIGMA POWER line!")
            sys.exit()

        siglev, siglay = do_geom(nsiglev, nsiglay, spow)
    elif "TANH" in sname:
        for line in fn:
            fields = line.split('=')
            if fields[0].strip() == 'DU':
                du2 = float(fields[1])
            elif fields[0].strip() == 'DL':
                dl2 = float(fields[1])
        siglev, siglay = do_tanh(nsiglev, du2, dl2)
    elif "GENERALIZED" in sname:
        for line in fn:
            fields = line.split('=')
            if fields[0].strip() == 'DU':
                du2 = float(fields[1])
            elif fields[0].strip() == 'DL':
                dl2 = float(fields[1])
        siglev, siglay = do_generalized(nsiglev, du2, dl2, hmin, fname + '_dep.dat')
    else:
        print ("ERROR! Unknown SIGMA COORDINATE TYPE:", sname)
        sys.exit()

    fn.close()
    return nsiglev, siglev, nsiglay, siglay


def make_grd_V3(fout, nv, ndxy):
    # ---------------------------------------------------------------------
    # creates the _grd file for fvcom-3.X
    # can be used for converting version 2 grid file to version 3
    # ---------------------------------------------------------------------
    if "_grd.dat" not in fout:
        fout = fout.strip() + "_grd.dat"

    with open(fout, 'w') as FO:
        FO.write('Node Number = {:d}\n'.format(len(ndxy)))
        FO.write('Cell Number = {:d}\n'.format(len(nv)))
        for i in range(len(nv)):
            FO.write('{:d} {:d} {:d} {:d}\n'.format(i + 1, nv[i, 0], nv[i, 1], nv[i, 2]))

        for i in range(len(ndxy)):
            FO.write('{:d} {:f} {:f}\n'.format(i + 1, ndxy[i, 0], ndxy[i, 1]))
    return 1


def make_obc_V3(fout, obcNodes, obcType):
    # ---------------------------------------------------------------------
    # creates the _obc file for fvcom-3.X
    # ---------------------------------------------------------------------
    if "_obc.dat" not in fout:
        fout = fout.strip() + '_obc.dat'

    with open(fout, 'w') as FO:
        FO.write('OBC Node Number = {:d}\n'.format(len(obcNodes)))
        for n in range(len(obcNodes)):
            l = '{0:4d} {1:6d} {2:2d}'.format(n + 1, obcNodes[n], obcType[n])
            FO.write(l + '\n')
    return 1


def make_spg_V3(casename, obcNodes, W, d):
    # ------------------------------------------------------------------------
    # creates the _spg.dat file with sponge layer information for fvcom-3.X
    # ------------------------------------------------------------------------
    "Usage: readFvcomNgh('grdFiles/kit4', kit4_obc.dat, W(idth), d(amping))"
    FOUT = casename + '_spg.dat'
    nobc = len(obcNodes)
    print ('nobc:', nobc)
    FO = open(FOUT, 'w')
    FO.write('Sponge Node Number = {:d}\n'.format(nobc))
    if W == 0:
        W = 15000.0
        d = 0.0001
    i = 0
    for n in obcNodes:
        l = '  {:7d} {:f} {:f}'.format(n, W, d)
        FO.write(l + '\n')
    FO.close()
    return 1


def convert_nonJulObc(casename, fObcListV2, fNJobc, date0):
    # -------------------------------------------------------------------------------
    # convert the elObc from fvcom version 2 ascii format to netcdf file
    # -------------------------------------------------------------------------------
    # read tidal constituents
    obcNodes = readObcFile_V3(fObcListV2)
    nObc = len(obcNodes)
    amp = np.zeros((8, nObc), 'float')
    phs = np.zeros((8, nObc), 'float')
    z0 = np.zeros((nObc), 'float')
    # read the old fvcom format
    fin = open(fNJobc, 'r')
    l = fin.readline()
    n = int(fin.readline())
    if n != nObc:
        print ("error. obc file and el_obc files do not have same number of nodes!")
        sys.exit()
    for i in range(nObc):
        z0[i] = float(fin.readline().split()[1])
        amp[:, i] = readN(fin, 8)
        phs[:, i] = readN(fin, 8)
    amp = amp * 0.01
    if make_nonJulianObc_V3(casename, obcNodes, amp, phs, z0, date0):
        print ("Finished converting el_obc from 2->3\n")


def convert_nonJulObc_c(casename, fObcList, fNJobc, date0):
    # -------------------------------------------------------------------------------
    # convert the elObc from custom ascii format to netcdf file
    # output order
    #               S2      M2       N2       K2        K1       P1      O1        Q1
    # -------------------------------------------------------------------------------
    obcNodes = readObcList(fObcList)
    fin = open(fNJobc, 'r')
    data = fin.readlines()
    amp = np.zeros((9, len(obcNodes)))
    phs = np.zeros((9, len(obcNodes)))

    for l in data:
        if 'S2amp' in l:  # S2
            amp[0, :] = np.asarray(l.split()[1:]).astype(float)
        elif 'S2phs' in l:
            phs[0, :] = np.asarray(l.split()[1:]).astype(float)
        elif 'M2amp' in l:  # M2
            amp[1, :] = np.asarray(l.split()[1:]).astype(float)
        elif 'M2phs' in l:
            phs[1, :] = np.asarray(l.split()[1:]).astype(float)
        elif 'N2amp' in l:  # N2
            amp[2, :] = np.asarray(l.split()[1:]).astype(float)
        elif 'N2phs' in l:
            phs[2, :] = np.asarray(l.split()[1:]).astype(float)
        elif 'K2amp' in l:  # K2
            amp[3, :] = np.asarray(l.split()[1:]).astype(float)
        elif 'K2phs' in l:
            phs[3, :] = np.asarray(l.split()[1:]).astype(float)
        elif 'K1amp' in l:  # K1
            amp[4, :] = np.asarray(l.split()[1:]).astype(float)
        elif 'K1phs' in l:
            phs[4, :] = np.asarray(l.split()[1:]).astype(float)
        elif 'P1amp' in l:  # P1
            amp[5, :] = np.asarray(l.split()[1:]).astype(float)
        elif 'P1phs' in l:
            phs[5, :] = np.asarray(l.split()[1:]).astype(float)
        elif 'O1amp' in l:  # O1
            amp[6, :] = np.asarray(l.split()[1:]).astype(float)
        elif 'O1phs' in l:
            phs[6, :] = np.asarray(l.split()[1:]).astype(float)
        elif 'Q1amp' in l:  # Q1
            amp[7, :] = np.asarray(l.split()[1:]).astype(float)
        elif 'Q1phs' in l:
            phs[7, :] = np.asarray(l.split()[1:]).astype(float)
        elif 'Z0amp' in l:  # Z0
            amp[8, :] = np.asarray(l.split()[1:]).astype(float)

    # ~ A = np.genfromtxt(fNJobc, dtype='float', usecols=range(1,nObc+1),delimiter=',',skip_header=3, missing_values=0)
    # ~ amp = A[0:nc2:2,:]
    # ~ phs = A[1:nc2:2,:]
    # ~ z0  = A[nc2,:]

    if make_nonJulianObc_V3(casename, obcNodes, amp, phs, z0, date0):
        print ("Finished converting el_obc from 2 -> 3\n")


# ------------------------------------------------------------------------------
#  NB: adapted from make_obc.py by Jason Chaffey
#  creates the _el_obc.nc file for fvcom-3.X
# ------------------------------------------------------------------------------
def make_nonJulianObc_V3(casename, obcNodes, amp, phs, z0, date0):
    nObc = len(obcNodes)
    if nObc != np.shape(amp)[1]:
        print("Error with array shape of amplitude")
        print(nObc, np.shape(amp))
    ncons = 8
    nc2 = ncons * 2
    # fix tidal periods etc.
    #               S2      M2       N2       K2        K1       P1      O1        Q1
    tPer = [43200.0, 44712.0, 45570.0, 43082.0, 86164.0, 86637.0, 92950.0, 96726.0]  # time period in seconds
    eqiAmp = [.112743, .242334, .046397, .030684, .141565, .046848, .100661, .019273]
    eqiBeta = [0.693, 0.693, 0.693, 0.693, 0.736, 0.706, 0.695, 0.695]
    # write netcdf file for nonJulian tidal forcing
    outfile = ncdata(casename + '_nonjulian_elobc.nc', 'w', format='NETCDF3_CLASSIC')
    setattr(outfile, 'type', 'FVCOM SPECTRAL ELEVATION FORCING FILE')
    setattr(outfile, 'title', 'Spectral forcing data')
    setattr(outfile, 'components', 'S2 M2 N2 K2 K1 P1 O1 Q1')
    setattr(outfile, 'history', 'created using fvcom_pyutils')
    # create the time/lat/lon dimensions.
    outfile.createDimension('nobc', nObc)
    outfile.createDimension('DateStrLen', 26)
    outfile.createDimension('tidal_components', ncons)
    # OBC nodes
    data = outfile.createVariable('obc_nodes', 'int32', ('nobc',))
    setattr(data, 'long_name', 'Open Boundary Node Number')
    setattr(data, 'grid', 'obc_grid')
    data[:] = obcNodes
    # reference elevations
    data = outfile.createVariable('tide_Eref', 'float32', ('nobc',))
    setattr(data, 'long_name', 'tidal elevation reference level')
    setattr(data, 'units', 'meters')
    data[:] = z0
    # tide_period
    data = outfile.createVariable('tide_period', 'float32', ('tidal_components',))
    setattr(data, 'long_name', 'tide angular period')
    setattr(data, 'units', 'seconds')
    data[:] = tPer
    # tide amplitudes
    data = outfile.createVariable('tide_Eamp', 'float32', ('tidal_components', 'nobc',))
    setattr(data, 'long_name', 'tidal elevation amplitude')
    setattr(data, 'units', 'meters')
    data[:] = amp
    # tide phases
    data = outfile.createVariable('tide_Ephase', 'float32', ('tidal_components', 'nobc',))
    setattr(data, 'long_name', 'tidal elevation phase angle')
    setattr(data, 'units', 'degrees, time of maximum elevation with respect to chosen time origin')
    data[:] = phs
    # equilibrium tide beta
    data = outfile.createVariable('equilibrium_beta_love', 'float32', ('tidal_components',))
    data[:] = eqiBeta
    setattr(data, 'formula', 'beta=1+klove-hlove')
    # equilibrium tide amplitude
    data = outfile.createVariable('equilibrium_tide_Eamp', 'float32', ('tidal_components',))
    setattr(data, 'long_name', 'equilibrium tidal elevation amplitude')
    setattr(data, 'units', 'metres')
    data[:] = eqiAmp
    # time variable fix origin time
    data = outfile.createVariable('time_origin', 'S1', ('DateStrLen',))
    setattr(data, 'format', 'modified julian day (MJD)')
    setattr(data, 'long_name', 'time')
    setattr(data, 'time_zone', 'UTC')
    for i in range(len(date0)):
        data[i] = date0[i]
    # close file
    outfile.close()
    return 1


def make_tsobc_V3(casename, fSigma, obcNodes, bathy, dtStr, temps, sals):
    # ------------------------------------------------------------------------------
    # creates the _tsobc.nc file for fvcom-3.X
    #  NB: adapted from make_obc.py by Jason Chaffey
    # ------------------------------------------------------------------------------
    if type(dtStr) == type('string'):
        dtStr = [dtStr]
    mjd = mjDate(dtStr)
    # ~ print mjd
    nsiglev, siglev, nsiglay, siglay = read_sigma(fSigma)
    ncfile = ncdata(casename + '_tsobc.nc', 'w', format='NETCDF3_CLASSIC')
    # create dimensions
    nobc = len(obcNodes)
    ncfile.createDimension('nobc', nobc)
    ncfile.createDimension('DateStrLen', 26)
    ncfile.createDimension('time', None)
    ncfile.createDimension('siglay', nsiglay)
    ncfile.createDimension('siglev', nsiglev)
    setattr(ncfile, 'type', 'FVCOM TIME SERIES OBC TS FILE')
    setattr(ncfile, 'title', 'FVCOM HYDROGRAPHIC OPEN BOUNDARY FORCING FILE')
    setattr(ncfile, 'history', 'created by ---')
    # add time variables
    addTimeVariables(ncfile, dtStr)
    # write obc nodes
    data = ncfile.createVariable('obc_nodes', 'int32', ('nobc',))
    data[:] = obcNodes
    setattr(data, 'long_name', 'Open Boundary Node Number')
    setattr(data, 'grid', 'obc_grid')
    # write depth in m at obc nodes
    data = ncfile.createVariable('obc_h', 'float32', ('nobc',))
    setattr(data, 'long_name', 'Open Boundary Node Depth')
    setattr(data, 'units', 'm')
    setattr(data, 'grid', 'obc_grid')
    data[:] = bathy
    # write sigma levels
    siglevels = np.empty([nsiglev, nobc], dtype=np.float32)
    for jj in np.arange(nobc):
        for ii in np.arange(nsiglev):
            siglevels[ii, jj] = siglev[ii]

    data = ncfile.createVariable('obc_siglev', 'float32', ('siglev', 'nobc',))
    setattr(data, 'long_name', 'ocean_sigma/general_coordinate')
    setattr(data, 'grid', 'obc_grid')
    data[:] = siglevels
    # create & write sigma layers
    siglayers = np.empty([nsiglay, nobc], dtype=np.float32)
    for jj in np.arange(nobc):
        for ii in np.arange(nsiglay):
            siglayers[ii, jj] = siglay[ii]

    data = ncfile.createVariable('obc_siglay', 'float32', ('siglay', 'nobc',))
    data[:] = siglayers
    setattr(data, 'long_name', 'ocean_sigma/general_coordinate')
    setattr(data, 'grid', 'obc_grid')
    # write temperature at sigma layers for obc nodes
    data = ncfile.createVariable('obc_temp', 'float32', ('time', 'siglay', 'nobc',))
    data[:] = temps
    setattr(data, 'long_name', 'Sea Water Temperature')
    setattr(data, 'units', 'Celsius')
    setattr(data, 'grid', 'obc_grid')
    # write salinity at sigma layers for obc nodes
    data = ncfile.createVariable('obc_salinity', 'float32', ('time', 'siglay', 'nobc',))
    data[:] = sals
    setattr(data, 'long_name', 'Sea Water Salinity')
    setattr(data, 'units', 'PSU')
    setattr(data, 'grid', 'obc_grid')
    ncfile.close()
    return 1


def make_observed_its(casename, dtStr, zlevs, tmps, sals):
    # ------------------------------------------------------------------------
    # creates the _its.nc file for fvcom-3.X
    # NB: adapted from an earlier version by Jason Chaffey
    # ------------------------------------------------------------------------
    if type(dtStr) is type('string'):
        dtStr = [dtStr]
    mjd = mjDate(dtStr)
    nz, nnode = np.shape(tmps)
    print ('number of nodes in its field are ', nnode)
    ncfile = ncdata(casename + '_its.nc', 'w', format='NETCDF3_CLASSIC')
    ncfile.createDimension('DateStrLen', 26)
    ncfile.createDimension('time', None)
    ncfile.createDimension('node', nnode)
    ncfile.createDimension('ksl', nz)
    # write global attributes
    setattr(ncfile, 'type', 'FVCOM TIME SERIES OBSERVED TS FILE')
    setattr(ncfile, 'title', 'FVCOM OBSERVED TS FILE')
    setattr(ncfile, 'history', 'created by ---')
    # add time variables
    addTimeVariables(ncfile, dtStr)
    # write depth levels
    data = ncfile.createVariable('zsl', 'float32', ('ksl',))
    setattr(data, 'long_name', 'Standard Z Levels, Positive Up')
    setattr(data, 'units', 'm')
    data[:] = zlevs[:]
    # write temperature
    data = ncfile.createVariable('tsl', 'float32', ('time', 'ksl', 'node',))
    setattr(data, 'long_name', 'Observed Temperature Profiles')
    setattr(data, 'units', 'degrees C')
    data[0, :, :] = tmps
    # write salinity
    data = ncfile.createVariable('ssl', 'float32', ('time', 'ksl', 'node',))
    setattr(data, 'long_name', 'Observed Salinity Profiles')
    setattr(data, 'units', 'PSU')
    data[0, :, :] = sals
    # close file
    ncfile.close()


def read_discharge(fInp):
    # -------------------------------------------------------------------------
    # reads the discharge data from hydrology model
    #
    # return variables
    # len(lat) = len(lon) = nriv
    # shape(disch) = (ntimes,nriv)
    # len(dtStr)   = ntimes
    # -------------------------------------------------------------------------
    f = open(fInp, 'r')
    # read the hydrodylogy file
    C = f.readlines()
    dtStr = []
    disch = []
    lat = []
    lon = []
    h = []
    i = 0
    for l in C:
        if 'Time' in l:
            t = l.split(':', 1)[1]
            dtStr.append(t.strip())
            i = i + 1
            if i > 1:
                disch.append(h)
                h = []
        else:
            if i == 1:
                lon.append(float(l.split()[0]))
                lat.append(float(l.split()[1]))
            h.append(float(l.split()[2]))

    disch.append(h)

    disch = np.asarray(disch)
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    return lat, lon, disch, dtStr


def make_riv_V3(casename, dtStr, rivName, disch, tmps, sals):
    # ----------------------------------------------------------------------------
    #   setup river forcing file for fvcom version 3.X
    # ----------------------------------------------------------------------------
    ntimes, nriv = np.shape(disch)
    ncfile = ncdata(casename + '_riv.nc', mode='w', format='NETCDF3_CLASSIC')
    setattr(ncfile, 'type', 'FVCOM RIVER FORCING FILE')
    setattr(ncfile, 'title', 'Discharge data')
    # dimensions
    ncfile.createDimension('time', None)
    ncfile.createDimension('namelen', 40)
    ncfile.createDimension('rivers', nriv)
    ncfile.createDimension('DateStrLen', 26)
    # write the time variables
    addTimeVariables(ncfile, dtStr)
    # write river names
    data = ncfile.createVariable('river_names', 'S1', ('rivers', 'namelen'))
    for i in range(nriv):
        name = rivName[i]
        for j in range(len(name)):
            data[i, j] = name[j]
        # write river flux/discharge
    data = ncfile.createVariable('river_flux', 'float32', ('time', 'rivers',))
    setattr(data, 'long_name', 'river runoff volume flux')
    setattr(data, 'units', 'm^3s^-1')
    data[:] = disch
    # write temperature
    data = ncfile.createVariable('river_temp', 'float32', ('time', 'rivers',))
    setattr(data, 'long_name', 'river runoff temperature')
    setattr(data, 'units', 'Celsius')
    data[:] = tmps
    # write salinity
    data = ncfile.createVariable('river_salt', 'float32', ('time', 'rivers',))
    setattr(data, 'long_name', 'river runoff salinity')
    setattr(data, 'units', 'PSU')
    data[:] = sals
    # close file
    ncfile.close()
    return 1


def make_riv_nml(casename, rivLoc, rivName):
    # ------------------------------------------------------------------------------
    # makes namelist for rivers based on template (1)
    # river names are lat, lon in '{:012.6f}:{:012.6f}'.format(lat[i],lon[i])
    # distribution is uniform in vertical
    # nearest nodes are identified based on provided information
    # assumes river discharge data are in casename_riv.nc
    # ------------------------------------------------------------------------------
    fo = open(casename + '_riv.nml', 'w')
    #    rivLoc = findClosestGeo(ndlon,ndlat,rivlon,rivlat)
    for i in range(len(rivLoc)):
        name = rivName[i]  # '{:012.6f}:{:012.6f}'.format(rivlat[i],rivlon[i])
        fo.write('&NML_RIVER\n')
        fo.write('   RIVER_NAME          = "{:s}",\n'.format(name))
        fo.write('   RIVER_GRID_LOCATION = {:d},\n'.format(rivLoc[i]))
        fo.write('   RIVER_VERTICAL_DISTRIBUTION = "unifor" / \n\n')
    return 1


def setup_hfx_ncfile(casename, ne, nn):
    # -----------------------------------------------------------------------------------
    #       setup heat flux variables definition in netcdf file for fvcom 3.X
    # -----------------------------------------------------------------------------------
    # open netcdf file
    ncfile = ncdata(casename + '_atmFlx.nc', mode='w', format='NETCDF3_CLASSIC')
    # write global attributes
    setattr(ncfile, 'type', 'FVCOM Forcing')
    setattr(ncfile, 'source', 'fvcom grid (unstructured) surface forcing')
    setattr(ncfile, 'history', 'created by python FVCOM toolbox')
    # dimensions
    ncfile.createDimension('time', None)
    ncfile.createDimension('nele', ne)
    ncfile.createDimension('node', nn)
    ncfile.createDimension('DateStrLen', 26)
    # write attributes of air_temperature
    if 'air_temperature' not in ncfile.variables:
        data = ncfile.createVariable('air_temperature', 'float32', ('time', 'node',))
        setattr(data, 'long_name', 'Air Temperature')
        setattr(data, 'standard_name', 'Air Temperature')
        setattr(data, 'units', 'celsius')
        setattr(data, 'type', 'data')
    if 'air_pressure' not in ncfile.variables:
        data = ncfile.createVariable('air_pressure', 'float32', ('time', 'node',))
        setattr(data, 'long_name', 'Air Pressure')
        setattr(data, 'standard_name', 'Air Pressure')
        setattr(data, 'units', '0.01*Pa')
        setattr(data, 'type', 'data')
    if 'relative_humidity' not in ncfile.variables:
        data = ncfile.createVariable('relative_humidity', 'float32', ('time', 'node',))
        setattr(data, 'long_name', 'Relative_humidity')
        setattr(data, 'standard_name', 'Relative_humidity')
        setattr(data, 'units', '%')
        setattr(data, 'type', 'data')
    if 'cloud_cover' not in ncfile.variables:
        data = ncfile.createVariable('cloud_cover', 'float32', ('time', 'node',))
        setattr(data, 'long_name', 'cloud_cover')
        setattr(data, 'standard_name', 'cloud_cover')
        setattr(data, 'units', '%')
        setattr(data, 'type', 'data')
    if 'short_wave' not in ncfile.variables:
        data = ncfile.createVariable('short_wave', 'float32', ('time', 'node',))
        setattr(data, 'long_name', 'short_wave')
        setattr(data, 'standard_name', 'short_wave')
        setattr(data, 'units', 'W m-2')
        setattr(data, 'type', 'data')
    if 'long_wave' not in ncfile.variables:
        data = ncfile.createVariable('long_wave', 'float32', ('time', 'node',))
        setattr(data, 'long_name', 'short_wave')
        setattr(data, 'standard_name', 'short_wave')
        setattr(data, 'units', 'W m-2')
        setattr(data, 'type', 'data')
    return ncfile


def setup_wind_ncfile(casename, ne, nn):
    # -----------------------------------------------------------------------
    # setup wind forcing ncfile. attributes etc.
    # -----------------------------------------------------------------------
    # open netcdf file
    ncfile = ncdata(casename + '_wnd.nc', mode='w', format='NETCDF3_CLASSIC')
    # print ("Finished opening ncfile:wind")
    # write global attributes
    setattr(ncfile, 'type', 'FVCOM U10/V10')
    setattr(ncfile, 'source', 'fvcom grid (unstructured) surface forcing')
    setattr(ncfile, 'history', 'created by convert_wndForcing')
    # dimensions
    ncfile.createDimension('time', None)
    ncfile.createDimension('nele', ne)
    ncfile.createDimension('node', nn)
    ncfile.createDimension('DateStrLen', 26)
    # print ("Finished creating dimensions:wind")
    # write attributes of wind variables
    U10 = ncfile.createVariable('U10', 'float32', ('time', 'nele',))
    setattr(U10, 'long_name', 'Eastward Wind speed at 10m height')
    setattr(U10, 'standard_name', 'Wind speed')
    setattr(U10, 'units', 'm/s')
    setattr(U10, 'type', 'data')

    V10 = ncfile.createVariable('V10', 'float32', ('time', 'nele',))
    setattr(V10, 'long_name', 'Northward Wind speed at 10m height')
    setattr(V10, 'standard_name', 'Wind speed')
    setattr(V10, 'units', 'm/s')
    setattr(V10, 'type', 'data')

    # print ("Finished creating variales:wind")
    return ncfile


def convert_wndForcing(casename, fwnd, ne, nn, ntimes, date0):
    # -------------------------------------------------------------------------------
    # converts wind forcing files from version 2 ascii to version 3 netcdf file
    # date0 has to be in YYYY-MM-DD hh:mm:ss format
    # assumes data is hourly
    # -------------------------------------------------------------------------------
    ne2 = ne * 2
    nn2 = nn * 2
    fin = open(fwnd, 'r')
    ncfile = setup_wind_ncfile(casename, ne, nn)
    U10 = ncfile.variables['U10']
    V10 = ncfile.variables['V10']
    # write the time variables
    t = time.strptime(date0, '%Y-%m-%d %H:%M:%S')
    t = datetime.datetime(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min)
    L = [t + n * datetime.timedelta(hours=1) for n in range(0, ntimes)]
    dtStr = [i.strftime('%Y-%m-%d %H:%M:%S') for i in L]
    addTimeVariables(ncfile, dtStr)
    # read and write U10, V10
    for i in range(0, ntimes):
        thr = readN(fin, 1)
        v = readN(fin, ne2)
        U10[i, :] = v[0:ne2:2]
        V10[i, :] = v[1:ne2:2]
        print ('Writing {:s}_wnd.nc - for time:{:s}'.format(casename, dtStr[i]))
    print ("Finished converting wind file ! \n")
    ncfile.close()


def convert_rivForcing(casename, friv, date0):
    # ----------------------------------------------------------------------------
    # convert river forcing file from v2 to v3 fvcom netcdf format
    # ----------------------------------------------------------------------------
    fin = open(friv, 'r')
    # setup dateString
    t = time.strptime(date0, '%Y-%m-%d %H:%M:%S')
    t = datetime.datetime(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min)
    times = []
    # read text file with river discharge data
    l = fin.readline()
    nRiv = int(fin.readline().split()[0])
    print ('Number of rivers in file:', nRiv)
    rivLocNode = []
    rivNames = []
    for n in range(nRiv):
        l = fin.readline().split()
        rivLocNode.append(int(l[0]))
        rivNames.append(l[1].strip())
    print ("Skipping river distribution info. Uniform distribution will be used")

    for n in range(nRiv):
        l = fin.readline()
    # read number of number of timesnaps in file
    ntimes = int(fin.readline().split()[0])
    # read tmp, sal, dischage data for each river in order
    disch = np.zeros((ntimes, nRiv), 'float')
    rivTmp = np.zeros((ntimes, nRiv), 'float')
    rivSal = np.zeros((ntimes, nRiv), 'float')
    for n in range(ntimes):
        thr = float(fin.readline().split()[0])
        times.append(t + datetime.timedelta(hours=thr))
        disch[n, :] = readN(fin, nRiv)
        rivTmp[n, :] = readN(fin, nRiv)
        rivSal[n, :] = readN(fin, nRiv)
    # create datestring
    dtStr = [i.strftime('%Y-%m-%d %H:%M:%S') for i in times]
    #
    # call function to create river nml and nc files !
    #
    if make_riv_nml(casename, rivLocNode, rivNames):
        print ("Finished creating river nml file!\n")
    if make_riv_V3(casename, dtStr, rivNames, disch, rivTmp, rivSal):
        print ("Finished creating river nc file !\n")


def convert_hfxForcing(casename, fhfx, ne, nn, ntimes, date0):
    # ---------------------------------------------------------------------------------
    # convert heatflux forcing files from version 2 (ASCII) to version 3 nc file
    # ---------------------------------------------------------------------------------
    ne2 = ne * 2
    nn2 = nn * 2
    fin = open(fhfx, 'r')
    # open netcdf file
    ncfile = ncdata(casename + '_hfx.nc', mode='w', format='NETCDF3_CLASSIC')
    # write global attributes
    setattr(ncfile, 'type', 'FVCOM HEAT FLUX')
    setattr(ncfile, 'source', 'fvcom grid (unstructured) surface forcing')
    setattr(ncfile, 'history', 'created by convert_hfxForcing')
    # dimensions
    ncfile.createDimension('time', None)
    ncfile.createDimension('nele', ne)
    ncfile.createDimension('node', nn)
    ncfile.createDimension('DateStrLen', 26)
    # write the time variables
    t = time.strptime(date0, '%Y-%m-%d %H:%M:%S')
    t = datetime.datetime(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min)
    L = [t + n * datetime.timedelta(hours=1) for n in range(0, ntimes)]
    dtStr = [i.strftime('%Y-%m-%d %H:%M:%S') for i in L]
    addTimeVariables(ncfile, dtStr)
    # write attributes of wind variables
    netHfx = ncfile.createVariable('net_heat_flux', 'float32', ('time', 'node',))
    setattr(netHfx, 'long_name', 'Net surface heat flux')
    setattr(netHfx, 'units', 'Watts meter-2')
    setattr(netHfx, 'type', 'data')
    setattr(netHfx, 'grid', 'fvcom_grid')
    setattr(netHfx, 'positive', 'downward flux, heating')
    setattr(netHfx, 'negative', 'upward flux, cooling')

    swr = ncfile.createVariable('short_wave', 'float32', ('time', 'node',))
    setattr(swr, 'long_name', 'Net solar shortwave radiation flux')
    setattr(swr, 'units', 'Watts meter-2')
    setattr(swr, 'type', 'data')
    setattr(swr, 'grid', 'fvcom_grid')
    setattr(swr, 'positive', 'downward flux, heating')
    setattr(swr, 'negative', 'upward flux, cooling')
    # read and write net_heat_flux and short_wave
    for i in range(0, ntimes):
        thr = readN(fin, 1)
        v = readN(fin, nn2)

        netHfx[i, :] = v[0:nn2:2]
        swr[i, :] = v[1:nn2:2]

        print ('Writing_hfx.nc - for time:', dtStr[i])

    ncfile.close()
    print ("Finished converting _hfx.dat to <casename>_hfx.nc !\n")


def convert_spg(casename, fspg):
    # -----------------------------------------------------------------------------------
    # convert spg files from fvcom 2 to 3 format
    # -----------------------------------------------------------------------------------
    fin = open(fspg, 'r')
    fo = open(casename + '_spg.dat', 'w')
    spg = fin.readlines()
    spg[0] = 'Sponge Node Number = {:d}\n'.format(int(spg[0]))
    for i in spg:
        fo.write(i)
    print ("Finished converting spg file 2 -> 3!\n")
    fo.close()
    fin.close()


def convert_obc(casename, fobc, obcType):
    obcList = readObcList_V2(fobc)
    if make_obc_V3(casename, obcList, obcType):
        print ("Finished converting obc file 2 -> 3!\n")


def convert_dep(casename, fdep):
    # -----------------------------------------------------------------------------------
    # convert dep files from fvcom 2 to 3 format
    # -----------------------------------------------------------------------------------
    fin = open(fdep, 'r')
    fo = open(casename + '_dep.dat', 'w')
    bathy = fin.readlines()
    nnode = len(bathy)
    l = 'Node Number = {:d}\n'.format(nnode)
    bathy.insert(0, l)
    for i in bathy:
        fo.write(i)
    fo.close()
    fin.close()
    print ("Finished converting depth file 2 -> 3 !\n")


def convert_cor(casename, fcor):
    # -----------------------------------------------------------------------------------
    # convert dep files from fvcom 2 to 3 format
    # -----------------------------------------------------------------------------------
    fin = open(fcor, 'r')
    fo = open(casename + '_cor.dat', 'w')
    cor = fin.readlines()
    nnode = len(cor)
    l = 'Node Number = {:d}\n'.format(nnode)
    cor.insert(0, l)
    for i in cor:
        fo.write(i)
    fo.close()
    fin.close()
    print ("Finished converting cor file 2 -> 3 !\n")


def make_cor_V3(casename, ndxy, lat):
    # -----------------------------------------------------------------------------------
    #           make _cor.dat file - Version 3
    # -----------------------------------------------------------------------------------
    fo = open(casename + '_cor.dat', 'w')
    fo.write('Node Number = {:d}\n'.format(len(lat)))
    for i in range(len(ndxy)):
        fo.write('{:f} {:f} {:f}\n'.format(ndxy[i, 0], ndxy[i, 1], lat[i]))
    fo.close()


def make_dep_V3(casename, ndxy, dep):
    # -----------------------------------------------------------------------------------
    #           make _dep.dat file - Version 3
    # -----------------------------------------------------------------------------------
    if '_dep.dat' not in casename:
        casename = casename+'_dep.dat'
    with open(casename + '_dep.dat', 'w') as fo:
        fo.write('Node Number = {:d}\n'.format(len(dep)))
        for i in range(len(ndxy)):
            fo.write('{:f} {:f} {:f}\n'.format(ndxy[i, 0], ndxy[i, 1], dep[i]))


def convert_grd(casename, fgrd):
    # -----------------------------------------------------------------------------------
    # convert grd files from fvcom 2 to 3 format
    # -----------------------------------------------------------------------------------
    nv, ndxy = readMesh_V2(fgrd)
    if make_grd_V3(casename, nv, ndxy):
        print ("Finished converting grd file 2 -> 3 !\n")


def convert_ngh_lonlat_grd(casename):
    nv, ndxy = readMesh_V3(casename)
    ndlon, ndlat = readNgh(casename)
    ndxy[:, 0] = ndlon
    ndxy[:, 1] = ndlat
    make_grd_V3(casename + '_lonlat', nv, ndxy)


def convert_its(casename, fits, fzlev, date0):
    # -----------------------------------------------------------------------------------
    # convert its files from fvcom 2 to 3 format
    # -----------------------------------------------------------------------------------
    tmps, sals = read_its_V2(fits)
    nz = np.shape(tmps)[0]
    fin = open(fzlev, 'r')
    zlev = readN(fin, nz)
    print ('zlev(+ve up) in meters:\n', zlev)
    if max(zlev) > 0.0:
        print ("Maxval of zlev cannot be >0.0. \n\tExiting...")
        sys.exit()
    if make_observed_its(casename, date0, zlev, tmps, sals):
        print ("Finished converting observed its file 2 -> 3 !\n")


def rotate(x, y, theta):
    # -----------------------------------------------------------------------------------
    # rotate vector in clockwise direction
    # -----------------------------------------------------------------------------------
    from math import radians
    a = radians(theta);
    xp = x * np.cos(a) - y * np.sin(a)
    yp = x * np.sin(a) + y * np.cos(a)
    return xp, yp


def getAvgValZlev(ncfile, var, t, zlev):
    # ---------------------------------------------------------------
    #  variable values at z levels
    # ---------------------------------------------------------------
    nv = ncfile.variables['nv'][:, :]
    nv = np.transpose(nv)
    siglay = ncfile.variables['siglay'][:, 0]
    siglev = ncfile.variables['siglev'][:, 0]
    bathy = ncfile.variables['h'][:]
    zeta = ncfile.variables['zeta'][t, :]

    if var in ['temp', 'salinity']:  # at siglay
        v = ncfile.variables[var][t, lev, :]
        v = np.mean(v, axis=0)
        s = siglay
    elif var in ['km', 'kh', 'kq']:  # at siglev
        v = ncfile.variables[var][t, lev, :]
        v = np.log10(v)
        v = np.mean(v, axis=0)
        s = siglev
    elif var in ['u', 'v']:  # at siglay but element centroid
        v = ncfile.variables[var][t, lev, :]
        v = np.mean(v, axis=0)
        s = siglay
    elif var == 'rmsvel':
        v = ncfile.variables['v'][t, lev, :]
        u = ncfile.variables['u'][t, lev, :]
        v = v * v + u * u
        v = np.mean(v, axis=0)
        v = np.sqrt(v)
        s = siglay
        print ("rmsvel shape", np.shape(v))
    else:
        print ("dont know how to read this variable type")
        sys.exit(0)
    ## interpolate variable to zlev (change coordinate so that positive is up)
    zlev = float(zlev) * 1.0
    for i in range(np.shape(v)[1]):
        z = s * (bathy[i] + zeta[i]) + zeta[i]
        f = interp1d(z, v[:, i])
        v[i] = f(zlev)
    return v


def getValUV(ncfile, t, lev=0, wind=False):
    # ---------------------------------------------------------------
    #   variable UV values
    # ---------------------------------------------------------------
    t = t - 1
    lev = lev - 1
    if 'Times' in ncfile.variables:
        tstr = ncfile.variables['Times'][t, :]
        tstr = ''.join(tstr)
    else:
        tstr = ncfile.variables['time'][t]
        tstr = str(tstr)

    print ('Time = ', tstr)
    if wind:
        u = ncfile.variables['U10'][t, :]
        v = ncfile.variables['V10'][t, :]
    else:
        u = ncfile.variables['u'][t, lev, :]
        v = ncfile.variables['v'][t, lev, :]
    return u, v, tstr


def getValZlev(ncfile, var, t, zlev):
    # ----------------------------------------------------------------------------------
    # returns the value at a particular zlev (in meters) for a given time.
    # -----------------------------------------------------------------------------------
    tstr = ncfile.variables['Times'][t, :]
    tstr = ''.join(tstr)
    tstr = tstr[0:10] + ' Hr:' + tstr[11:13]
    nv = ncfile.variables['nv'][:, :]
    nv = np.transpose(nv)
    siglay = ncfile.variables['siglay'][:, 0]
    siglev = ncfile.variables['siglev'][:, 0]
    bathy = ncfile.variables['h'][:]
    nnode = len(ncfile.dimensions['node'])
    nelem = len(ncfile.dimensions['nele'])
    zeta = ncfile.variables['zeta'][t, :]
    if var in ['km', 'kh', 'kq']:  # at siglev
        v = ncfile.variables[var][t, :, :]
        s = siglev
        vz = np.zeros((nnode, 0), dtype='float')
    elif var in ['u', 'v']:  # at siglay but element centroid
        v = ncfile.variables[var][t, :, :]
        s = siglay
        vz = np.zeros((nelem, 1), dtype='float')
        bathy = convertNodal2ElemVals(nv, bathy)
        zeta = convertNodal2ElemVals(nv, zeta)
    elif var in ['temp', 'salinity']:  # at siglay
        v = ncfile.variables[var][t, :, :]
        s = siglay
        vz = np.zeros((nnode, 1), dtype='float')
    elif var == 'rmsvel':
        v = ncfile.variables['v'][t, :, :]
        u = ncfile.variables['u'][t, :, :]
        v = np.sqrt(v * v + u * u)
        s = siglay
        vz = np.zeros((nelem, 1), dtype='float')
        bathy = convertNodal2ElemVals(nv, bathy)
    else:
        print ("dont know how to read this variable type OR it is a 2D variable !")
        sys.exit(0)
    ## interpolate variable to zlev (change coordinate so that positive is up)
    zlev = float(zlev) * 1.0
    ##    print "******", len(vz)
    for i in range(len(vz)):
        z = s * (bathy[i] + zeta[i]) + zeta[i]
        f = interp1d(z, v[:, i], bounds_error=False, fill_value=0.0)
        ###print 'z ', z
        vz[i] = f(zlev)
    return vz[:, 0], tstr


def setup_metrics(nv, ndxy):
    #
    # calculate the edges and determine neighbours of each node.
    # assumes the maximum number of neighbours are 12
    #
    nElems = np.shape(nv)[0]
    nNodes = np.shape(ndxy)[0]
    nvm1 = nv - 1
    # ~ print nElems, nNodes
    # ~ calculate the xc/yc
    xc = convertNodal2ElemVals(nv, ndxy[:, 0])
    yc = convertNodal2ElemVals(nv, ndxy[:, 0])
    # ~ determine the edges
    nEdges = nElems * 3
    edge = np.zeros((nEdges, 2), dtype='int')
    icnt = 0
    for i in range(nElems):
        edge[icnt, :] = nv[i, (0, 1)]
        edge[icnt + 1, :] = nv[i, (1, 2)]
        edge[icnt + 2, :] = nv[i, (2, 0)]
        icnt = icnt + 3
    # ~ determine nodes surrounding nodes (no specific order)
    ntsn = np.zeros((nNodes, 1), dtype='int')
    nbsn = np.zeros((nNodes, 12), dtype='int')
    # ~ print nEdges
    for i in range(nEdges):
        i1 = edge[i, 0]
        i2 = edge[i, 1]
        lmin = min(np.abs(nbsn[i1 - 1, :] - i2))
        if lmin != 0:
            ntsn[i1 - 1] = ntsn[i1 - 1] + 1
            nbsn[i1 - 1, ntsn[i1 - 1] - 1] = i2

        lmin = min(np.abs(nbsn[i2 - 1, :] - i1))
        if lmin != 0:
            ntsn[i2 - 1] = ntsn[i2 - 1] + 1
            nbsn[i2 - 1, ntsn[i2 - 1] - 1] = i1
    print('finished calculating ntsn, nbsn')
    return ntsn, nbsn


def plot_shapefile(ax, fshp, colour = '#6699cc'):
    from descartes import PolygonPatch
    import fiona

    # For each feature in the collection, add a patch to the axes.
    with fiona.open(fshp, "r") as C:
        for f in C:
            ax.add_patch(
                PolygonPatch(
                    f['geometry'], fc=colour, ec=colour, alpha=0.5))
    return ax


def its_to_tsobc(dates, case_out, fits, fobc, fbathy, fsigma):
    import scipy.interpolate as interpolate
    from netCDF4 import Dataset as ncdata

    obc_nodes = fvt.readObcFile_V3(fobc)
    with ncdata(fits, 'r') as ncfile:
        tsl, ssl, zsl = ncfile.variables['tsl'][:], ncfile.variables['ssl'][:], ncfile.variables['zsl'][:]
        print len(ncfile.dimensions['node'])
    print tsl.shape, ssl.shape, zsl.shape, len(obc_nodes)
    dep = fvt.readBathy(fbathy)
    nsiglev, siglev, nsiglay, siglay = fvt.read_sigma(fsigma)

    tsiglay = np.zeros((2, nsiglay, len(obc_nodes)), dtype=float)
    ssiglay = np.zeros((2, nsiglay, len(obc_nodes)), dtype=float)

    c = 0
    for i in obc_nodes:
        tz = tsl[0, :, i - 1]
        sz = ssl[0, :, i - 1]
        sigz = siglay * dep[i - 1]
        f = interpolate.interp1d(zsl, tz, 'linear')
        tsiglay[0, :, c] = f(sigz)
        f = interpolate.interp1d(zsl, sz, 'linear')
        ssiglay[0, :, c] = f(sigz)
        c = c + 1
    tsiglay[1, :, :] = tsiglay[0, :, :]
    ssiglay[1, :, :] = ssiglay[0, :, :]
    print 'shape', np.shape(tsiglay)
    fvt.make_tsobc_V3(case_out, fsigma, obc_nodes, dep[obc_nodes - 1], dates, tsiglay, ssiglay)
