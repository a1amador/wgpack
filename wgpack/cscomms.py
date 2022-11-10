# Client-server communications module
import os
import paramiko
import datetime
import numpy as np
import pandas as pd
from .creds import SCun,SCpw,CDun,CDpw

def sftp_mirror_seachest(REMOTEdir, LOCALdir):
    '''
    This function populates a local directory with all the files in a given SeaChest directory
    references: http://docs.paramiko.org/en/stable/
                https://www.youtube.com/watch?v=dtvV2xKaVjw
    :param REMOTEdir (str): SeaChest (remote) directory
    :param LOCALdir (str):  Local directory to populate
    :return:                string list with updated file names
    '''

    # List local file names
    LOCALfnames = []
    for root, dirs, files in os.walk(LOCALdir):
        for filename in files:
            LOCALfnames.append(filename)

    # Connect to SeaChest
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect(hostname='seachest.ucsd.edu',
                   username=SCun,
                   password=SCpw)

    # Get file(s) from SeaChest and download to local server via sftp
    sftp_client = client.open_sftp()
    # print(dir(sftp_client))

    # Get file names
    newREMOTEdir = REMOTEdir.replace(' ', '\ ')
    stdin, stdout, stderr = client.exec_command('ls ' + newREMOTEdir)
    lines = stdout.readlines()

    # Transfer most recent file file to local directory
    latest = 0
    fname = None
    for fileattr in sftp_client.listdir_attr(REMOTEdir):
        if np.logical_or(fileattr.filename.startswith('metbuoy'),
                         fileattr.filename.startswith('mwb')) and fileattr.st_mtime > latest:
            latest = fileattr.st_mtime
            fname = fileattr.filename
    if fname is not None:
        sftp_client.get(REMOTEdir + fname, LOCALdir + fname)
        print('updating latest file: ' + fname)
    else:
        print('Latest file was not updated')

    # Transfer remaining files to local directory
    updated = [fname]
    for line in lines:
        fname = line[0:-1]
        if fname not in LOCALfnames:
            print('also updating: ' + fname)
            sftp_client.get(REMOTEdir + fname, LOCALdir + fname)
            updated.append(fname)

    # close connection
    client.close()
    print('done syncing local directory with SeaChest')
    return list(dict.fromkeys(updated))

def sftp_mirror_seachest_adcp(REMOTEdir, LOCALdir, tst, ten):
    '''
    This function populates a local directory with all the files in a given SeaChest directory
    references: http://docs.paramiko.org/en/stable/
                https://www.youtube.com/watch?v=dtvV2xKaVjw
    :param REMOTEdir (str): SeaChest (remote) directory
    :param LOCALdir (str):  Local directory to populate
    :param tst (timestamp): Start date
    :param ten (timestamp): End date
    :return:                string list with file names and updated file names
    '''

    # List local file names
    LOCALfnames = []
    for root, dirs, files in os.walk(LOCALdir):
        for filename in files:
            LOCALfnames.append(filename)

    # Connect to SeaChest
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect(hostname='seachest.ucsd.edu',
                   username=SCun,
                   password=SCpw)

    # Get file(s) from SeaChest and download to local server via sftp
    sftp_client = client.open_sftp()
    # print(dir(sftp_client))

    # Get file names
    newREMOTEdir = REMOTEdir.replace(' ', '\ ')
    stdin, stdout, stderr = client.exec_command('ls ' + newREMOTEdir)
    lines = stdout.readlines()

    # ----------------------------------------------------
    # Get dates based on adcp filename
    remote_files = [x.filename for x in sorted(sftp_client.listdir_attr(REMOTEdir), key = lambda f: f.st_mtime)]
    adcpfile_dates=[]
    for rfiles in remote_files:
        date_time_str = rfiles[:15]
        adcpfile_dates.append(datetime.datetime.strptime(date_time_str, '%Y%m%d_%H%M%S'))
    adcpfile_dates = pd.to_datetime(adcpfile_dates)
    # ----------------------------------------------------
    # Find ADCP data according to the time window to be considered
    ii = np.where(np.logical_and(adcpfile_dates>tst,adcpfile_dates<ten))[0]
    adcp_fnames = remote_files[ii[0]:ii[-1]]
    # ----------------------------------------------------
    # Download corresponding ADCP data
    files,updated=[],[]
    for fname in adcp_fnames:
        files.append(fname)
        if not os.path.exists(os.path.join(LOCALdir,fname)):
            sftp_client.get(REMOTEdir + fname, LOCALdir + fname)
            updated.append(fname)
            print('updating latest file: ' + fname)
        else:
            print('file: ' + fname + ' already exists in local directory')

    # ----------------------------------------------------
    # close connection
    client.close()
    print('done syncing local directory with SeaChest')
    return files,updated

def sftp_put_cordcdev(LOCALpath, REMOTEpath):
    '''
    This function copies a local file (LOCALpath) to the CORDCdev server as REMOTEpath via sftp.
    :param LOCALpath (str): absolute path to local file
    :param REMOTEpath (str): absolute path to remote file
    :return: SFTPAttributes object (http://docs.paramiko.org/en/stable/api/sftp.html#paramiko.sftp_attr.SFTPAttributes)
    containing attributes about the given file
    '''
    # Connect to CORDCdev
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect(hostname='cordcdev.ucsd.edu',
                   username= CDun,
                   password= CDpw)

    # Copy a local file (LOCALpath) to the CORDCdev server (REMOTEpath) via sftp
    sftp_client = client.open_sftp()
    SFTPAttributes = sftp_client.put(LOCALpath,REMOTEpath)

    # close connection
    client.close()
    return SFTPAttributes

def sftp_remove_cordcdev(REMOTEpath):
    '''
    This function removes a remote file given by REMOTEpath from the CORDCdev server via sftp.
    :param LOCALpath (str): absolute path to local file
    :param REMOTEpath (str): absolute path to remote file
    :return: SFTPAttributes object (http://docs.paramiko.org/en/stable/api/sftp.html#paramiko.sftp_attr.SFTPAttributes)
    containing attributes about the given file
    '''
    # Connect to CORDCdev
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect(hostname='cordcdev.ucsd.edu',
                   username= CDun,
                   password= CDpw)

    # Copy a local file (LOCALpath) to the CORDCdev server (REMOTEpath) via sftp
    sftp_client = client.open_sftp()
    SFTPAttributes = sftp_client.remove(REMOTEpath)

    # close connection
    client.close()
    return SFTPAttributes
