
# def sftp_put_cordcftp(LOCALpath, REMOTEpath):
#     '''
#     This function copies a local file (LOCALpath) to the CORDCdev server as REMOTEpath via sftp.
#     :param LOCALpath (str): absolute path to local file
#     :param REMOTEpath (str): absolute path to remote file
#     :return: SFTPAttributes object (http://docs.paramiko.org/en/stable/api/sftp.html#paramiko.sftp_attr.SFTPAttributes)
#     containing attributes about the given file
#     '''
#     import paramiko
#     # Connect to CORDCdev
#     client = paramiko.SSHClient()
#     client.load_system_host_keys()
#     client.connect(hostname='cordcftp.ucsd.edu',
#                    username= 'anonymous',
#                    password= 'norse@')
#
#     # Copy a local file (LOCALpath) to the CORDCdev server (REMOTEpath) via sftp
#     sftp_client = client.open_sftp()
#     SFTPAttributes = sftp_client.put(LOCALpath,REMOTEpath)
#
#     # close connection
#     client.close()
#     return SFTPAttributes