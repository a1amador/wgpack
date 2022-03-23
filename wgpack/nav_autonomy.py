import os,sys,subprocess
from pathlib import Path

# define parent and module path
parent_path = os.path.abspath(os.path.join('..'))
module_path = os.path.join(os.path.abspath(os.path.join('..')),'wgpack')
print(parent_path)
print(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)
from wgpack.creds import WGMSun,WGMSpw

# define xml and shell script folders
xml_folder  = os.path.join(module_path,'xmls')
sh_folder   = os.path.join(module_path,'shell_scripts')

# create dictionary for for vehicle id's (WGMS)
veh_list = {
    "sv3-125" : '2598',
    "sv3-251" : '4980',
    "sv3-253" : '5036',
    "magnus"  : '5550'
        }

# Login xml string
loginXMLstring = '''<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope 
xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
xmlns:xsd="http://www.w3.org/2001/XMLSchema" 
xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
  <soap:Body>
    <CreateLoginSession xmlns="http://gliders.liquidr.com/webservicesWebServices">
      <login>WGMSun</login>
      <password>WGMSpw</password>
      <orgName>cordc</orgName>
    </CreateLoginSession>
  </soap:Body>
</soap:Envelope>
'''
# Edit login xml
loginXMLstring = loginXMLstring.replace('WGMSun', WGMSun)
loginXMLstring = loginXMLstring.replace('WGMSpw', WGMSpw)


def sendWPT(vnam,WaypointName,WaypointNumber,lat,lon):
    '''
    This function issues a series of commands to create a new waypoint via WGMS web services.
    This method can be used to issue any command defined by the XML entity
    References:
    http://cordc.wgms.com/help/admin-guide/apbs06.html
    :param vnam (str): vehicle name (e.g., 'sv3-253', 'magnus')
    :param WaypointName (str): waypoint name (e.g., 'Waypoint66')
    :param WaypointNumber (str): waypoint number (e.g., '66')
    :param lat (str): latitude (e.g., '32.879')
    :param lon (str): longitude (e.g., '-117.345')
    :return:
    '''

    # Develop a createentity.xml file with the New Waypoint command SOAP 1.1 or SOAP 1.2 XML envelope.
    # You will need to know the VehicleId, and the communications channel being used.
    XMLstring = '''<?xml version="1.0" encoding="utf-8"?>
    <soap:Envelope 
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
    xmlns:xsd="http://www.w3.org/2001/XMLSchema" 
    xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
     <soap:Body>
        <CreateEntity xmlns="http://gliders.liquidr.com/webservicesWebServices"><entityType>53</entityType>
         <entityFieldXml>
       %3Centity%3E%3CWaypointName%3E
    _WaypointName_
       %3C%2FWaypointName%3E%3CWaypointNumber
       %3E
    _WaypointNumber_
       %3C%2FWaypointNumber%3E%3CVehicleId%3E
    _vid_
       %3C%2FVehicleId%3E%3CCommunicationChannelSelect
       %3E
    6
       %3C%2FCommunicationChannelSelect%3E%3CVisible%3E1%3C%2FVisible%3E%3CWaypointStatus%3E1%3C%2FWaypointStatus%3E
       %3CLattitude%3E
    _lat_
       %3C%2FLattitude%3E%3CLongitude%3E
    _lon_
       %3C%2FLongitude%3E%3C%2Fentity%3E
         </entityFieldXml>
        </CreateEntity>
      </soap:Body>
    </soap:Envelope>
    '''

    # Edit command xml
    XMLstring = XMLstring.replace('_WaypointName_',WaypointName)
    XMLstring = XMLstring.replace('_WaypointNumber_',WaypointNumber)
    XMLstring = XMLstring.replace('_vid_',veh_list[vnam])
    XMLstring = XMLstring.replace('_lat_',lat)
    XMLstring = XMLstring.replace('_lon_',lon)

    # Save login xml file
    text_file = open(os.path.join(xml_folder,'login.xml'), "w")
    text_file.write(loginXMLstring)
    text_file.close()

    # Save command xml file
    text_file = open(os.path.join(xml_folder,'wgms_new_WPT.xml'), "w")
    text_file.write(XMLstring)
    text_file.close()

    # Execute shell script that runs curl command
    subprocess.call(['sh', os.path.join(sh_folder,'new_WPT.sh')])
