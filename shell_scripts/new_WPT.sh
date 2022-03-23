#!/bin/bash
echo 'LOGIN TO WGMS'
/usr/bin/curl -k -H "Content-Type: text/xml; charset=utf-8" --dump-header headers -H "SOAPAction:" -d @login.xml -X POST https://gliders.wgms.com/webservices/entityapi.asmx

echo ''
echo 'ISSUING A NEW WAYPOINT'
/usr/bin/curl -X POST  -L -b headers -d @wgms_new_WPT.xml http://gliders.wgms.com/webservices/entityapi.asmx --header "Content-Type:text/xml"
echo ''
