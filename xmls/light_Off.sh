#!/bin/bash
echo 'LOGIN TO WGMS'
/usr/bin/curl -k -H "Content-Type: text/xml; charset=utf-8" --dump-header headers -H "SOAPAction:" -d @login.xml -X POST https://gliders.wgms.com/webservices/entityapi.asmx


echo ''
echo 'TURN LIGHT OFF'
# printf 'TURN LIGHT OFF'
/usr/bin/curl -X POST  -L -b headers -d @wgms_light_Off.xml http://gliders.wgms.com/webservices/entityapi.asmx --header "Content-Type:text/xml"
echo ''
