clc
clear
close all
wm=webmap('World Imagery');
zoomLevel=3;
wmcenter(wm,28,124);
wmzoom(wm,zoomLevel);
lat =  [42.299827,50];
lon = [-71.350273,120];
wmmarker(wm,lat,lon);
