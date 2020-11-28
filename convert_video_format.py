# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 08:57:13 2018

@author: xuwe421
"""
import ffmpy
def convertVideo():
    ff = ffmpy.FFmpeg(
    inputs={'C:\2016\59_detect_fish\3_data\Aquatera_VoithHydro\20140625_102743_194B_00408CCA70D1\Voith_20140625_102743_194B_00408CCA70D1_0-30.mkv': None},
    outputs={'C:\2016\59_detect_fish\3_data\Aquatera_VoithHydro\20140625_102743_194B_00408CCA70D1\Voith_20140625_102743_194B_00408CCA70D1_0-30.mp4': None}
    )
    ff.run()