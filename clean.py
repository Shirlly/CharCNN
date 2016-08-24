# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 11:41:06 2016

@author: I310627
"""


# -- Load library --
import numpy as np
import re


def replacePun(__msg):
    __msg = __msg.decode('iso-8859-1').encode('utf-8').strip()
    __msg = __msg.lower()
    __msg = re.sub(r"\*", ' ', __msg)
    __msg = re.sub(r"\(", ' ', __msg)
    __msg = re.sub(r"\)", ' ', __msg)
    __msg = re.sub(r"<", ' ', __msg)
    __msg = re.sub(r">", ' ', __msg)
    __msg = re.sub(r"!", ' ', __msg)
    __msg = re.sub(r"#", ' ', __msg)
    __msg = re.sub(r"$", ' ', __msg)
    __msg = re.sub(r"_{2,}", ' ', __msg)
    __msg = re.sub(r"={2,}", ' ', __msg)
    __msg = re.sub(r"-{2,}", ' ', __msg)
    __msg = re.sub(r"~{2,}", ' ', __msg)
    __msg = re.sub(r"\.{2,}", ' ', __msg)
    __msg = re.sub(r"\?{2,}", ' ', __msg)
    __msg = re.sub(r"\{2,}", ' ', __msg)
    return __msg



# -- Clean Data --
def cleanData(__msg):
    urlRegex1 = "^(http|https|ftp|file)://.*"
    urlRegex2 = ".*[/]*.*(.html)"
    fileRegex = "[.|\w]*/.*[/]*.*"
    pathRegex = ".*[:\\\\]+.*[>]*.*"
    ipRegex = "\\W[0-9]+[.]+[0-9]+[.]+[0-9]+[.]+[0-9]+\\W"
    hourRegex = "[0-9]+[:]+[0-9]+[:]+[0-9]+.*"
    timeRegex = ".*[0-9]+(/|-|.)+[0-9]+(/|-|.)+[0-9]+"
    telRegex = "[+]+[0-9]+-[0-9]+"
    NARegex = "[a-zA-Z0-9]+"
    alphaRegex = "[a-zA-Z]+"
    numRegex = "[0-9]+"
    __sen = []
    #print __msg
    for w in __msg.split():
        if w.endswith((',','.',';',':','?','!',')','>','\'','"','&')):
            w = w[:-1]

        if w.startswith((',','.',';',':','?','!','(','<','\'','"','&')):
            w = w[1:]

        if re.match(urlRegex1, w, re.M|re.I) or re.match(urlRegex2, w, re.M|re.I):
            w = 'url'
        elif re.match(pathRegex, w, re.M|re.I):
            w = 'pathRegex'
        elif re.match(fileRegex, w, re.M|re.I):
            w = 'fileRegex'
        elif re.match(ipRegex, w, re.M|re.I):
            w = 'ipAddress'
        elif re.match(timeRegex, w, re.M|re.I):
            w = "TTime"
        elif re.match(hourRegex, w, re.M|re.I):
            w = "THour"
        elif re.match(numRegex, w, re.M|re.I):
            w = "allNum"
        elif re.match(telRegex, w, re.M|re.I):
            w = "TelNum"
        elif re.match("\\W", w, re.M|re.I):
            w = ""
        elif re.match(NARegex, w, re.M|re.I) and not re.match(numRegex, w, re.M|re.I) and not re.match(alphaRegex,w, re.M|re.I) and len(list(w))>10:
            w = "NA"
        elif re.match(alphaRegex, w, re.M|re.I) and len(list(w))>15:
            w = "LW"

        if w.endswith((',','.',';',':','?','!',')','>','\'','"','&')):
            w = w[:-1]
        if w.startswith((',','.',';',':','?','!','(','<','\'','"','&')):
            w = w[1:]
        if w.strip():
            __sen.append(w)
    text = " ".join(__sen)
    #print text
    return text


def processText(text_iterator):
    rs = []
    for text in text_iterator:
        __msg = replacePun(text)
        __clean = cleanData(__msg)
        rs.append(__clean)
    return rs
    
def preprocessText(text):
    __texts = []
    for line in text:
        #print '....original'
        __msg = replacePun(line)
        #print '....pun'
        __sen = cleanData(__msg)
        #print '....clean'
        #print '----------------'
        __texts.append(__sen)
    return __texts


# -- PreProcess Sample Data --
def preprocessData(sample_data):
    __texts = []
    #__txt = []
    with open(sample_data, "r") as f:
        for line in f:
            #sample = line.split("&&")
            #if len(sample) == 2:
            print '....original'
            #print sample[0]
            __msg = replacePun(line)
            print '....pun'
            #print __msg
            __sen = cleanData(__msg)
            print '....clean'
            #print __sen
            print '----------------'
            #con = __sen + "\t&&\t" +sample[1].strip()
            __texts.append(__sen)
            #else:
            #for learning only
            #    print("***************error***************\n")
    f.close()
    return __texts


# -- Main --
if __name__ == "__main__":

    # -- Sample Processing --
    sourcedata = "route_1_10.txt"
    _texts = preprocessData(sourcedata)

    # -- Write Sample Data --
    out = open("route_1_10_clean.txt", "wb")
    # for i in range(len(_texts)):
    #     print _texts[i]
    #     out.write(_texts[i] + "\n")
    out.close()
