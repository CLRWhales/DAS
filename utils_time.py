#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 16:38:18 2022

@author: keving
"""
import time
import pytz; from datetime import datetime, timedelta, timezone

FORMAT_TIME="%H:%M:%S"
FORMAT_DATE_TIME="%Y-%m-%dT%H:%M:%S"
FORMAT_TIMESTR = "%H%M%S"
FORMAT_DATE_INT = "%Y%m%d"

format_time='%d.%m.%YT%H:%M:%S.%f'
tz_local = pytz.timezone('Europe/Amsterdam')  
tz_utc = pytz.timezone('UTC')


def assign_utc(dto):
    dto = tz_utc.localize(dto)
    return dto

def assign_tz_local(dto):
    dto = tz_local.localize(dto)
    return dto

def dto_loc2utc(dto, local=True):
    """
    converts datetime object from the local timezone to utc-timezone

    Parameters
    ----------
    dto : datetime-object (DTO)
        DTO containing the local time.
    local : bool, optional
        True if local time zone is already assigned to dto. The default is True.
        Assignes local TZ to DTO first before converting if False. 
    Returns
    -------
    dto_utc : DTO
        DTO with UTC TZ assigned.

    """
    if local==False: 
        dto = tz_local.localize(dto)
    dto_utc = dto.astimezone(tz_utc)
    
    return dto_utc

#dto_utc=dto.replace(tzinfo=tim

def time_loc2utc(dto, tz_local='Europe/Amsterdam'):
    
    mytz = pytz.timezone(tz_local)  
    tz_utc = pytz.timezone('UTC')
        
    dto_loc = mytz.localize(dto)
    dto_utc = dto_loc.astimezone(tz_utc)
    
    return dto_utc

def timestr2ints(time_str):
    "convert timestring of format HHMMSS to separated ints"
    hh, mm, ss = [int(time_str[0+i*2:i*2+2]) for i in range(3)]
    return (hh,mm,ss)

def date2ints(date):
    "convert date strings of YYYYMMDD format to separated integers"
    if type(date) != str: 
        date=str(date)
    if len(date) != 8: 
        raise ValueError('!! wrong string length for date2ints conversion, use YYYYMMDD format')
    else: 
        year=int(date[0:4]); month=int(date[4:6]); day=int(date[6:None])
    return (year, month, day)

def time2ints(timex):
    """
    converts time strings of HH:MM:SS format into integers
    """
    nums = timex.split(':')
    for num in nums: 
        if len(str(num))!=2: 
            raise ValueError('!! wrong string length for date2ints converersion, use HH:MM:SS format')
    if len(nums)==2: 
        nums.append('00')
    hour, mins, secs = [int(num) for num in nums]
    
    return hour, mins, secs

def time2timestr(timex):
    hour, mins, secs = time2ints(timex)
    return '{:02d}{:02d}{:02d}'.format(hour,mins,secs)

def timestr2time(time_str):
    hour, mins, secs  = timestr2ints(time_str)
    return '{:02d}:{:02d}:{:02d}'.format(hour,mins,secs)
    

def date2dto(date,timex='08:00:00', tz='utc'):
    
    year, month, day = date2ints(date)
    if time: 
         hour, mins, secs = time2ints(timex)
    else: 
        hour=0; mins=0; secs=0; 
        
    dto = datetime(year=year, month=month, day=day, hour=hour, minute=mins, second=secs)
    if tz=='utc':
        dto = assign_utc(dto)
        
    return dto


def dto_round_to_seconds(dto):
    "round microseconds of datetime object to seconds"
    if dto.microsecond >= 500_000:
        dto += timedelta(seconds=1)
    return dto.replace(microsecond=0)

def timestr_now():
    timestamp = time.time()
    dto = datetime.utcfromtimestamp(timestamp)
    return dto.strftime(FORMAT_TIMESTR)

def date_now(return_int=True):
    timestamp = time.time()
    dto = datetime.utcfromtimestamp(timestamp)
    date = dto.strftime("%Y%m%d")
    if return_int:
        date=int(date)    
    return date
