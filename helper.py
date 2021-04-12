def ms_hm(value):
    """
    This function will convert from 10-milliseconds(centi second) format to
    hours and minutes.
    """
    no_of_ten_msecs = value
    no_of_msecs = 10 * value
    no_of_secs = int(no_of_msecs/1000)
    no_of_hours = int(no_of_secs/3600)
    no_of_mins = int((no_of_secs % 3600)/60)
    hm ={'hours' : no_of_hours,
         'minutes' : no_of_mins}
    return hm


