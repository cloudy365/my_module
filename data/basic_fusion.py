
# -*- coding: utf-8 -*-

from .. import h5py

def bf_info(file_path, instrument):
    """
    Input: basic fusion file path, instrument
    Output: data structure
    """
    h5f = h5py.File(file_path, 'r')
    a = h5f[instrument]
    
    for _, i in a.iteritems():
        lev1 = i.name
        print ""
        print lev1

        for _, j in h5f[lev1].iteritems():
            lev2 = j.name
            lev2_main = h5f[lev2]

            if isinstance(lev2_main, h5py.Dataset):
                print "  ", lev2, j.maxshape
            else:
                print "  ", lev2

                for _, k in lev2_main.iteritems():
                    lev3 = k.name
                    lev3_main = h5f[lev3]
                    if isinstance(lev3_main, h5py.Dataset):
                        print "    ", lev3, k.maxshape
                    else:        
                        print "    ", lev3

                        for _, l in lev3_main.iteritems():
                            lev4 = l.name
                            lev4_main = h5f[lev4]
                            if isinstance(lev4_main, h5py.Dataset):
                                print "      ", lev4, l.maxshape
                            else:
                                print "      ", lev4


def git_sublime_test():
    print "git_sublime_test runs successfully."