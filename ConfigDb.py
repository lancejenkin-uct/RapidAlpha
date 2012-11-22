#!/usr/bin/env python
""" Provides an interface to retrieve and save configuration settings.  

The configuration database contains sane default measurement and analysis 
settings.  It also contains other program information
"""

import logging
import pickle
import sqlite3
import os
import sys
import zlib

__author__ = "Lance Jenkin"
__email__ = "lancejenkin@gmail.com"

class ConfigDb(object):
    if getattr(sys, 'frozen', None):
        basedir = sys._MEIPASS
    else:
        basedir = os.path.dirname(__file__)
    _CONFIG_FILENAME = os.path.join(basedir, "config.db")

    def __init__(self):
        """ Constructor for ConfigDb object """

        self.logger = logging.getLogger("Alpha")
        self.logger.debug("Creating ConfigDb Object")

        try:
            self.conn = sqlite3.connect(self._CONFIG_FILENAME)
        except sqlite3.OperationalError as error:
            self.logger.error("Could not open %s: %s"%(self._CONFIG_FILENAME, 
                                                       error))
            raise Exception("Database Error: %s"%(error))
        
        self.conn.row_factory = self._dict_factory

        self._setupDatabase()
    
    def _setupDatabase(self):
        """ If no database exisits in the filename, create a default schema.

        The default schema is 3 tables, config, signal, and analysis.  They are
        all key / value tables, with they key the primary index.  The config 
        table contains application configuration, like the audio device to use.
        The signal table contains default signal parameters to use.  And the
        analysis table contains default analysis paramaters to use.
        """
        self.logger.debug("Entering _setupDatabase")

        # Determine if the database schema is already in place
        cursor = self.conn.cursor()
        
        cursor.execute("""SELECT COUNT(name) AS table_count
                          FROM sqlite_master 
                          WHERE type='table' 
                          AND name='config';""")
        row = cursor.fetchone()
        
        if row["table_count"] == 0:
            self.logger.info("Setting up configuration database")
            # Create the three required tables
            for table_name in ("config", "signal", "analysis"):
                cursor.execute("""CREATE TABLE '%s' 
                                    ('key' TEXT PRIMARY KEY,
                                    'value' TEXT)""" % (table_name))
            self.conn.commit()   
        
            self._setDefaults()

        cursor.close()
    
    def _setDefaults(self):
        """ Set default values to use for application, signal generation and
            measurement analysis.
        """
        self.logger.debug("Entering _setDefaults")

        config_defaults = {
            "input device": 0,
            "output device": 0,
            "sample rate": 44100,
            "fft size": 2 ** 18,
            "noise samples": 1000,
            "impulse constant": 15,
            "impulse threshold": 0.02,
            "gain": 0.5
        }

        self.saveSettings("config", config_defaults)

        signal_defaults = {
            "signal type" : "maximum length sequence",
            "mls reps" : 1,
            "mls taps" : 14,

            "lpf cutoff" : 3500,
            "lpf enabled" : 1,
            "lpf order" : 4,
            "hpf cutoff" : 500,
            "hpf enabled" : 1,
            "hpf order" : 1,

            "pad signal" : 1,
            "signal padding" : 200 * 10 ** -3,
            "impulse delay" : 20 * 10 ** -3,

            "signal reps" : 10,

            "signal length" : 100 * 10 ** -3,
            "lower frequency" : 0,
            "upper frequency" : 6400
        }
        self.saveSettings("signal", signal_defaults)
        
        analysis_defaults = {
            "window type" : "two sided",
            "window start" : 2.8 * 10 ** -3,
            "window end" : 10 * 10 ** -3,
            "taper length" : 0.6 * 10 ** -3,

            "decimation factor" : 5,
            "antialiasing filter order": 3,
        }
        self.saveSettings("analysis", analysis_defaults)
    
    def saveSettings(self, table, settings):
        """ Saves settings into the specified table.

            :param table:
                The name of the table to save the settings, one of "config", 
                "signal", or "analysis".
            :type table:
                str
            :param settings:
                A dictionary containing the settings to save.
            :type settings:
                dict
        """
        self.logger.debug("Entering saveSettings")

        cursor = self.conn.cursor()

        for key, value in settings.items():
            cursor.execute("REPLACE INTO %s (key, value) VALUES (?, ?)" % table,
                (key, value))
        
        self.conn.commit()
    
    def getSettings(self, table):
        """ Returns all the settings in the requested table.

            :param table:
                The name of the table to get the settings from.
            :type table:
                str

            :returns:
                dict - A dictionary containing all the settings in the table
        """
        self.logger.debug("Entering getSettings")

        cursor = self.conn.cursor()

        cursor.execute("SELECT * FROM %s" % (table))

        results = cursor.fetchall()

        settings = {}
        for row in results:
            settings[row["key"]] = row["value"]
        
        return settings

    @staticmethod
    def _dict_factory(cursor, row):
        """ Dictionary factory used for sqlite3 to return rows as dictionaries,
            with column names as keys.
        """
        
        d = {}
        for idx,col in enumerate(cursor.description):
            d[col[0]] = row[idx]
        return d

if __name__=="__main__":
    logger = logging.getLogger("Alpha")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    
    config_db = ConfigDb()

    print config_db.getSettings("config")
    print config_db.getSettings("signal")
    print config_db.getSettings("analysis")
   