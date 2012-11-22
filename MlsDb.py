#!/usr/bin/env python
""" Provides an interface for the MLS database to retrieve the MLS signal, also
    using the Fast Hadamard Transform, determines the system response.

The MLS interface provides methods to generate MLS signals, permutation vectors
and also provides a method to determine the system response using the Fast
Hadamard Transform.

The MLS database is a SQLite database, with the filename "mls.db".  If it does
not exisit, it will be created and populated automatically.
"""

import logging
from numpy import ones, zeros, append
import pickle
import sqlite3
import zlib

__author__ = "Lance Jenkin"
__email__ = "lancejenkin@gmail.com"

class MlsDb(object):
    _MLS_FILENAME = "./mls.db"

    def __init__(self):
        """ Constructor to create a MlsDb object."""
        self.logger = logging.getLogger("Alpha")
        self.logger.debug("Creating MlsDb Object")

        try:
            self.conn = sqlite3.connect(self._MLS_FILENAME)
        except sqlite3.OperationalError as error:
            self.logger.error("Could not open %s: %s"%(self._MLS_FILENAME, 
                                                       error))
            raise Exception("Database Error: %s"%(error))
        
        self.conn.row_factory = self._dict_factory

        self._setupDatabase()
    
    def getSystemResponse(self, response, number_taps):
        """ Determines the impulse of the response of the system, which has been
        excited by a mls signal, with the specified number of taps.
        
        It should be noted, that the impulse response is determined by
        circularly convolving the response with the orginal MLS signal.  The
        consequence of this, is that the MLS signal should be played twice in
        succession, with the first burst used to bring the system into a stable
        state, and the second burst used to determine the impulse response of
        the system.
        
        :param response:
            The response recorded from the system, needs to be of length 
            2 ^ (number_taps) - 1
        :type response:
            array of float
        :param number_taps:
            The number of taps used to generate the MLS signal.
        :type number_taps:
            int
        
        :returns:
            The system response.
        """
        self.logger.debug("Entering getSystemResponse")
        
        if len(response) < 2 ** number_taps - 1:
            self.logger.error("Reponse too short")
            return
            
        response = response[:2 ** number_taps - 1]
        cursor = self.conn.cursor()
        
        cursor.execute("""SELECT tag_r, tag_s FROM mls
                       WHERE number_taps = ?""", (number_taps, ))

        row = cursor.fetchone()
        
        tag_r = pickle.loads(zlib.decompress(row["tag_r"]))
        tag_s = pickle.loads(zlib.decompress(row["tag_s"]))
        
        permuatation = self._permutateSignal(response, tag_s)
        transformed_signal = self._fht(permuatation, number_taps)

        response = self._permutateResponse(transformed_signal, tag_r)
        
        return response
    
    def _setupDatabase(self):
        """ Ensures that the MLS table exisits, if it does not, then creates 
            the database schema.
        """
        self.logger.debug("Entering setup_database")

        # Determine if the database schema is already in place
        cursor = self.conn.cursor()
        
        cursor.execute("""SELECT COUNT(name) AS mls_count
                          FROM sqlite_master 
                          WHERE type='table' 
                          AND name='mls';""")
        row = cursor.fetchone()
        
        if row["mls_count"] == 0:
            self.logger.debug("Creating MLS table")
            cursor.execute("""CREATE TABLE "mls" 
                        ("id" INTEGER PRIMARY KEY  NOT NULL  UNIQUE, 
                         "mls" BLOB, "tag_r" BLOB, "tag_s" BLOB,
                         "number_taps" INTEGER)""")
        
            self.conn.commit()   
        
            self._rebuildDatabase()

        cursor.close()
    
    def _rebuildDatabase(self):
        """ Populates the MLS database with MLS signals from 3 taps to 18.

        Generates MLS signals for all valid tap configurations available,
        (3 to 18).  Along with the MLS, generates the tag_s and tag_r vectors
        used to permutate the signal and response for the Fast Hadamard 
        Transform.
        """
        self.logger.debug("Entering rebuild_database")
        
        cursor = self.conn.cursor()
        
        for number_taps in range(3, 18 + 1):
            self.logger.debug("Generating MLS, tag_s, tag_r with %d taps" % 
                                (number_taps) )
                                
            mls_signal = self._generateMls(number_taps)
            tag_s = self._generateTagS(mls_signal, number_taps)
            tag_r = self._generateTagR(mls_signal, number_taps)
            
            compressed_mls = buffer(zlib.compress(pickle.dumps(mls_signal), 9))
            compressed_tag_s = buffer(zlib.compress(pickle.dumps(tag_s), 9))
            compressed_tag_r = buffer(zlib.compress(pickle.dumps(tag_r),9 ))
            
            cursor.execute("""INSERT INTO mls (mls, tag_s, tag_r, number_taps) 
                        VALUES (?, ?, ?, ?)""", (compressed_mls, 
                                                compressed_tag_s,
                                                compressed_tag_r,
                                                number_taps) )
        
        self.conn.commit()   
        cursor.close()
    
    def getMls(self, number_taps):
        """ Returns the MLS signal with the specified number of taps.
        
        :param number_taps:
            The number of taps used to generate the MLS signal.
        :type number_taps:
            int
        
        :returns:
            array of int : The MLS signal using {0, 1}

        """
        self.logger.debug("Entering getMls (%s)"%(number_taps))
        
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT mls FROM mls WHERE number_taps = ?",
                        (number_taps, ) )
        
        row = cursor.fetchone()
        
        cursor.close()

        mls_signal = pickle.loads(zlib.decompress((row["mls"])))
        
        return mls_signal

    def _generateMls(self, number_taps):
        """Generate a Maximum Length Sequence with a specified number of taps.
            
        The sequence is period with period number_taps ^ 2 - 1
        
        :param number_taps:
            The number of taps to use to generate the MLS signal.
        :type number_taps:
            int
        
        :returns:
            array of int : The MLS signal of 2-nary digits, {0,1}                     
        """
        self.logger.debug("Entering _generateMls (%s)"%(number_taps))
        
        MAX_TAPS = 18 # Maximum number of taps in table
        if number_taps > MAX_TAPS:
            self.logger.info("Maximum number of taps is 18, using 18 taps")
            number_taps = 18
        if number_taps < 3:
            self.logger.info("Minimum number of taps is 3, using 3 taps")
            number_taps = 3
        
        # Due to the difficulty in calculating primitive polynomials, the
        # following taps table is taken from:
        # Impulse response measuremnts using MLS - Jens Hee,
        # url: http://jenshee.dk

    
        taps_table = [
                [0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # taps = 3
                [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # taps = 4
                [0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0], # taps = 5
                [0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0], # taps = 6
                [0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0], # taps = 7 
                [0,0,0,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0], # taps = 8
                [0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0], # taps = 9
                [0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0], # taps = 10
                [0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0], # taps = 11
                [0,0,0,0,0,1,0,1,0,0,1,1,0,0,0,0,0,0], # taps = 12
                [0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,0,0,0], # taps = 13
                [0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0], # taps = 14
                [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0], # taps = 15
                [0,0,0,1,0,0,0,0,0,0,0,0,1,0,1,1,0,0], # taps = 16
                [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0], # taps = 17
                [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1]  # taps = 18
                ]
    
        # Initialize Variables
        length = 2 ** number_taps - 1
        mls_signal = []
        delay_line = ones(number_taps)
        filter_taps = taps_table[number_taps - 3]
    
        for signal_index in range(length):
            sum = 0
            for tap_index in range(number_taps):
                sum += filter_taps[tap_index] * delay_line[tap_index]
            sum = sum % 2
    
            mls_signal = append(mls_signal, delay_line[number_taps - 1])
    
            for delay_index in range(number_taps - 1, 0, -1):
                delay_line[delay_index] = delay_line[delay_index - 1]
            delay_line[0] = sum
    
        return mls_signal
    
    def _generateTagS(self, mls_signal, number_taps):
        """  Generate the permutation vector to permetate the signal for the 
        Fast Hadamard Transform.
        
        :param mls_signal:
            The MLS signal used to generate the S vector, using the digits 
            {0, 1}.
        :type mls_signal:
            array of int
        :param number_taps:
            The number of taps used to generate the MLS signal, it is implied
            that len(mls_signal) == 2 ** number_taps - 1
        :type number_taps:
            int
        
        :returns:
            array of int: The Tag S vector
        """
        self.logger.debug("Entering _generateTagS")
        
        length = len(mls_signal)
        tag_s = zeros(len(mls_signal))
        
        for signal_index in range(len(mls_signal)):
            for i in range(number_taps):
                tag_s[signal_index] += mls_signal[(length + signal_index - i) % 
                                        length] * ( 2 ** (number_taps - 1 - i))
    
        return tag_s
        
    def _generateTagR(self, mls_signal, number_taps):
        """  Generate the permuation vector to permetate the response from the 
        Fast Hadamard Trasform.
        
        :param mls_signal:
            The MLS signal used to generate the S vector, using the digits 
            {0, 1}.
        :type mls_signal:
            array of int
        :param number_taps:
            The number of taps used to generate the MLS signal, it is implied
            that len(mls_signal) == 2 ** number_taps - 1
        :type number_taps:
            int
        
        :returns:
            array of int: The Tag R vector
        """
        self.logger.debug("Entering _generateTagR")

        length = len(mls_signal)
        col_sum = zeros(length)
        index = zeros(number_taps)
    
        for signal_index in range(length):
            for i in range(number_taps):
                col_sum[signal_index] += int(mls_signal[(length + 
                    signal_index - i) % length]) << (number_taps - 1 - i)
            
            for i in range(number_taps):
                if col_sum[signal_index] == (2 ** i):
                    index[i] = signal_index
    
        tag_r = zeros(length)
        for l_index in range(length):
            for i in range(number_taps):
                tag_r[l_index] += mls_signal[(length + index[i] - l_index) % 
                                            length] * ( 1 << i)
        return tag_r
    
    def _permutateSignal(self, signal, tag_s):
        """ Permutate the signal according to the indices in tag_s.
            
            :param signal:
                The response to the system that was excited by the MLS signal
                used to generate tag_s.  Must be the same length as tag_s.
            :type signal:
                array of float
            :param tag_s:
                The S vector used to permuate the signal to prepare for the 
                Fast Hadamard Transform.
            :type tag_s:
                array of int
            
            :returns:
                The signal vector, permutated so that it may be transformed 
                using the Fast Hadamard Transform.
        """
        self.logger.debug("Entering _permutateSignal")
        
        dc = 0
        for sample in signal:
            dc += sample
        permutation = zeros(len(signal) + 1)
        permutation[0] = -dc

        for sample_index in range(len(signal)):
            permutation[tag_s[sample_index]] = signal[sample_index]
    
        return permutation
    
    def _permutateResponse(self, permutation, tag_r):
        """ Permutate the permutated response according to the incides in tag_r.
            
            :param permuation:
                The resulting vector from the Fast Hadamard Transform, it is 
                a permutated vection of the actual system response.
            :type permuation:
                array of float
            :param tag_r:
                The R vector used to permutate the transformed vector into the
                system response.

            :returns:
                The permutated response from the FHT, which is the system 
                response.
        """
        self.logger.debug("Entering _permutateResponse")
    
        scale_factor = 1 / float(len(permutation))
        
        response = []
        for i in range(len(permutation) - 1):
            response = append(response, permutation[tag_r[i]] * scale_factor)
        
        response = append(response, 0)

        return response
        
    def _fht(self, x, number_taps):
        """ Preform the Fast Hadamard Transform on the permutated vector x.
            
            @param x: The vector to preform the Fast Hadamard Transform on
            @param number_taps: The number of taps used to generate the orginal
                                MLS signal.
                                
            @return: The transformed vector
            
        """
    
        k1 = len(x)
        for k in range(number_taps):
            k2 = k1 / 2
            for j in range(k2):
                for i in range(j, len(x), k1):
                    i1 = i + k2
                    temp = x[i] + x[i1]
                    x[i1] = x[i] - x[i1]
                    x[i]=temp
            k1 /= 2
    
        return x

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
    
    mls_db = MlsDb()

    mls_signal = mls_db.getMls(4)
    mls_signal = -2 * mls_signal + 1

    print mls_signal

    response = mls_db.getSystemResponse(mls_signal, 4)
    print response
