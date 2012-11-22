# RapidAlpha
RapidAlpha is an application developed for Lance Jenkin's MSc Thesis, *Application of Cepstral Techniques to the Automated Determination of the Sound Power Absorption Coefficient*.  It is used to measure the sound power absorption coefficient of material samples.

The application can be started from the command line as follows:
> python RapidDelgate.py

## Software Architecture
*RapidDelegate* is a subclass of *BaseDelegate*.  It is responsible creating a new *RapidController* - the user interface.  It creates a *PreferenceDelegate* - to set the settings for the program.  It also handles the new measurement, save measurement and load measurement signals.  

A new measurement is handled by *BaseDelegate*.  It creates a new *Measurement* object with using the configuration in the **measurement_settings** property.  The *Measurement* object create a new *AudioIO* object using the audio devices in **measurement_settings**.  The *AudioIO* object handles the playback and recording of the audio signals.  The **newMeasurement** method of the *Measurement* object creates a new *SignalGenerator* object.  This object generates the excitation signal to be used in the measurement, including repeating the signal, and padding the signal with silence.  The *AudioIO* objects plays the signal and captures the response.  The *Measurement* object then locates the synchronization impulse.  It then splits the one long recorded response into the individual signal responses.

Once the measurement has completed, it would be saved by calling the **saveMeasurement** method of the *BaseDelegate* object.  A new *MeasurementDB* object is created, and the measurement settings and signals are then saved to a measurement database.

**loadAbsorptionCoefficeint** method of the *BaseDelegate* object loads the measurement database.  The microphone and generator signals and the measurement settings are used to create a new *AbsorptionCoefficient*.  The **determineAlpha** method the *AbsorptionCoefficient* object then extracts the signals from the raw responses.  It then averages the responses together.  From the averaged signals, the frequency response of the system is determined.  The responses are then down sampled.
The resampled signals then used to determine the power cepstra of the microphone and generator.  The impulse response of the material sample is then **liftered** from the system's power cepstrum, and transformed to the Frequency domain.  The absorption coefficient is then determined.  Once the absorption coefficient is determined, the *RapidController* graphs the absorption coefficient using the *Grapher* object.

All the database files are standard *SQLite* databases.  The *config.db* is the main configuration database, where default settings can be set.  Although there are 3 tables in the *config.db* database, when a measurement is saved, the three tables are merged into an *attributes* table in the measurement database.
