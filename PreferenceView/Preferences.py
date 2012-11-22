# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file './/PreferenceView/Preferences.ui'
#
# Created: Fri Sep 14 09:31:14 2012
#      by: PyQt4 UI code generator 4.9
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_Preference(object):
    def setupUi(self, Preference):
        Preference.setObjectName(_fromUtf8("Preference"))
        Preference.setWindowModality(QtCore.Qt.WindowModal)
        Preference.resize(595, 434)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8(":/Icons/alpha.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Preference.setWindowIcon(icon)
        self.verticalLayout = QtGui.QVBoxLayout(Preference)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.preferenceTab = QtGui.QTabWidget(Preference)
        self.preferenceTab.setObjectName(_fromUtf8("preferenceTab"))
        self.tab = QtGui.QWidget()
        self.tab.setObjectName(_fromUtf8("tab"))
        self.verticalLayout_7 = QtGui.QVBoxLayout(self.tab)
        self.verticalLayout_7.setObjectName(_fromUtf8("verticalLayout_7"))
        self.horizontalLayout_26 = QtGui.QHBoxLayout()
        self.horizontalLayout_26.setObjectName(_fromUtf8("horizontalLayout_26"))
        self.label_26 = QtGui.QLabel(self.tab)
        self.label_26.setMinimumSize(QtCore.QSize(121, 16))
        self.label_26.setMaximumSize(QtCore.QSize(121, 16))
        self.label_26.setObjectName(_fromUtf8("label_26"))
        self.horizontalLayout_26.addWidget(self.label_26)
        self.inputDeviceBox = QtGui.QComboBox(self.tab)
        self.inputDeviceBox.setObjectName(_fromUtf8("inputDeviceBox"))
        self.horizontalLayout_26.addWidget(self.inputDeviceBox)
        self.verticalLayout_7.addLayout(self.horizontalLayout_26)
        self.horizontalLayout_27 = QtGui.QHBoxLayout()
        self.horizontalLayout_27.setObjectName(_fromUtf8("horizontalLayout_27"))
        self.label_27 = QtGui.QLabel(self.tab)
        self.label_27.setMinimumSize(QtCore.QSize(121, 16))
        self.label_27.setMaximumSize(QtCore.QSize(121, 16))
        self.label_27.setObjectName(_fromUtf8("label_27"))
        self.horizontalLayout_27.addWidget(self.label_27)
        self.outputDeviceBox = QtGui.QComboBox(self.tab)
        self.outputDeviceBox.setObjectName(_fromUtf8("outputDeviceBox"))
        self.horizontalLayout_27.addWidget(self.outputDeviceBox)
        self.verticalLayout_7.addLayout(self.horizontalLayout_27)
        self.horizontalLayout_28 = QtGui.QHBoxLayout()
        self.horizontalLayout_28.setObjectName(_fromUtf8("horizontalLayout_28"))
        self.label_28 = QtGui.QLabel(self.tab)
        self.label_28.setMinimumSize(QtCore.QSize(121, 16))
        self.label_28.setMaximumSize(QtCore.QSize(121, 16))
        self.label_28.setObjectName(_fromUtf8("label_28"))
        self.horizontalLayout_28.addWidget(self.label_28)
        self.horizontalLayout_25 = QtGui.QHBoxLayout()
        self.horizontalLayout_25.setObjectName(_fromUtf8("horizontalLayout_25"))
        self.gainSlider = QtGui.QSlider(self.tab)
        self.gainSlider.setMinimum(-1000)
        self.gainSlider.setMaximum(0)
        self.gainSlider.setSingleStep(1)
        self.gainSlider.setOrientation(QtCore.Qt.Horizontal)
        self.gainSlider.setTickPosition(QtGui.QSlider.NoTicks)
        self.gainSlider.setTickInterval(5)
        self.gainSlider.setObjectName(_fromUtf8("gainSlider"))
        self.horizontalLayout_25.addWidget(self.gainSlider)
        self.gainLabel = QtGui.QLabel(self.tab)
        self.gainLabel.setMinimumSize(QtCore.QSize(70, 0))
        self.gainLabel.setMaximumSize(QtCore.QSize(70, 16))
        self.gainLabel.setObjectName(_fromUtf8("gainLabel"))
        self.horizontalLayout_25.addWidget(self.gainLabel)
        self.horizontalLayout_28.addLayout(self.horizontalLayout_25)
        self.verticalLayout_7.addLayout(self.horizontalLayout_28)
        self.horizontalLayout_30 = QtGui.QHBoxLayout()
        self.horizontalLayout_30.setObjectName(_fromUtf8("horizontalLayout_30"))
        self.label_29 = QtGui.QLabel(self.tab)
        self.label_29.setMinimumSize(QtCore.QSize(121, 16))
        self.label_29.setMaximumSize(QtCore.QSize(121, 16))
        self.label_29.setObjectName(_fromUtf8("label_29"))
        self.horizontalLayout_30.addWidget(self.label_29)
        self.bufferSizeBox = QtGui.QSpinBox(self.tab)
        self.bufferSizeBox.setMinimumSize(QtCore.QSize(181, 25))
        self.bufferSizeBox.setMaximum(99999)
        self.bufferSizeBox.setObjectName(_fromUtf8("bufferSizeBox"))
        self.horizontalLayout_30.addWidget(self.bufferSizeBox)
        self.verticalLayout_7.addLayout(self.horizontalLayout_30)
        spacerItem = QtGui.QSpacerItem(20, 168, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout_7.addItem(spacerItem)
        self.preferenceTab.addTab(self.tab, _fromUtf8(""))
        self.tab_2 = QtGui.QWidget()
        self.tab_2.setObjectName(_fromUtf8("tab_2"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.tab_2)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.measurementSettingsList = QtGui.QListWidget(self.tab_2)
        self.measurementSettingsList.setMaximumSize(QtCore.QSize(171, 16777215))
        self.measurementSettingsList.setObjectName(_fromUtf8("measurementSettingsList"))
        item = QtGui.QListWidgetItem()
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(_fromUtf8(":/Icons/signal.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        item.setIcon(icon1)
        self.measurementSettingsList.addItem(item)
        item = QtGui.QListWidgetItem()
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(_fromUtf8(":/Icons/filter.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        item.setIcon(icon2)
        self.measurementSettingsList.addItem(item)
        item = QtGui.QListWidgetItem()
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(_fromUtf8(":/Icons/impulse.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        item.setIcon(icon3)
        self.measurementSettingsList.addItem(item)
        self.horizontalLayout.addWidget(self.measurementSettingsList)
        self.measurementStacked = QtGui.QStackedWidget(self.tab_2)
        self.measurementStacked.setObjectName(_fromUtf8("measurementStacked"))
        self.page = QtGui.QWidget()
        self.page.setObjectName(_fromUtf8("page"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.page)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.horizontalLayout_5 = QtGui.QHBoxLayout()
        self.horizontalLayout_5.setObjectName(_fromUtf8("horizontalLayout_5"))
        self.label = QtGui.QLabel(self.page)
        self.label.setMaximumSize(QtCore.QSize(38, 16))
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout_5.addWidget(self.label)
        self.signalBox = QtGui.QComboBox(self.page)
        self.signalBox.setObjectName(_fromUtf8("signalBox"))
        self.signalBox.addItem(_fromUtf8(""))
        self.signalBox.addItem(_fromUtf8(""))
        self.signalBox.addItem(_fromUtf8(""))
        self.signalBox.addItem(_fromUtf8(""))
        self.horizontalLayout_5.addWidget(self.signalBox)
        self.verticalLayout_2.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_6 = QtGui.QHBoxLayout()
        self.horizontalLayout_6.setObjectName(_fromUtf8("horizontalLayout_6"))
        self.label_11 = QtGui.QLabel(self.page)
        self.label_11.setMaximumSize(QtCore.QSize(121, 16))
        self.label_11.setObjectName(_fromUtf8("label_11"))
        self.horizontalLayout_6.addWidget(self.label_11)
        self.signalRepetitionsBox = QtGui.QSpinBox(self.page)
        self.signalRepetitionsBox.setObjectName(_fromUtf8("signalRepetitionsBox"))
        self.horizontalLayout_6.addWidget(self.signalRepetitionsBox)
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem1)
        self.verticalLayout_2.addLayout(self.horizontalLayout_6)
        self.signalStacked = QtGui.QStackedWidget(self.page)
        self.signalStacked.setObjectName(_fromUtf8("signalStacked"))
        self.page_3 = QtGui.QWidget()
        self.page_3.setObjectName(_fromUtf8("page_3"))
        self.label_2 = QtGui.QLabel(self.page_3)
        self.label_2.setGeometry(QtCore.QRect(10, 20, 111, 16))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.label_3 = QtGui.QLabel(self.page_3)
        self.label_3.setGeometry(QtCore.QRect(10, 50, 111, 16))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.label_4 = QtGui.QLabel(self.page_3)
        self.label_4.setGeometry(QtCore.QRect(20, 80, 91, 16))
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.layoutWidget = QtGui.QWidget(self.page_3)
        self.layoutWidget.setGeometry(QtCore.QRect(150, 50, 161, 27))
        self.layoutWidget.setObjectName(_fromUtf8("layoutWidget"))
        self.horizontalLayout_3 = QtGui.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_3.setMargin(0)
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.upperFrequencyBox = QtGui.QSpinBox(self.layoutWidget)
        self.upperFrequencyBox.setMaximum(99999)
        self.upperFrequencyBox.setObjectName(_fromUtf8("upperFrequencyBox"))
        self.horizontalLayout_3.addWidget(self.upperFrequencyBox)
        self.label_6 = QtGui.QLabel(self.layoutWidget)
        self.label_6.setMaximumSize(QtCore.QSize(21, 16))
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.horizontalLayout_3.addWidget(self.label_6)
        self.layoutWidget1 = QtGui.QWidget(self.page_3)
        self.layoutWidget1.setGeometry(QtCore.QRect(150, 20, 161, 27))
        self.layoutWidget1.setObjectName(_fromUtf8("layoutWidget1"))
        self.horizontalLayout_4 = QtGui.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout_4.setMargin(0)
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.lowerFrequencyBox = QtGui.QSpinBox(self.layoutWidget1)
        self.lowerFrequencyBox.setMaximum(99999)
        self.lowerFrequencyBox.setObjectName(_fromUtf8("lowerFrequencyBox"))
        self.horizontalLayout_4.addWidget(self.lowerFrequencyBox)
        self.label_5 = QtGui.QLabel(self.layoutWidget1)
        self.label_5.setMaximumSize(QtCore.QSize(21, 16))
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.horizontalLayout_4.addWidget(self.label_5)
        self.layoutWidget2 = QtGui.QWidget(self.page_3)
        self.layoutWidget2.setGeometry(QtCore.QRect(150, 80, 161, 27))
        self.layoutWidget2.setObjectName(_fromUtf8("layoutWidget2"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout(self.layoutWidget2)
        self.horizontalLayout_2.setMargin(0)
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.signalLengthBox = QtGui.QSpinBox(self.layoutWidget2)
        self.signalLengthBox.setMaximum(99999)
        self.signalLengthBox.setObjectName(_fromUtf8("signalLengthBox"))
        self.horizontalLayout_2.addWidget(self.signalLengthBox)
        self.label_7 = QtGui.QLabel(self.layoutWidget2)
        self.label_7.setMaximumSize(QtCore.QSize(21, 16))
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.horizontalLayout_2.addWidget(self.label_7)
        self.signalStacked.addWidget(self.page_3)
        self.page_4 = QtGui.QWidget()
        self.page_4.setObjectName(_fromUtf8("page_4"))
        self.verticalLayout_8 = QtGui.QVBoxLayout(self.page_4)
        self.verticalLayout_8.setObjectName(_fromUtf8("verticalLayout_8"))
        self.horizontalLayout_31 = QtGui.QHBoxLayout()
        self.horizontalLayout_31.setObjectName(_fromUtf8("horizontalLayout_31"))
        self.label_9 = QtGui.QLabel(self.page_4)
        self.label_9.setObjectName(_fromUtf8("label_9"))
        self.horizontalLayout_31.addWidget(self.label_9)
        self.mlsTapsBox = QtGui.QSpinBox(self.page_4)
        self.mlsTapsBox.setObjectName(_fromUtf8("mlsTapsBox"))
        self.horizontalLayout_31.addWidget(self.mlsTapsBox)
        spacerItem2 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_31.addItem(spacerItem2)
        self.verticalLayout_8.addLayout(self.horizontalLayout_31)
        self.horizontalLayout_29 = QtGui.QHBoxLayout()
        self.horizontalLayout_29.setObjectName(_fromUtf8("horizontalLayout_29"))
        self.label_10 = QtGui.QLabel(self.page_4)
        self.label_10.setObjectName(_fromUtf8("label_10"))
        self.horizontalLayout_29.addWidget(self.label_10)
        self.mlsRepetitionsBox = QtGui.QSpinBox(self.page_4)
        self.mlsRepetitionsBox.setObjectName(_fromUtf8("mlsRepetitionsBox"))
        self.horizontalLayout_29.addWidget(self.mlsRepetitionsBox)
        spacerItem3 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_29.addItem(spacerItem3)
        self.verticalLayout_8.addLayout(self.horizontalLayout_29)
        self.signalLengthLabel = QtGui.QLabel(self.page_4)
        self.signalLengthLabel.setObjectName(_fromUtf8("signalLengthLabel"))
        self.verticalLayout_8.addWidget(self.signalLengthLabel)
        spacerItem4 = QtGui.QSpacerItem(20, 99, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout_8.addItem(spacerItem4)
        self.signalStacked.addWidget(self.page_4)
        self.verticalLayout_2.addWidget(self.signalStacked)
        self.measurementStacked.addWidget(self.page)
        self.page_2 = QtGui.QWidget()
        self.page_2.setObjectName(_fromUtf8("page_2"))
        self.verticalLayout_3 = QtGui.QVBoxLayout(self.page_2)
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.groupBox = QtGui.QGroupBox(self.page_2)
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.verticalLayout_5 = QtGui.QVBoxLayout(self.groupBox)
        self.verticalLayout_5.setObjectName(_fromUtf8("verticalLayout_5"))
        self.horizontalLayout_9 = QtGui.QHBoxLayout()
        self.horizontalLayout_9.setObjectName(_fromUtf8("horizontalLayout_9"))
        self.label_12 = QtGui.QLabel(self.groupBox)
        self.label_12.setObjectName(_fromUtf8("label_12"))
        self.horizontalLayout_9.addWidget(self.label_12)
        self.horizontalLayout_7 = QtGui.QHBoxLayout()
        self.horizontalLayout_7.setObjectName(_fromUtf8("horizontalLayout_7"))
        self.highPassCutOffBox = QtGui.QSpinBox(self.groupBox)
        self.highPassCutOffBox.setMaximumSize(QtCore.QSize(128, 25))
        self.highPassCutOffBox.setMaximum(99999)
        self.highPassCutOffBox.setObjectName(_fromUtf8("highPassCutOffBox"))
        self.horizontalLayout_7.addWidget(self.highPassCutOffBox)
        self.label_14 = QtGui.QLabel(self.groupBox)
        self.label_14.setMaximumSize(QtCore.QSize(21, 16))
        self.label_14.setObjectName(_fromUtf8("label_14"))
        self.horizontalLayout_7.addWidget(self.label_14)
        self.horizontalLayout_9.addLayout(self.horizontalLayout_7)
        self.verticalLayout_5.addLayout(self.horizontalLayout_9)
        self.horizontalLayout_10 = QtGui.QHBoxLayout()
        self.horizontalLayout_10.setObjectName(_fromUtf8("horizontalLayout_10"))
        self.label_13 = QtGui.QLabel(self.groupBox)
        self.label_13.setObjectName(_fromUtf8("label_13"))
        self.horizontalLayout_10.addWidget(self.label_13)
        spacerItem5 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_10.addItem(spacerItem5)
        self.horizontalLayout_8 = QtGui.QHBoxLayout()
        self.horizontalLayout_8.setObjectName(_fromUtf8("horizontalLayout_8"))
        self.highPassOrderBox = QtGui.QSpinBox(self.groupBox)
        self.highPassOrderBox.setObjectName(_fromUtf8("highPassOrderBox"))
        self.horizontalLayout_8.addWidget(self.highPassOrderBox)
        spacerItem6 = QtGui.QSpacerItem(88, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem6)
        self.horizontalLayout_10.addLayout(self.horizontalLayout_8)
        self.verticalLayout_5.addLayout(self.horizontalLayout_10)
        self.horizontalLayout_15 = QtGui.QHBoxLayout()
        self.horizontalLayout_15.setObjectName(_fromUtf8("horizontalLayout_15"))
        self.highPassEnabledBox = QtGui.QCheckBox(self.groupBox)
        self.highPassEnabledBox.setObjectName(_fromUtf8("highPassEnabledBox"))
        self.horizontalLayout_15.addWidget(self.highPassEnabledBox)
        spacerItem7 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_15.addItem(spacerItem7)
        self.verticalLayout_5.addLayout(self.horizontalLayout_15)
        self.verticalLayout_3.addWidget(self.groupBox)
        self.groupBox_2 = QtGui.QGroupBox(self.page_2)
        self.groupBox_2.setObjectName(_fromUtf8("groupBox_2"))
        self.verticalLayout_4 = QtGui.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.horizontalLayout_11 = QtGui.QHBoxLayout()
        self.horizontalLayout_11.setObjectName(_fromUtf8("horizontalLayout_11"))
        self.label_15 = QtGui.QLabel(self.groupBox_2)
        self.label_15.setObjectName(_fromUtf8("label_15"))
        self.horizontalLayout_11.addWidget(self.label_15)
        self.horizontalLayout_12 = QtGui.QHBoxLayout()
        self.horizontalLayout_12.setObjectName(_fromUtf8("horizontalLayout_12"))
        self.lowPassCutOffBox = QtGui.QSpinBox(self.groupBox_2)
        self.lowPassCutOffBox.setMaximumSize(QtCore.QSize(128, 25))
        self.lowPassCutOffBox.setMaximum(99999)
        self.lowPassCutOffBox.setObjectName(_fromUtf8("lowPassCutOffBox"))
        self.horizontalLayout_12.addWidget(self.lowPassCutOffBox)
        self.label_16 = QtGui.QLabel(self.groupBox_2)
        self.label_16.setMaximumSize(QtCore.QSize(21, 16))
        self.label_16.setObjectName(_fromUtf8("label_16"))
        self.horizontalLayout_12.addWidget(self.label_16)
        self.horizontalLayout_11.addLayout(self.horizontalLayout_12)
        self.verticalLayout_4.addLayout(self.horizontalLayout_11)
        self.horizontalLayout_13 = QtGui.QHBoxLayout()
        self.horizontalLayout_13.setObjectName(_fromUtf8("horizontalLayout_13"))
        self.label_17 = QtGui.QLabel(self.groupBox_2)
        self.label_17.setObjectName(_fromUtf8("label_17"))
        self.horizontalLayout_13.addWidget(self.label_17)
        spacerItem8 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_13.addItem(spacerItem8)
        self.horizontalLayout_14 = QtGui.QHBoxLayout()
        self.horizontalLayout_14.setObjectName(_fromUtf8("horizontalLayout_14"))
        self.lowPassOrderBox = QtGui.QSpinBox(self.groupBox_2)
        self.lowPassOrderBox.setObjectName(_fromUtf8("lowPassOrderBox"))
        self.horizontalLayout_14.addWidget(self.lowPassOrderBox)
        spacerItem9 = QtGui.QSpacerItem(88, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_14.addItem(spacerItem9)
        self.horizontalLayout_13.addLayout(self.horizontalLayout_14)
        self.verticalLayout_4.addLayout(self.horizontalLayout_13)
        self.horizontalLayout_16 = QtGui.QHBoxLayout()
        self.horizontalLayout_16.setObjectName(_fromUtf8("horizontalLayout_16"))
        self.lowPassEnabledBox = QtGui.QCheckBox(self.groupBox_2)
        self.lowPassEnabledBox.setObjectName(_fromUtf8("lowPassEnabledBox"))
        self.horizontalLayout_16.addWidget(self.lowPassEnabledBox)
        spacerItem10 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_16.addItem(spacerItem10)
        self.verticalLayout_4.addLayout(self.horizontalLayout_16)
        self.verticalLayout_3.addWidget(self.groupBox_2)
        self.measurementStacked.addWidget(self.page_2)
        self.page_5 = QtGui.QWidget()
        self.page_5.setObjectName(_fromUtf8("page_5"))
        self.verticalLayout_6 = QtGui.QVBoxLayout(self.page_5)
        self.verticalLayout_6.setObjectName(_fromUtf8("verticalLayout_6"))
        self.horizontalLayout_23 = QtGui.QHBoxLayout()
        self.horizontalLayout_23.setObjectName(_fromUtf8("horizontalLayout_23"))
        self.label_18 = QtGui.QLabel(self.page_5)
        self.label_18.setObjectName(_fromUtf8("label_18"))
        self.horizontalLayout_23.addWidget(self.label_18)
        self.horizontalLayout_19 = QtGui.QHBoxLayout()
        self.horizontalLayout_19.setObjectName(_fromUtf8("horizontalLayout_19"))
        self.windowStartBox = QtGui.QDoubleSpinBox(self.page_5)
        self.windowStartBox.setMaximumSize(QtCore.QSize(91, 25))
        self.windowStartBox.setDecimals(5)
        self.windowStartBox.setObjectName(_fromUtf8("windowStartBox"))
        self.horizontalLayout_19.addWidget(self.windowStartBox)
        self.label_22 = QtGui.QLabel(self.page_5)
        self.label_22.setObjectName(_fromUtf8("label_22"))
        self.horizontalLayout_19.addWidget(self.label_22)
        self.horizontalLayout_23.addLayout(self.horizontalLayout_19)
        self.verticalLayout_6.addLayout(self.horizontalLayout_23)
        self.horizontalLayout_22 = QtGui.QHBoxLayout()
        self.horizontalLayout_22.setObjectName(_fromUtf8("horizontalLayout_22"))
        self.label_19 = QtGui.QLabel(self.page_5)
        self.label_19.setObjectName(_fromUtf8("label_19"))
        self.horizontalLayout_22.addWidget(self.label_19)
        self.horizontalLayout_18 = QtGui.QHBoxLayout()
        self.horizontalLayout_18.setObjectName(_fromUtf8("horizontalLayout_18"))
        self.windowEndBox = QtGui.QDoubleSpinBox(self.page_5)
        self.windowEndBox.setMaximumSize(QtCore.QSize(91, 25))
        self.windowEndBox.setDecimals(5)
        self.windowEndBox.setObjectName(_fromUtf8("windowEndBox"))
        self.horizontalLayout_18.addWidget(self.windowEndBox)
        self.label_23 = QtGui.QLabel(self.page_5)
        self.label_23.setObjectName(_fromUtf8("label_23"))
        self.horizontalLayout_18.addWidget(self.label_23)
        self.horizontalLayout_22.addLayout(self.horizontalLayout_18)
        self.verticalLayout_6.addLayout(self.horizontalLayout_22)
        self.horizontalLayout_21 = QtGui.QHBoxLayout()
        self.horizontalLayout_21.setObjectName(_fromUtf8("horizontalLayout_21"))
        self.label_21 = QtGui.QLabel(self.page_5)
        self.label_21.setObjectName(_fromUtf8("label_21"))
        self.horizontalLayout_21.addWidget(self.label_21)
        self.windowTypeBox = QtGui.QComboBox(self.page_5)
        self.windowTypeBox.setObjectName(_fromUtf8("windowTypeBox"))
        self.windowTypeBox.addItem(_fromUtf8(""))
        self.windowTypeBox.addItem(_fromUtf8(""))
        self.horizontalLayout_21.addWidget(self.windowTypeBox)
        self.verticalLayout_6.addLayout(self.horizontalLayout_21)
        self.horizontalLayout_20 = QtGui.QHBoxLayout()
        self.horizontalLayout_20.setObjectName(_fromUtf8("horizontalLayout_20"))
        self.label_20 = QtGui.QLabel(self.page_5)
        self.label_20.setObjectName(_fromUtf8("label_20"))
        self.horizontalLayout_20.addWidget(self.label_20)
        self.horizontalLayout_17 = QtGui.QHBoxLayout()
        self.horizontalLayout_17.setObjectName(_fromUtf8("horizontalLayout_17"))
        self.taperLengthBox = QtGui.QDoubleSpinBox(self.page_5)
        self.taperLengthBox.setMaximumSize(QtCore.QSize(91, 25))
        self.taperLengthBox.setDecimals(5)
        self.taperLengthBox.setObjectName(_fromUtf8("taperLengthBox"))
        self.horizontalLayout_17.addWidget(self.taperLengthBox)
        self.label_24 = QtGui.QLabel(self.page_5)
        self.label_24.setObjectName(_fromUtf8("label_24"))
        self.horizontalLayout_17.addWidget(self.label_24)
        self.horizontalLayout_20.addLayout(self.horizontalLayout_17)
        self.verticalLayout_6.addLayout(self.horizontalLayout_20)
        self.horizontalLayout_24 = QtGui.QHBoxLayout()
        self.horizontalLayout_24.setObjectName(_fromUtf8("horizontalLayout_24"))
        self.label_25 = QtGui.QLabel(self.page_5)
        self.label_25.setObjectName(_fromUtf8("label_25"))
        self.horizontalLayout_24.addWidget(self.label_25)
        spacerItem11 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_24.addItem(spacerItem11)
        self.decimationBox = QtGui.QSpinBox(self.page_5)
        self.decimationBox.setObjectName(_fromUtf8("decimationBox"))
        self.horizontalLayout_24.addWidget(self.decimationBox)
        spacerItem12 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_24.addItem(spacerItem12)
        self.verticalLayout_6.addLayout(self.horizontalLayout_24)
        spacerItem13 = QtGui.QSpacerItem(20, 137, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout_6.addItem(spacerItem13)
        self.measurementStacked.addWidget(self.page_5)
        self.horizontalLayout.addWidget(self.measurementStacked)
        self.preferenceTab.addTab(self.tab_2, _fromUtf8(""))
        self.verticalLayout.addWidget(self.preferenceTab)
        self.buttonBox = QtGui.QDialogButtonBox(Preference)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(Preference)
        self.preferenceTab.setCurrentIndex(0)
        self.measurementSettingsList.setCurrentRow(0)
        self.measurementStacked.setCurrentIndex(0)
        self.signalStacked.setCurrentIndex(1)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), Preference.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), Preference.reject)
        QtCore.QMetaObject.connectSlotsByName(Preference)

    def retranslateUi(self, Preference):
        Preference.setWindowTitle(QtGui.QApplication.translate("Preference", "Dialog", None, QtGui.QApplication.UnicodeUTF8))
        self.label_26.setText(QtGui.QApplication.translate("Preference", "Input Device", None, QtGui.QApplication.UnicodeUTF8))
        self.label_27.setText(QtGui.QApplication.translate("Preference", "Output Device", None, QtGui.QApplication.UnicodeUTF8))
        self.label_28.setText(QtGui.QApplication.translate("Preference", "Gain", None, QtGui.QApplication.UnicodeUTF8))
        self.gainLabel.setText(QtGui.QApplication.translate("Preference", "-10.00 dB", None, QtGui.QApplication.UnicodeUTF8))
        self.label_29.setText(QtGui.QApplication.translate("Preference", "Buffer Size", None, QtGui.QApplication.UnicodeUTF8))
        self.preferenceTab.setTabText(self.preferenceTab.indexOf(self.tab), QtGui.QApplication.translate("Preference", "Audio Preferences", None, QtGui.QApplication.UnicodeUTF8))
        __sortingEnabled = self.measurementSettingsList.isSortingEnabled()
        self.measurementSettingsList.setSortingEnabled(False)
        item = self.measurementSettingsList.item(0)
        item.setText(QtGui.QApplication.translate("Preference", "Excitation Signal", None, QtGui.QApplication.UnicodeUTF8))
        item = self.measurementSettingsList.item(1)
        item.setText(QtGui.QApplication.translate("Preference", "Filter Settings", None, QtGui.QApplication.UnicodeUTF8))
        item = self.measurementSettingsList.item(2)
        item.setText(QtGui.QApplication.translate("Preference", "Exctraction Settings", None, QtGui.QApplication.UnicodeUTF8))
        self.measurementSettingsList.setSortingEnabled(__sortingEnabled)
        self.label.setText(QtGui.QApplication.translate("Preference", "Signal", None, QtGui.QApplication.UnicodeUTF8))
        self.signalBox.setItemText(0, QtGui.QApplication.translate("Preference", "Swept Sine", None, QtGui.QApplication.UnicodeUTF8))
        self.signalBox.setItemText(1, QtGui.QApplication.translate("Preference", "Low Pass Swept Sine", None, QtGui.QApplication.UnicodeUTF8))
        self.signalBox.setItemText(2, QtGui.QApplication.translate("Preference", "Maximum Length Sequence", None, QtGui.QApplication.UnicodeUTF8))
        self.signalBox.setItemText(3, QtGui.QApplication.translate("Preference", "Inverse Repeat Sequence", None, QtGui.QApplication.UnicodeUTF8))
        self.label_11.setText(QtGui.QApplication.translate("Preference", "Signal Repetitions", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("Preference", "Lower Frequency", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("Preference", "Upper Frequency", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(QtGui.QApplication.translate("Preference", "Signal Length", None, QtGui.QApplication.UnicodeUTF8))
        self.label_6.setText(QtGui.QApplication.translate("Preference", "Hz", None, QtGui.QApplication.UnicodeUTF8))
        self.label_5.setText(QtGui.QApplication.translate("Preference", "Hz", None, QtGui.QApplication.UnicodeUTF8))
        self.label_7.setText(QtGui.QApplication.translate("Preference", "ms", None, QtGui.QApplication.UnicodeUTF8))
        self.label_9.setText(QtGui.QApplication.translate("Preference", "MLS Taps", None, QtGui.QApplication.UnicodeUTF8))
        self.label_10.setText(QtGui.QApplication.translate("Preference", "Repetitions", None, QtGui.QApplication.UnicodeUTF8))
        self.signalLengthLabel.setText(QtGui.QApplication.translate("Preference", "Signal Length", None, QtGui.QApplication.UnicodeUTF8))
        self.groupBox.setTitle(QtGui.QApplication.translate("Preference", "High Pass Filter", None, QtGui.QApplication.UnicodeUTF8))
        self.label_12.setText(QtGui.QApplication.translate("Preference", "Cut Off Frequency", None, QtGui.QApplication.UnicodeUTF8))
        self.label_14.setText(QtGui.QApplication.translate("Preference", "Hz", None, QtGui.QApplication.UnicodeUTF8))
        self.label_13.setText(QtGui.QApplication.translate("Preference", "Order", None, QtGui.QApplication.UnicodeUTF8))
        self.highPassEnabledBox.setText(QtGui.QApplication.translate("Preference", "Enabled", None, QtGui.QApplication.UnicodeUTF8))
        self.groupBox_2.setTitle(QtGui.QApplication.translate("Preference", "Low Pass Filter", None, QtGui.QApplication.UnicodeUTF8))
        self.label_15.setText(QtGui.QApplication.translate("Preference", "Cut Off Frequency", None, QtGui.QApplication.UnicodeUTF8))
        self.label_16.setText(QtGui.QApplication.translate("Preference", "Hz", None, QtGui.QApplication.UnicodeUTF8))
        self.label_17.setText(QtGui.QApplication.translate("Preference", "Order", None, QtGui.QApplication.UnicodeUTF8))
        self.lowPassEnabledBox.setText(QtGui.QApplication.translate("Preference", "Enabled", None, QtGui.QApplication.UnicodeUTF8))
        self.label_18.setText(QtGui.QApplication.translate("Preference", "Window Start", None, QtGui.QApplication.UnicodeUTF8))
        self.label_22.setText(QtGui.QApplication.translate("Preference", "ms", None, QtGui.QApplication.UnicodeUTF8))
        self.label_19.setText(QtGui.QApplication.translate("Preference", "Window End", None, QtGui.QApplication.UnicodeUTF8))
        self.label_23.setText(QtGui.QApplication.translate("Preference", "ms", None, QtGui.QApplication.UnicodeUTF8))
        self.label_21.setText(QtGui.QApplication.translate("Preference", "Window Type", None, QtGui.QApplication.UnicodeUTF8))
        self.windowTypeBox.setItemText(0, QtGui.QApplication.translate("Preference", "One Sided", None, QtGui.QApplication.UnicodeUTF8))
        self.windowTypeBox.setItemText(1, QtGui.QApplication.translate("Preference", "Two Sided", None, QtGui.QApplication.UnicodeUTF8))
        self.label_20.setText(QtGui.QApplication.translate("Preference", "Taper Length", None, QtGui.QApplication.UnicodeUTF8))
        self.label_24.setText(QtGui.QApplication.translate("Preference", "ms", None, QtGui.QApplication.UnicodeUTF8))
        self.label_25.setText(QtGui.QApplication.translate("Preference", "Decimation Factor", None, QtGui.QApplication.UnicodeUTF8))
        self.preferenceTab.setTabText(self.preferenceTab.indexOf(self.tab_2), QtGui.QApplication.translate("Preference", "Measurement Settings", None, QtGui.QApplication.UnicodeUTF8))

import icons_rc
