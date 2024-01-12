# This file is a part of the program: MultiVariate Viz
# A simple python GUI tool to visualize multivariate data and create PCA models
# (see <https://github.com/hilary-luo/mv_viz>)
#
# Copyright (c) 2024, Hilary Luo and contributors, All right reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import os
import numpy as np
import pandas as pd
import pyqtgraph as pg
import sklearn.decomposition as skd

from PyQt6.QtCore import (
    pyqtProperty,
    pyqtSignal,
    pyqtSlot,
    QAbstractTableModel,
    QModelIndex,
    QObject,
    QSize,
    Qt,
    QVariant
)
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QStatusBar,
    QTabBar,
    QTableView,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

class DataFrameModel(QAbstractTableModel):
    DtypeRole = Qt.ItemDataRole.UserRole + 1000
    ValueRole = Qt.ItemDataRole.UserRole + 1001

    def __init__(self, df=pd.DataFrame(), parent=None):
        super().__init__(parent)
        self._dataframe = df

    def setDataFrame(self, dataframe):
        self.beginResetModel()
        self._dataframe = dataframe.copy()
        self.endResetModel()

    def dataFrame(self):
        return self._dataframe

    dataFrame = pyqtProperty(pd.DataFrame, fget=dataFrame, fset=setDataFrame)

    @pyqtSlot(int, Qt.Orientation, result=str)
    def headerData(self, section: int, orientation: Qt.Orientation,
                   role: int = Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return self._dataframe.columns[section]
            return str(self._dataframe.index[section])
        return QVariant()

    def rowCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        return len(self._dataframe.index)

    def columnCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        return self._dataframe.columns.size

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if (not index.isValid() or not (0 <= index.row() < self.rowCount()
            and 0 <= index.column() < self.columnCount())):
            return QVariant()
        row = self._dataframe.index[index.row()]
        col = self._dataframe.columns[index.column()]
        dt = self._dataframe[col].dtype

        val = self._dataframe.loc[row][col]
        if role == Qt.ItemDataRole.DisplayRole:
            return str(val)
        if role == DataFrameModel.ValueRole:
            return val
        if role == DataFrameModel.DtypeRole:
            return dt
        return QVariant()

    def roleNames(self):
        roles = {
            Qt.ItemDataRole.DisplayRole: b'display',
            DataFrameModel.DtypeRole: b'dtype',
            DataFrameModel.ValueRole: b'value'
        }
        return roles

class PCA(QObject):
    pcaStatusChanged = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.calculated = False
        self.dimensions = 1
        self.components = []
        self.pca_model = []
        self.data = []
        self.data_modeled = []
        self.scores = []
        self.model_status = "No Model"
        self.pca_widget = QWidget()
        self.pcaStatusChanged.emit(self.calculated)

    def setup(self, data: pd.DataFrame, dimensions: int):
        self.data = data
        self.dimensions = dimensions
        self.pca_model = skd.PCA(n_components=self.dimensions)
        scores = self.pca_model.fit_transform(self.data)
        self.scores = pd.DataFrame(scores, index=self.data.index,
                                   columns=range(0,scores.shape[1]))
        self.components = self.pca_model.components_
        self.calculated = True
        self.pcaStatusChanged.emit(self.calculated)
        i = 0
        for col in self.scores:
            if i == 0:
                self.data_modeled = np.outer(self.scores[col],
                                             self.components[i])
            else:
                self.data_modeled = (self.data_modeled +
                                     np.outer(self.scores[col],
                                              self.components[i]))

    def set_component_count(self, dimensions: int):
        self.setup(self.data, dimensions)

    def get_r2(self):  # R^2 per component
        if self.calculated:
            return self.pca_model.explained_variance_ratio_
        return 0

    def get_components(self):
        if self.calculated:
            return self.components
        return []

    def get_scores(self, dim: int):
        if self.calculated:
            return self.scores.iloc[:,dim-1]
        return []

    def get_labels(self):
        return self.data.index

    def get_spe(self) -> pd.DataFrame:
        return np.sum(np.square(self.data - self.data_modeled), axis = 1)

    def get_hotellings_t2(self) -> pd.DataFrame:
        std_dev = self.scores.std(axis=0)
        return np.sum(np.square(np.divide(self.scores, std_dev)), axis=1)


class PlotSelectWindow(QWidget):
    # A QWidget without a parent will appear as a free-floating window

    add_plot_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.layout = QFormLayout()
        self.setWindowTitle("Select Plot")
        self.label = QLabel("Select Plot")
        self.plot_option = QComboBox()
        self.plot_option.addItem("Loading Plot")
        self.plot_option.addItem("Score Plot")
        self.plot_option.addItem("SPE")
        self.plot_option.addItem("Hotelling's T2")
        self.layout.addRow("Plot Type", self.plot_option)
        self.btn_submit = QPushButton("Submit")
        self.btn_cancel = QPushButton("Cancel")
        self.layout.addRow(self.btn_submit, self.btn_cancel)
        self.setLayout(self.layout)
        self.btn_submit.clicked.connect(self.submit)
        self.btn_cancel.clicked.connect(self.close_window)

    def submit(self):
        self.add_plot_signal.emit(self.plot_option.currentText())
        self.close()

    def close_window(self):
        self.close()


class PCAConfigWindow(QWidget):
    # A QWidget without a parent will appear as a free-floating window

    pca_configured_signal = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.layout = QFormLayout()
        self.setWindowTitle("Configure PCA")
        self.label = QLabel("Configure PCA")
        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setRange(1,8)
        self.slider_label = QLabel(str(self.slider.value()))
        self.layout.addRow("Number of Dimensions", self.slider)
        self.layout.addRow("", self.slider_label)
        self.btn_submit = QPushButton("Submit")
        self.btn_cancel = QPushButton("Cancel")
        self.layout.addRow(self.btn_submit, self.btn_cancel)
        self.setLayout(self.layout)
        self.btn_submit.clicked.connect(self.submit)
        self.btn_cancel.clicked.connect(self.close_window)
        self.slider.valueChanged.connect(self.update_label)

    def update_label(self):
        self.slider_label.setText(str(self.slider.value()))

    def submit(self):
        self.pca_configured_signal.emit(self.slider.value())
        self.close()

    def close_window(self):
        self.close()


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.media_path = os.path.join(os.path.dirname(__file__), "media")

        self.setWindowTitle("MultiVariate Viz")
        self.resize(800, 600)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)

        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.model_status_label = QLabel("No Model")

        self.status_bar.addPermanentWidget(QLabel("Model: "))
        self.status_bar.addPermanentWidget(self.model_status_label)

        self.user_df_raw = []
        self.user_df = []
        self.pca = PCA()
        self.tableview = QTableView()
        self.table_model = []
        self.table_tab = self.tableview

        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(QLabel("Model: "))
        self.dialog = None

        self.pca.pcaStatusChanged.connect(self.pca_status_changed)

        self.init_toolbar()
        self.init_tabs()

    def init_toolbar(self):
        self.toolbar = QToolBar("Main Toolbar")
        self.toolbar.setIconSize(QSize(20,20))
        self.toolbar.setMovable(False)
        self.addToolBar(self.toolbar)

        # Data Section
        self.btn_import = QAction(QIcon(os.path.join(
            self.media_path, "table-import.png")), "Import Data", self)
        self.btn_import.setStatusTip("Select data...")
        self.btn_import.triggered.connect(self.btn_import_onclick)
        self.toolbar.addAction(self.btn_import)

        self.btn_explore_data = QAction(QIcon(os.path.join(
            self.media_path, "document-table.png")), "Explore Data", self)
        self.btn_explore_data.setStatusTip(
            "A view to explore data relationships...")
        self.btn_explore_data.triggered.connect(self.btn_explore_data_tab_onclick)
        self.btn_explore_data.setDisabled(True)
        self.toolbar.addAction(self.btn_explore_data)

        self.toolbar.addSeparator()

        # PCA Section
        self.btn_pca = QAction(QIcon(os.path.join(
            self.media_path, "block.png")), "PCA Model", self)
        self.btn_pca.setStatusTip("Generate PCA Model")
        self.btn_pca.triggered.connect(self.btn_pca_onclick)
        self.btn_pca.setDisabled(True)
        self.toolbar.addAction(self.btn_pca)

        self.btn_pls = QAction(QIcon(os.path.join(
            self.media_path, "block.png")), "PLS Model", self)
        self.btn_pls.setStatusTip("Not implemented -- Generate PLS Model")
        self.btn_pls.triggered.connect(self.btn_pls_onclick)
        self.btn_pls.setDisabled(True)
        self.toolbar.addAction(self.btn_pls)

        self.btn_add_plots = QAction(QIcon(os.path.join(
            self.media_path, "chart.png")), "Add Plots", self)
        self.btn_add_plots.setStatusTip("Add Plots...")
        self.btn_add_plots.triggered.connect(self.btn_add_plot_onclick)
        self.btn_add_plots.setDisabled(True)
        self.toolbar.addAction(self.btn_add_plots)

        self.toolbar.addSeparator()

        self.menu = self.menuBar()
        self.file_menu = self.menu.addMenu("&File")
        self.file_menu.addAction(self.btn_import)
        self.file_menu.addSeparator()

    def init_tabs(self):
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.setMovable(True)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        self.tabs.addTab(self.table_tab, "Raw Data")
        self.tabs.tabBar().setTabButton(0, QTabBar.ButtonPosition.RightSide,
                                        None)
        self.setCentralWidget(self.tabs)

    def close_tab (self, index):
        self.tabs.removeTab(index)

    def pca_status_changed(self, status):
        self.btn_add_plots.setDisabled(not status)
        self.model_status_label.setText("PCA Model Successful")
        self.update()

    def loading_plot(self):
        tab = QWidget()
        loading_plot_layout = QVBoxLayout()
        i = 1
        for p in self.pca.get_components():
            loading_plot_layout.addWidget(
                QLabel("Loading plot " + str(i)))
            loading_plot = pg.PlotWidget()
            graph = pg.BarGraphItem(x = range(1,len(p)+1),
                                    height = p, width = 0.6)
            loading_plot.addItem(graph)
            loading_plot_layout.addWidget(loading_plot)
            i = i + 1
        tab.setLayout(loading_plot_layout)
        return tab

    def spe_plot(self):
        widget = pg.PlotWidget()
        spe = self.pca.get_spe()
        plotitem = pg.ScatterPlotItem(pen=pg.mkPen(width=5, color='w'),
                                      symbol='o', size=1)
        plotitem.addPoints(range(0,spe.shape[0]), spe, hoverable=True,
                           hoverSymbol='star', data=spe.index)
        plotitem.sigClicked.connect(self.scoreplot_onclick)
        widget.addItem(plotitem)
        return widget

    def hotelling_plot(self):
        widget = pg.PlotWidget()
        hotelling = self.pca.get_hotellings_t2()
        plotitem = pg.ScatterPlotItem(pen=pg.mkPen(width=5, color='w'),
                                      symbol='o', size=1)
        plotitem.addPoints(range(0, hotelling.shape[0]), hotelling,
                           hoverable=True, hoverSymbol='star',
                           data=hotelling.index)
        plotitem.sigClicked.connect(self.scoreplot_onclick)
        widget.addItem(plotitem)
        return widget

    def score_plot(self):
        tab = QWidget()
        if self.pca.dimensions < 2:
            return tab

        score_plot_layout = QVBoxLayout()
        grid_layout = QGridLayout()

        grid_layout.setColumnStretch(1, 1)
        grid_layout.setColumnStretch(2, 4)
        grid_layout.addWidget(QLabel('X'),0,0)
        x_select = QComboBox()
        x_select.addItem('T1')
        grid_layout.addWidget(x_select,0,1)
        grid_layout.addWidget(QLabel('Y'),1,0)
        y_select = QComboBox()
        y_select.addItem('T2')
        grid_layout.addWidget(y_select,1,1)
        score_plot_layout.addLayout(grid_layout)

        plot_graph = pg.PlotWidget()
        t1 = self.pca.get_scores(1)
        t2 = self.pca.get_scores(2)
        score_plot_item = pg.ScatterPlotItem(
            pen=pg.mkPen(width=5, color='w'), symbol='o', size=1)
        score_plot_item.addPoints(t1, t2, hoverable=True,
                                       hoverSymbol='star',
                                       data=self.pca.get_labels())
        score_plot_item.sigClicked.connect(self.scoreplot_onclick)

        plot_graph.addItem(score_plot_item)
        score_plot_layout.addWidget(plot_graph)
        tab.setLayout(score_plot_layout)
        return tab

    def scoreplot_onclick(self, plot, points):
        del plot # Unused
        tab = QWidget()
        point = points[0].data()
        contribution_plot_layout = QVBoxLayout()
        contribution_plot_layout.addWidget(QLabel(
            f"Contribution plot for data point {point}"))
        contribution = []
        for i, p in enumerate(self.pca.get_components()):
            if i == 0:
                contribution = p*self.pca.get_scores(i)[point]
            else:
                contribution = contribution + p*self.pca.get_scores(i)[point]
        loading_plot = pg.PlotWidget()
        graph = pg.BarGraphItem(x=range(1, len(contribution)+1),
                                height=contribution, width=0.6)
        loading_plot.addItem(graph)
        contribution_plot_layout.addWidget(loading_plot)
        tab.setLayout(contribution_plot_layout)
        self.tabs.addTab(tab, str(point))
        self.tabs.setCurrentIndex(self.tabs.indexOf(tab))

    def btn_explore_data_tab_onclick(self, _):
        widget = pg.ScatterPlotWidget()
        fields = []
        for c in self.user_df.columns:
            fields.append((c,{}))
        widget.setFields(fields)
        widget.setData(self.user_df.to_records())
        self.tabs.addTab(widget, "Explore Data")
        self.tabs.setCurrentIndex(self.tabs.indexOf(widget))

    def btn_add_plot_onclick(self, _):
        self.dialog = PlotSelectWindow()
        self.dialog.show()
        self.dialog.add_plot_signal.connect(self.btn_add_plot_submit)

    def btn_add_plot_submit(self, plot_type):
        widget = None
        match plot_type:
            case "Loading Plot":
                widget = self.loading_plot()
            case "Score Plot":
                widget = self.score_plot()
            case "SPE":
                widget = self.spe_plot()
            case "Hotelling's T2":
                widget = self.hotelling_plot()
            case _:
                return
        self.tabs.addTab(widget, plot_type)
        self.tabs.setCurrentIndex(self.tabs.indexOf(widget))

    def btn_pca_onclick(self, _):
        self.dialog = PCAConfigWindow()
        self.dialog.show()
        self.dialog.pca_configured_signal.connect(self.pca_model)

    def pca_model(self, n_dim):
        if self.tabs.indexOf(self.pca.pca_widget) >= 0:
            self.tabs.removeTab(self.tabs.indexOf(self.pca.pca_widget))
        self.pca.setup(self.user_df, n_dim)
        self.btn_pca.setStatusTip("Modify PCA Model")

        self.pca.pca_widget = QWidget()
        model_desc_layout = QGridLayout()
        model_desc_layout.setColumnStretch(1, 1)
        model_desc_layout.setColumnStretch(2, 4)
        model_desc_layout.addWidget(QLabel("Model Status:"),0,0)
        model_desc_layout.addWidget(self.model_status_label,0,1)

        i = 1
        for p in self.pca.get_components():
            model_desc_layout.addWidget(QLabel(
                f"Loading Vector {i}:"), i, 0)
            model_desc_layout.addWidget(QLabel(str(p)),i,1)
            model_desc_layout.setRowMinimumHeight(i-1,10)
            model_desc_layout.setRowStretch(i-1,1)
            i = i + 1

        j = 1
        i = i + 1
        r2_array = self.pca.get_r2()
        for r2 in r2_array:
            model_desc_layout.addWidget(QLabel(
                f"Variance Explained by component {j}:"), i, 0)
            model_desc_layout.addWidget(QLabel(str(r2)),i,1)
            model_desc_layout.setRowMinimumHeight(i-1,10)
            model_desc_layout.setRowStretch(i-1,1)
            i = i + 1
            j = j+ 1

        model_desc_layout.addWidget(QLabel(
            "Total Variance Explained:"), i, 0)
        model_desc_layout.addWidget(QLabel(
            str(np.sum(r2_array))), i, 1)
        model_desc_layout.setRowMinimumHeight(i - 1, 10)
        model_desc_layout.setRowStretch(i - 1, 1)

        model_desc_layout.setRowStretch(i + 1, 10)
        self.pca.pca_widget.setLayout(model_desc_layout)
        self.tabs.addTab(self.pca.pca_widget,"PCA Model")
        self.tabs.setCurrentIndex(self.tabs.indexOf(self.pca.pca_widget))

    def btn_pls_onclick(self, _):
        # TODO: Implement PLS
        pass

    def btn_import_onclick(self, _):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Import Data", "", "All Files (*);;Python Files (*.py)")
        if file_name:
            self.user_df_raw = pd.read_csv(file_name, index_col=0)
            self.user_df = ((self.user_df_raw - self.user_df_raw.mean())
                            / self.user_df_raw.std())  #TODO: Handle text data
            self.table_model = DataFrameModel(self.user_df_raw)
            self.tableview.setModel(self.table_model)

            self.btn_explore_data.setDisabled(False)
            self.btn_pca.setDisabled(False)


def main():
    app = QApplication([])
    mainwindow = MainWindow()
    mainwindow.show()
    app.exec()


if __name__ == "__main__":
    main()
