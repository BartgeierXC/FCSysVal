import sys

import numpy as np
import os
import platform
import subprocess
import pandas as pd

try:
    from OMPython import ModelicaSystem, OMCSessionZMQ
except ModuleNotFoundError:
    # Only needed to run OpenModelica, not to read results
    ModelicaSystem, OMCSessionZMQ = object, None
from sklearn import preprocessing
from scipy.interpolate import LinearNDInterpolator, griddata

def import_mat_result(mat_path: str, variables=None, timestep=1, rtol=1e-3) -> pd.DataFrame:
    """Import an OpenModelica simulation result .mat file as a pandas DataFrame.

    Uses OMPython's OMCSessionZMQ to call readSimulationResultVars and
    readSimulationResult directly on the .mat file – no model build required.

    :param mat_path: str
                Absolute or relative path to the OpenModelica result .mat file.
    :param variables: list, optional
                List of variable names to import. If None, all variables will be imported.
    :param timestep: int or float, optional
                Sampling interval for filtering the results. Default is 1.
    :param rtol: float, optional
                Relative tolerance for matching result times to the timestep grid. The absolute
                tolerance is ``timestep * rtol``. For each grid point, the nearest matching
                result is retained and its time is set to the exact grid value. Default is 1e-3.
    :return: pd.DataFrame
                DataFrame containing simulation results with variable columns.
    """
    if "time" in (variables or ()) or variables is None:
        if not np.isfinite(timestep) or timestep <= 0:
            raise ValueError(f"timestep must be a finite number greater than 0, got {timestep}")
        if not np.isfinite(rtol) or rtol < 0:
            raise ValueError(f"rtol must be a finite number greater than or equal to 0, got {rtol}")

    try:
        return _import_mat_result_direct(mat_path, variables, timestep, rtol)
    except _UnsupportedMatFormat:
        return _import_mat_result_omc(mat_path, variables, timestep, rtol)


class _UnsupportedMatFormat(Exception):
    pass


def _select_grid_rows(times, timestep, rtol):
    """Row indices (ascending) of the sample nearest to each timestep grid point, and the snapped grid times."""
    with np.errstate(invalid="ignore"):
        grid = np.rint(times / timestep)
        distances = np.abs(times - grid * timestep)
        matches = np.isfinite(times) & (distances <= timestep * rtol)
    idx = np.flatnonzero(matches)
    if idx.size == 0:
        return idx, np.empty(0)
    order = np.lexsort((distances[idx], grid[idx]))
    idx_sorted = idx[order]
    _, first = np.unique(grid[idx_sorted], return_index=True)
    rows = np.sort(idx_sorted[first])
    return rows, grid[rows] * timestep


def _read_mat_v4(mat_path):
    from scipy.io import loadmat
    try:
        mat = loadmat(mat_path, variable_names=["Aclass", "name", "dataInfo", "data_1", "data_2"])
    except Exception as exc:
        raise _UnsupportedMatFormat(str(exc)) from exc
    if not all(k in mat for k in ("Aclass", "name", "dataInfo", "data_2")):
        raise _UnsupportedMatFormat("not an OpenModelica result file")
    if str(mat["Aclass"].ravel()[-1]).strip() != "binTrans":
        raise _UnsupportedMatFormat("unsupported Aclass layout")
    rows = [str(r) for r in mat["name"].ravel()]
    width = max(len(r) for r in rows)
    names = ["".join(chars).strip("\x00 ") for chars in zip(*[r.ljust(width) for r in rows])]
    data_1 = mat["data_1"] if "data_1" in mat else np.empty((0, 2))
    return names, mat["dataInfo"], data_1, mat["data_2"]


def _import_mat_result_direct(mat_path, variables, timestep, rtol) -> pd.DataFrame:
    """Read an OpenModelica .mat (v4, binTrans) directly, decimating before copying any data."""
    mat_path = os.path.abspath(mat_path)
    names, info, data_1, data_2 = _read_mat_v4(mat_path)
    index = {n: i for i, n in enumerate(names)}

    if variables is None:
        variables = tuple(names)
    else:
        variables = tuple(variables)
        missing_vars = [v for v in variables if v not in index]
        if missing_vars:
            raise ValueError(
                f"The following variables not found in {mat_path}:\n"
                f"  Missing: {missing_vars}\n"
                f"  Available: {names}"
            )
    if not variables:
        raise RuntimeError(f"No variables found in {mat_path}")

    n_samples = data_2.shape[1]
    rows = slice(None)
    grid_times = None
    if "time" in variables:
        rows, grid_times = _select_grid_rows(data_2[0].astype(float), timestep, rtol)

    columns = {}
    for v in variables:
        table, col = int(info[0, index[v]]), int(info[1, index[v]])
        sign = -1.0 if col < 0 else 1.0
        if table == 1:
            n_out = n_samples if isinstance(rows, slice) else len(rows)
            columns[v] = np.full(n_out, sign * data_1[abs(col) - 1, 0])
        else:
            columns[v] = sign * data_2[0 if table == 0 else abs(col) - 1][rows]

    df = pd.DataFrame(columns, columns=list(variables))
    if grid_times is not None:
        df["time"] = grid_times
    return df


def _import_mat_result_omc(mat_path: str, variables, timestep, rtol) -> pd.DataFrame:
    mat_path = os.path.abspath(mat_path).replace("\\", "/")
    omc = OMCSessionZMQ()
    available_vars = omc.sendExpression(f'readSimulationResultVars("{mat_path}")')

    if variables is None:
        variables = available_vars
    else:
        variables = tuple(variables)
        # Validate that all requested variables exist
        missing_vars = [v for v in variables if v not in available_vars]
        if missing_vars:
            raise ValueError(
                f"The following variables not found in {mat_path}:\n"
                f"  Missing: {missing_vars}\n"
                f"  Available: {available_vars}"
            )

    if not variables:
        raise RuntimeError(f"No variables found in {mat_path}")

    var_expr = ",".join(variables)
    data = omc.sendExpression(f'readSimulationResult("{mat_path}", {{{var_expr}}})')
    omc.sendExpression("closeSimulationResultFile()")

    df = pd.DataFrame(np.array(data).T, columns=list(variables))

    if "time" in variables:
        if not np.isfinite(timestep) or timestep <= 0:
            raise ValueError(f"timestep must be a finite number greater than 0, got {timestep}")
        if not np.isfinite(rtol) or rtol < 0:
            raise ValueError(f"rtol must be a finite number greater than or equal to 0, got {rtol}")

        times = pd.to_numeric(df["time"], errors="raise").to_numpy(dtype=float)
        selected_rows, selected_grid_times = _select_grid_rows(times, timestep, rtol)
        df = df.iloc[selected_rows].reset_index(drop=True)
        df["time"] = selected_grid_times

    return df

def mat2csv(mat_path: str, csv_path: str = None) -> str:
    """Convert an OpenModelica simulation result .mat file to CSV.

    Uses OMPython's OMCSessionZMQ to call readSimulationResultVars and
    readSimulationResult directly on the .mat file – no model build required.

    :param mat_path: str
                Absolute or relative path to the OpenModelica result .mat file.
    :param csv_path: str, optional
                Output CSV path. Defaults to mat_path with .mat replaced by .csv.
    :return: str
                Absolute path of the written CSV file.
    """

    df = import_mat_result(mat_path)

    if csv_path is None:
        csv_path = os.path.splitext(mat_path)[0] + ".csv"
    csv_path = os.path.abspath(csv_path).replace("\\", "/")

    df.to_csv(csv_path, index=False)
    return csv_path


class MSWrapper(ModelicaSystem):
    # to simulate or re-simulate model
    def simulate(self, resultfile=None, simflags=None, logfile=None):  # 11
        """
        This method simulates model according to the simulation options.
        usage
        simulate()
        simulate(resultfile="a.mat")
        simulate(simflags="-noEventEmit -noRestart -override=e=0.3,g=10) set runtime simulation flags
        simulate(logfile="file.log")
        """
        if (resultfile is None):
            r = ""
            self.resultfile = "".join([self.modelName, "_res.mat"])
        else:
            r = " -r=" + resultfile
            self.resultfile = resultfile

        # allow runtime simulation flags from user input
        if (simflags is None):
            simflags = ""
        else:
            simflags = " " + simflags;

        if (self.overridevariables or self.simoptionsoverride):
            tmpdict = self.overridevariables.copy()
            tmpdict.update(self.simoptionsoverride)
            values1 = ','.join("%s=%s" % (key, val) for (key, val) in list(tmpdict.items()))
            override = " -override=" + values1
        else:
            override = ""

        if (self.inputFlag):  # if model has input quantities
            for i in self.inputlist:
                val = self.inputlist[i]
                if (val == None):
                    val = [(float(self.simulateOptions["startTime"]), 0.0),
                           (float(self.simulateOptions["stopTime"]), 0.0)]
                    self.inputlist[i] = [(float(self.simulateOptions["startTime"]), 0.0),
                                         (float(self.simulateOptions["stopTime"]), 0.0)]
                if float(self.simulateOptions["startTime"]) != val[0][0]:
                    print("!!! startTime not matched for Input ", i)
                    return
                if float(self.simulateOptions["stopTime"]) != val[-1][0]:
                    print("!!! stopTime not matched for Input ", i)
                    return
                if val[0][0] < float(self.simulateOptions["startTime"]):
                    print('Input time value is less than simulation startTime for inputs', i)
                    return
            self.__simInput()  # create csv file
            csvinput = " -csvInput=" + self.csvFile
        else:
            csvinput = ""

        if (platform.system() == "Windows"):
            getExeFile = os.path.join(os.getcwd(), '{}.{}'.format(self.modelName, "exe")).replace("\\", "/")
        else:
            getExeFile = os.path.join(os.getcwd(), self.modelName).replace("\\", "/")

        if (os.path.exists(getExeFile)):
            cmd = getExeFile + override + csvinput + r + simflags
            # print(cmd)
            if (platform.system() == "Windows"):
                omhome = os.path.join(os.environ.get("OPENMODELICAHOME"))
                dllPath = os.path.join(omhome, "bin").replace("\\", "/") + os.pathsep + os.path.join(omhome,
                                                                                                     "lib/omc").replace(
                    "\\", "/") + os.pathsep + os.path.join(omhome, "lib/omc/cpp").replace("\\",
                                                                                          "/") + os.pathsep + os.path.join(
                    omhome, "lib/omc/omsicpp").replace("\\", "/")
                my_env = os.environ.copy()
                my_env["PATH"] = dllPath + os.pathsep + my_env["PATH"]

                if logfile:
                    with open("log_" + logfile, 'w') as my_stdout, open("err_" + logfile, 'w') as my_stderr:
                        p = subprocess.Popen(cmd, env=my_env, stdout=my_stdout, stderr=my_stderr)
                else:
                    p = subprocess.Popen(cmd, env=my_env)

                p.wait()
                p.terminate()

            else:
                os.system(cmd)
            self.simulationFlag = True

        else:
            raise Exception("Error: application file not generated yet")


class OMTabParser:
    """Parser for OpenModelica tables

    """

    def __parse_path(self, path_to_file):
        omc = OMCSessionZMQ()
        path_to_file = str(path_to_file)
        if os.path.isabs(path_to_file) or sys.platform == 'linux':
            return path_to_file.replace("\\", "/")
        else:
            if type(path_to_file) is str:
                path_to_file = path_to_file.lstrip('/').lstrip('\\')
            return os.path.join(os.path.dirname(omc.sendExpression("getInstallationDirectoryPath()")),
                                path_to_file).replace("\\", "/")

    def conv2DLuT(self, x_cols, z_col, path_to_file, dim=None, normalize=True, interp=True):
        path = self.__parse_path(path_to_file)
        data = pd.read_csv(path)

        x_min = min(data[x_cols[0]].values)
        x_max = max(data[x_cols[0]].values)
        y_min = min(data[x_cols[1]].values)
        y_max = max(data[x_cols[1]].values)

        # Normalization
        if normalize:
            z = data.values
            min_max_scaler = preprocessing.MinMaxScaler()
            z_scaled = min_max_scaler.fit_transform(z)
            data = pd.DataFrame(z_scaled, columns=data.columns)
            x_min = 0
            x_max = 1
            y_min = 0
            y_max = 1

        if interp:
            # Instantiate interpolator, hier stimmt was nicht!
            interp = LinearNDInterpolator(list(zip(data[x_cols[0]].values, data[x_cols[1]].values)), data[z_col].values)
            # interp = lambda xi, yi: griddata(
            #     list(zip(data[x_cols[0]].values, data[x_cols[1]].values)),
            #     data[z_col].values,
            #     (xi, yi),
            #     method="linear"
            # )

            if dim is None:
                dim = (50, 50)

            # Create target plane
            x = np.linspace(x_min, x_max, num=dim[0])
            y = np.linspace(y_min, y_max, num=dim[1])
            X, Y = np.meshgrid(x, y)

            return pd.DataFrame(interp(X, Y), y, x)
        else:
            lut_df = pd.DataFrame(None, np.unique(data[x_cols[1]].values), np.unique(data[x_cols[0]].values))
            for index, row in data.iterrows():
                lut_df[row[x_cols[0]]][row[x_cols[1]]] = row[z_col]
            return lut_df

    def parse1DT(data, time=None, timedelta=50):
        """
        Parse data from a list to a String suitable for a Modelica.Blocks.Tables.CombiTable.

        Example String: [0, 0; 10, 1; 20, 2; 30, 7]

        :param data: list
                    List containing data to parse.
        :param time: list
                    List containing time values for data. If not specified, time values are generated from timedelta.
        :param timedelta: float
                    Time difference between two consecutive values in data.
        :return: String
                String containing Modelica table in the form [time, data; time, data; ...]
        """
        if time is None:
            time = [i * timedelta for i in range(len(data))]
        return "[" + "; ".join([str(time[i]) + ", " + str(data[i]) for i in range(len(data))]) + "]"

    def parseCombiTab(self, data, name, withCol=True):
        """Parse a pandas.DataFrame to String containing a Modelica.Blocks.Tables.CombiTable instance.

        :param data: pandas.DataFrame
                    Source data to parse.
        :param name: String
                    Name of Modelica table.
        :return: String
                String containing Modelica table.
        """
        rows, cols = data.shape

        if withCol:
            table = f"float {name}({rows + 1},{cols + 1})\n"
            # Add column names as the first row
            table += "0.0 " + " ".join(str(col) for col in data.columns) + "\n"
        else:
            table = f"float {name}({rows},{cols + 1})\n"

        # Iterate over DataFrame rows and add values
        for index, row in data.iterrows():
            table += f"{index} " + " ".join(str(val) for val in row) + "\n"

        return table

    def exportLuTtoFile(self, data, name, path_to_file, withCol=True, verbose=False):
        """Use parseCombiTab method to create a Modelica.Blocks.Tables.CombiTable2D instance and store it in .txt file.

        :param data: pandas.DataFrame
                    Source data to parse.
        :param name: String
                    Name of Modelica table.
        :param name: String
                    Absolute or relative path (to Modelica installation) pointing to table file.
        """
        path = self.__parse_path(path_to_file)

        with open(path, "a+") as fp:
            fp.seek(0)  # Move cursor to the beginning of the file
            lines = fp.readlines()

            found = False
            t_flag = False
            new_lines = []
            table_content = self.parseCombiTab(data, name, withCol)

            for line in lines:
                if line.startswith(f"float {name}"):
                    found = True
                    t_flag = True
                    new_lines.append(table_content)
                elif t_flag:
                    if line.startswith(f"float"):
                        t_flag = False
                        new_lines.append('\n' + line)
                else:
                    new_lines.append(line)

            if not found:
                if not new_lines:
                    new_lines.append("#1")
                new_lines.append('\n' + table_content)

            fp.seek(0)  # Move cursor to the beginning of the file
            fp.truncate()  # Clear the file content
            fp.writelines(new_lines)  # Write new content
            if verbose:
                print(new_lines)


class OMWrapper:
    """Interface for OMPython

    """

    def __init__(self, path, model):
        """ Intitialize OMWrapper with path to modelica model and model name.

        :param path: String
                    Path to model (.mo-file), can be relative from OpenModelica installation folder.
        :param model:
                    Model name.
        """

        if type(path) is str and os.name == 'nt':
            path = path.lstrip('/').lstrip('\\')
        self.omc = OMCSessionZMQ()

        if os.name == 'nt' and not os.path.isabs(path):
            path = os.path.join(os.path.dirname(self.omc.sendExpression("getInstallationDirectoryPath()")),
                                path).replace("\\", "/")

        if not os.getcwd().endswith('results'):
            res_path = os.path.join(os.getcwd(), 'results').replace("\\", "/")
            if not os.path.exists(res_path):
                os.mkdir(res_path)
            os.chdir(res_path)
        self.mod = MSWrapper(path, model, ["Modelica"], commandLineOptions='-d=openmp')

        self.mod.buildModel()

    def setSimOpt(self, stopTime, startTime=0, stepSize=1, tolerance=1e-4):
        """

        :param stopTime:
        :param startTime:
        :param stepSize:
        :param tolerance:
        :return:
        """
        self.mod.setSimulationOptions(
            ["startTime=" + str(startTime), "stopTime=" + str(stopTime), 'stepSize=' + str(stepSize),
             'tolerance=' + str(tolerance), 'solver=dassl'])

    def getResultNames(self, resultPath=None) -> np.array:
        """

        :param resultPath: String
                            Path to external .mat result file.
        :return:
        """
        if resultPath:
            resultPath = resultPath.lstrip('/').lstrip('\\')
            self.omc = OMCSessionZMQ()
            if not os.path.isabs(resultPath):
                resultPath = os.path.join(os.path.dirname(self.omc.sendExpression("getInstallationDirectoryPath()")),
                                          resultPath).replace("\\", "/")
        # return self.mod.getSolutions(resultfile= os.path.join(self.res_path, self.mod.resultfile))
        return self.mod.getSolutions(resultfile=resultPath)

    def getResults(self, targetValues, withTime=False, resultPath=None, groupBy=1) -> pd.DataFrame:
        """

        :param targetValues: list
                            List with target values.
        :param withTime: bool
                        True if time should be a column of the result.
        :param resultPath: String
                            Path to external .mat result file, has to be the same model.
        :param groupBy: int
                        Group results by this timespan.
        :return:
        """

        if resultPath:
            # resultPath = resultPath.lstrip('/').lstrip('\\')
            self.omc = OMCSessionZMQ()
            resultPath = resultPath.replace("\\", "/")
            if not os.path.isabs(resultPath):
                resultPath = os.path.join(os.path.dirname(self.omc.sendExpression("getInstallationDirectoryPath()")),
                                          resultPath).replace("\\", "/")

        targetValues = ['time'] + targetValues

        data = self.mod.getSolutions(targetValues, resultfile=resultPath)
        data = np.array(data).transpose()

        # if groupBy > 0, filter out every row where 'time' is not divisible by groupBy
        if groupBy > 0:
            data = np.array([row for row in data if row[0] % groupBy == 0])

        if not withTime:
            targetValues = targetValues[1:]
            data = data[:, 1:]

        return pd.DataFrame(data, columns=targetValues)

    def simulate(self, override_params=None, resultfile=None, simflags="-cpu", verbose=False):
        """

        :param override_params: dict
                    Pass new parameters to re-run simulation without compile.
        :param resultfile: String
                    Pass path to resultfile in new location if necessary.
        :param simflags: String
                    Add simflags to simulation command as in simflags="-noEventEmit -noRestart"
        :param verbose: bool
                    Print out all simulation messages to stdout if verbose is True, redirect to logging-files otherwise.
        :return: None
        """
        if override_params:
            self.mod.overridevariables = override_params
        else:
            self.mod.overridevariables = {}

        if verbose:
            self.mod.simulate(resultfile=resultfile, simflags=simflags)
        else:
            self.mod.simulate(resultfile=resultfile, simflags=simflags, logfile=self.mod.modelName + '.log')

    def checkSimSuccess(self, verbose=False) -> bool:
        """ Check if simulation ended successfully

        :param verbose: bool
        :return: bool
                Returns None if simulation failed completely, False if simulation aborted too early and True if simulation ended successfully.
        """
        if len(self.mod.getSolutions()) == 0 or len(self.mod.getSolutions('time')[0]) == 0:
            if verbose:
                print("Simulation failed")
            return None
        elif self.mod.getSolutions('time')[0][-1] < float(self.mod.getSimulationOptions("stopTime")[0]):
            if verbose:
                print("Simulation aborted at time=" + str(self.mod.getSolutions('time')[0][-1]))
            return False
        else:
            if verbose:
                print("Simulation successful")
            return True
