import csv
import logging
import os
import shutil
import numpy as np
from pyfanova.fanova import Fanova

class FanovaFromCSV(Fanova):

    def __init__(self, csv_file, header=False,**kwargs):

        self._scenario_dir = "tmp_smac_files"

        if not os.path.isdir(self._scenario_dir):
            os.mkdir(self._scenario_dir)

        self.header=header

        X, y = self._read_csv_file(csv_file,header=header)
        self._write_instances_file()
        self._write_runs_and_results_file(y)
        self._write_param_file()
        self._write_paramstrings_file(X)
        self._write_scenario_file()

        logging.debug("Write temporary smac files in " + self._scenario_dir)
        super(FanovaFromCSV, self).__init__(self._scenario_dir)

    def __del__(self):
        shutil.rmtree(self._scenario_dir)
        super(FanovaFromCSV, self).__del__()

    def _write_scenario_file(self):

        fh = open(os.path.join(self._scenario_dir, "scenario.txt"), "w")

        fh.write("algo = .\n")
        fh.write("execdir = .\n")
        fh.write("deterministic = 0\n")
        fh.write("run_obj = qual\n")
        fh.write("overall_obj = mean\n")
        fh.write("cutoff_time = 1e100\n")
        fh.write("cutoff_length = 0\n")
        fh.write("tunerTimeout = 0\n")
        fh.write("paramfile = .\n")
        fh.write("instance_file = .\n")
        fh.write("test_instance_file = .\n")

        fh.close()

    def _write_instances_file(self):

        fh = open(os.path.join(self._scenario_dir, "instances.txt"), "w")
        fh.write(".")
        fh.close()

    def _write_runs_and_results_file(self, values):

        fh = open(os.path.join(self._scenario_dir, "runs_and_results.csv"), "wb")
        writer = csv.writer(fh)
        writer.writerow(("Run Number", "Run History Configuration ID", "Instance ID", "Response Value (y)", "Censored?", "Cutoff Time Used",
                                      "Seed", "Runtime", "Run Length", "Run Result Code", "Run Quality", "SMAC Iteration", "SMAC Cumulative Runtime", "Run Result"))

        for i in range(0, len(values)):
            line = (i, i, 1, 0, 0, 0, 1, 0, 0, 0, values[i], 0, 0, "SAT")
            writer.writerow(line)

        fh.close()

    def _write_param_file(self):

        fh = open(os.path.join(self._scenario_dir, "param-file.txt"), "w")
        for i in range(0, self._num_of_params):
            if self.header:
                param_string = self.parameter_names[i] + " " + str(self._bounds[i]) + " " + "[" + str(self._defaults[i]) + "]\n"
            else:
                param_string = "X" + str(i) + " " + str(self._bounds[i]) + " " + "[" + str(self._defaults[i]) + "]\n"
            logging.debug(param_string)
            fh.write(param_string)

        fh.close()

    def _write_paramstrings_file(self, params):

        fh = open(os.path.join(self._scenario_dir, "paramstrings.txt"), "w")
        for i in range(0, params.shape[0]):
            line = str(i) + ": "
            for j in range(0, params.shape[1]):
                if self.header:
                    line = line + self.parameter_names[j] + "='" + str(params[i][j]) + "', "
                else:
                    line = line + "X" + str(j) + "='" + str(params[i][j]) + "', "
            #remove the last comma and whitespace from the string again
            line = line[:-2]
            line = line + '\n'

            fh.write(line)
        fh.close()

    def _read_csv_file(self, filename,header=False):

        fh = open(filename, "r")
        if header:
            reader = csv.DictReader(fh)
            self.parameter_names=reader.fieldnames
        else:
            reader = csv.reader(fh)
            self.parameter_names=None            
        
        #Count how many data points are in the csv file
        number_of_points = 0
        for line in reader: 
            number_of_points += 1

        if header:
            fh.seek(0)
            reader = csv.DictReader(fh)
        else:
            fh.seek(0)

        #Count the dimension of the the data points
        line = fh.readline()
        s = line.split(',')
        self._num_of_params = len(s) - 1

        logging.debug("number of parameters: " + str(self._num_of_params))

        X = np.zeros([number_of_points, self._num_of_params])
        y = np.zeros([number_of_points])

        if header:
            fh.seek(0)
            reader = csv.DictReader(fh)
        else:
            fh.seek(0)

        rownum = 0
        for line in reader:
            for param in range(0, self._num_of_params):
                if header:
                    X[rownum][param] = line[self.parameter_names[param]]
                else:
                    X[rownum][param] = line[param]
            if header:
                y[rownum] = line[self.parameter_names[-1]]
            else:
                y[rownum] = line[-1] 

            rownum += 1

        fh.close()

        self._bounds = []
        self._defaults = []
        for i in range(0, self._num_of_params):
            #Take min and max value as bounds for smac parameter file
            self._bounds.append([np.min(X[:, i]), np.max(X[:, i])])
            #Set min value as default value for smac parameter file
            self._defaults.append(np.min(X[:, i]))
        return X, y
