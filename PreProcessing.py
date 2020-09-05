import numpy as np
from scipy.io import wavfile
from numpy.fft import fft
import glob
import pickle
import json
import math
from decimal import Decimal
class PreProcessor:
    def BatchProcess(self, folder):
        np.seterr('print')
        preprocessed = list()
        counter = 0
        for file in glob.glob(folder + "*.wav"):
            print("Processing {0}".format(file))
            data = self.LoadFile(file)

            outfile = open("./PreprocessedData/Data_{0}.json".format(counter), "w")
            jsonStr = json.dump(data, outfile)
            outfile.close()
            counter += 1

    def LoadFile(self, filename):
        sr, data = wavfile.read(filename)
        monoSamples = np.zeros(len(data), dtype='double')
        count = 0
        for x in data:
            monoSamples[count] = x[0]
            count += 1
        print(len(monoSamples))
        # fft size is gonna be 1024, what about hop size, what actually is that...
        numTransforms = int(len(monoSamples) / 1024)
        magPhases = list()
        mags = list()
        phases = list()
        for i in range(numTransforms):
            startSample = i * 1024
            endSample = startSample + 1024
            transform = fft(monoSamples[startSample : endSample])
            counter = 0
            for bin in transform:
                mag = np.abs(bin)
                phase = np.angle(bin, deg=True)
                mags.append(np.double(mag))
                phases.append(phase)
        mags = np.asarray(mags)
        phases = np.asarray(phases)
        counter = 0
        for mag in mags:
            if(mags.max() != 0):
                mags[counter] = mag / mags.max()
            else:
                mags[counter] = 0
            counter += 1
        counter = 0
        phases = [x / 180 for x in phases]
        magPhases.append([mags.tolist(), phases])
        return magPhases




processor = PreProcessor()
processor.BatchProcess("./KickDrumsEqualLength/")