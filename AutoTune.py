import numpy as np
from scipy.signal import decimate
from scipy.interpolate import interp1d
from scipy.signal import resample
import sounddevice as sd
# working version detection

class AutoTune():

    def __init__(self, wav, sensitivity=.02, min_freq=50.0, max_freq=2500.0,
                 preserve_orig_on_fail=False ):

        # L inversly proportional to freq
        min_L = int(44100.0/(float(max_freq)*8))
        max_L = int(44100.0/(float(min_freq)*8))

        assert len(wav) > 8*220

        # original audio
        self.wav = np.array(wav)

        # 1/8 decimation
        self.down_wav = decimate(self.wav, q=8)
        print(self.wav.shape)
        # declare tables
        self.Ei_Table = np.zeros((len(self.down_wav), 111), dtype=float)
        self.Hi_Table = np.zeros((len(self.down_wav), 111), dtype=float)
        self.subTable = np.zeros((len(self.down_wav), 111), dtype=float)
        self.real_freq = np.zeros(len(self.wav), dtype=float)

        self.idxTable = []

        # first value of e table
        for L in range(min_L, max_L):
            self.Ei_Table[220,L] = sum(np.square(self.down_wav[221-2*L:221]))
            self.Hi_Table[220,L] = sum(np.multiply(self.down_wav[221-2*L:221-L], self.down_wav[221-L:221]))

        # loop to initialize tables
        for i in range(221, len(self.down_wav)):

            for L in range(min_L, max_L):
                self.Ei_Table[i, L] = self.Ei_Table[i-1, L] + self.down_wav[i]**2 - self.down_wav[i-2*L]**2
                self.Hi_Table[i, L] = self.Hi_Table[i-1, L] + self.down_wav[i]*self.down_wav[i-L] - self.down_wav[i-L]*self.down_wav[i-2*L]
                self.subTable[i, L] = self.Ei_Table[i, L] - 2*self.Hi_Table[i, L]


                if self.subTable[i, L] < sensitivity*self.Ei_Table[i, L] and L > min_L and L < max_L:
                    value, freq = self.get_real_freq(i, L)
                    self.real_freq[i*8:(i+1)*8] = freq
                    if not i % 1000:
                        print('i:',i*8,'bigL:',8*L,'freq:',freq)
                    break

                if L == max_L-1:
                    if preserve_orig_on_fail:
                        continue
                    else:
                        self.real_freq[i*8:(i+1)*8] = self.real_freq[i*8-1]


        print('done!')


    def get_real_freq(self, down_indx, down_f_est):

        # convert to real signal indecies
        real_indx = 8*down_indx
        real_f_est = 8*down_f_est

        # create Ls index to calculate E and H for
        real_line = np.linspace(real_f_est-9,real_f_est+9,19, dtype=int)
        real_mins = np.zeros(len(real_line), dtype=float)

        for L in real_line:
            a = np.array(self.wav[real_indx-2*L+1:real_indx-L+1], dtype=np.float64)
            b = np.array(self.wav[real_indx-L+1:real_indx+1], dtype=np.float64)
            E = np.sum(np.square(a)) + np.sum(np.square(b))
            H = sum(a*b)

            val = (E-2*H)/E
            real_mins[L-real_line[0]] = val

        interpolation_line = np.linspace(real_line[0],real_line[-1],100, endpoint=True)
        int_obj = interp1d(real_line, real_mins, kind='cubic')
        interpolated = int_obj(interpolation_line)
        final_lag = interpolation_line[np.argmin(interpolated)]
        value = np.min(interpolated)

        return(value, (1/(final_lag/44100)))


    def build_output(self, mod_type='linear', **kwargs):


        # declare array to hold output signal (member)
        self.output = np.zeros_like(self.wav)

        # call output frequency builders
        if mod_type=='identity': self.desired = self.real_freq
        elif mod_type=='linear': self.desired = self.__linear__(**kwargs)
        elif mod_type=='vibrato': self.desired = self.__vibrato__(**kwargs)
        elif mod_type=='vibrato build': self.desired = self.__vibrato__build__(**kwargs)
        elif mod_type=='vibratoflat': self.desired = self.__vibrato_flat__(**kwargs)
        elif mod_type=='constant': self.desired = self.__constant__(**kwargs)
        elif mod_type=='note': self.desired = self.__note__()

        elif mod_type=='custom': self.desired = self.__custom__(**kwargs)
        else: raise Exception('No \''+str(mod_type)+'\' modulation type.')

        i_pnt = 21000
        o_pnt = 21000

        while i_pnt < (len(self.desired)-500) and o_pnt < (len(self.desired)-500):

            cur_freq = self.real_freq[i_pnt]
            des_freq = self.desired[o_pnt]

            if cur_freq < 70 or cur_freq > 5000:
                self.output[o_pnt:o_pnt+100] = self.wav[i_pnt:i_pnt+100]
                i_pnt += 100
                o_pnt += 100
                continue

            i_sz = int(np.rint(44100/cur_freq))
            o_sz = int(np.rint(i_sz*(cur_freq/des_freq)))

            out_slice = resample(self.wav[i_pnt:i_pnt+i_sz+1], o_sz)
            #print(len(out_slice))
            self.output[o_pnt:o_pnt+o_sz] = out_slice

            i_pnt += int(i_sz)
            o_pnt += int(o_sz)

            if abs(i_pnt - o_pnt) > o_sz:
                if i_pnt < o_pnt:
                    o_pnt -= o_sz
                if i_pnt > o_pnt:
                    self.output[o_pnt:o_pnt+o_sz] = out_slice
                    o_pnt += o_sz

    def __linear__(self, start_freq=70, end_freq=300):
        return np.linspace(start_freq,end_freq,len(self.output),endpoint=True)

    def __vibrato__(self, vib_freq=10, amp=8):
        return (amp*np.sin(2*np.pi*(vib_freq)*np.arange(len(self.wav))/44100))+self.real_freq

    def __vibrato__build__(self, vib_freq=10, amp=8):
        return (amp*np.sin(2*np.pi*(vib_freq)*np.arange(len(self.wav))/44100))\
            *np.linspace(0,1,len(self.output))\
            +self.real_freq

    def __vibrato_flat__(self, fund_freq=120, vib_freq=10, amp=8):
        return (amp*np.sin(2*np.pi*(vib_freq)*np.arange(len(self.wav))/44100))+fund_freq

    def __constant__(self, freq=120):
        return(np.ones_like(self.output)*freq)

    def __note__(self):
        notes = np.array([55.00,58.2,61.74,
            65.41,69.30,73.42,77.78,82.41,87.31,92.50,98.00,103.83,110.00,116.54,123.47,130.81,
            138.59,146.83,155.56,164.81,174.61,185.00,196.00,207.65,220.00,233.08,246.94,261.63,
            277.18,293.66,311.13,329.63,349.23,369.99,392.00,415.30,440.00,466.16,493.88,523.25,
            554.37,587.33,622.25,659.25,698.46,739.99,783.99,830.61,880.00,932.33,987.77,1046.50])

        size = len(self.output)
        desired = np.zeros_like(self.output)
        for t in range(size):
            dist = abs(notes-self.real_freq[t])
            desired[t] = notes[np.argmin(dist)]

        return desired

    def __custom__(self, desired):
        if len(desired) != len(self.output):
            raise Exception('desired array of length ', str(len(desired)), ', expected ', str(len(self.output)))
        return desired

    def play(self):
        sd.play(self.output)
