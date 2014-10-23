#!/usr/bin/env python

import math
import numpy as np
import pylab
from scipy import fft

def frequency(f, n):
    return f*math.pow(2, n/12.0)

def minor_2nd(f, just=False):
    if just:
        return f*16.0/15.0
    else:
        return frequency(f, 1)

def major_2nd(f, just=False):
    if just:
        return f*9.0/8.0
    else:
        return frequency(f, 2)

def minor_3rd(f, just=False):
    if just:
        return f*6.0/5.0
    else:
        return frequency(f, 3)

def major_3rd(f, just=False):
    if just:
        return f*5.0/4.0
    else:
        return frequency(f, 4)

def perfect_4th(f, just=False):
    if just:
        return f*4.0/3.0
    else:
        return frequency(f, 5)

def diminished_5th(f, just=False):
    if just:
        return f*7.0/5.0
    else:
        return frequency(f, 6)

def perfect_5th(f, just=False):
    if just:
        return f*3.0/2.0
    else:
        return frequency(f, 7)

def minor_6th(f, just=False):
    if just:
        return f*8.0/5.0
    else:
        return frequency(f, 8)

def major_6th(f, just=False):
    if just:
        return f*5.0/3.0
    else:
        return frequency(f, 9)

def minor_7th(f, just=False):
    if just:
        return f*16.0/9.0
    else:
        return frequency(f, 10)

def major_7th(f, just=False):
    if just:
        return f*15.0/8.0
    else:
        return frequency(f, 11)

def perfect_8th(f, just=False):
    if just:
        return f*2.0/1.0
    else:
        return frequency(f, 12)


def pitch(t, fs):
    ret = 0
    for f in fs:
        ret += np.sin(2*np.pi*f*t)
    return ret
    
def draw(fs):

    max_p = 0
    for f in fs:
        p = 1.0/f
        if p > max_p:
            max_p = p
    
    min = 0
    max = 24*max_p
    step = 0.01*max_p
    times = np.arange(min, max+step, step)

    pylab.plt.plot(times, pitch(times, fs))

    pylab.show()


def plotSpectrum(y,Fs):
    """
    Plots a Single-Sided Amplitude Spectrum of y(t)
    """
    n = len(y) # length of the signal
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(n/2)] # one side frequency range

    Y = fft(y)/n # fft computing and normalization
    Y = Y[range(n/2)]
 
    pylab.plt.plot(frq,abs(Y),'r') # plotting the spectrum
    pylab.plt.xlabel('Freq (Hz)')
    pylab.plt.ylabel('|Y(freq)|')

def major_chord(f, just=False):
    return [f, major_3rd(f,just), perfect_5th(f,just)]

def minor_chord(f, just=False):
    return [f, minor_3rd(f,just), perfect_5th(f,just)]

def diminished_chord(f, just=False):
    return [f, minor_3rd(f,just), diminished_5th(f,just)]

STRING_LENGTH = 50.0/100.0   # 50 cm
PLUCK_LENGTH  = 10.0/100.0   # 10 cm
PLUCK_HEIGHT  = 10.0/1000.0  # 10 mm

def harmonic_weight(mode, fret=0):
    """
    Calculate the harmonic coefficient for a given mode of a plucked
    string. 
    
    Args:
        mode (int): The harmonic mode of interest
        
    Kwargs:
        fret (int): The fret number of a fretted string (open = 0)
        
    Returns:
        The weight of the harmonic mode (float)
        
    >>> harmonic_weight(1)
    0.0074443871862229392
    >>> harmonic_weight(2)
    0.0030113178731807268
    >>> harmonic_weight(3)
    0.0013383634991914342
    >>> harmonic_weight(4)
    0.00046527419913893375
    >>> harmonic_weight(5)
    6.2041331616705041e-20

    """
    if mode < 1:
        return None
    
    
    L = STRING_LENGTH
    d = PLUCK_LENGTH
    h = PLUCK_HEIGHT
    
    C = 2*h*L*L/(np.pi*np.pi*mode*mode*d*(L-d))
    a = np.pi*mode*d/L
    
    return C*np.sin(a)

def harmonic_frequency(mode, string=1, fret=0, stiff=0.008):
    """
    Get the frequency of a specific harmonic mode.
    
    Args:
        mode (int): The mode
        
    Kwargs:
        string (int): The string
        fret (int): The fret
        stiff (float): The stiffness factor
        
    Returns:
        The frequency
        
    >>> harmonic_frequency(1)
    220.00703988736362
    >>> harmonic_frequency(2)
    440.0563163959813
    >>> harmonic_frequency(3)
    660.19005263636006
    >>> harmonic_frequency(4)
    880.45044471565791
    >>> harmonic_frequency(5)
    1100.8796482813189

    """
    base = 220.0
    
    return mode*base*np.sqrt(1 + stiff*stiff*mode*mode)

def harmonic_decay(time, mode=1, gamma=0.01):
    """
    This function describes the decay (in time) of a specific harmonic mode.
    
    Args:
        time (float): The time to evaluate
        
    Kwargs:
        mode (int): The harmonic mode
        gamma (float): The gamma factor
        
    Returns:
        The decay factor at the given time
        
    >>> harmonic_decay(0)
    1.0
    >>> harmonic_decay(100)
    0.36787944117144233
    
    """
    return np.exp(-mode*gamma*time)

def harmonic_waveform(time, string=1, fret=0):
    """
    Calculate the harmonic series of a given note.
    
    Args:
        time (float): The time
        
    Kwargs:
        string (int): The string
        fret (int): The fret
        
    Returns:
        The value of the waveform
        
    >>> harmonic_waveform(0)
    0.012259342757734033
    >>> harmonic_waveform(100)
    -0.00098186543861745156

    """
    ret = 0
    
    for n in range(5):
        mode = n+1
        
        weight = harmonic_weight(mode, fret)
        frequency = harmonic_frequency(mode, string, fret)

        osc = np.cos(2*np.pi*frequency*time)
        decay = harmonic_decay(time, mode)
        
        ret += weight*osc*decay
        
    return ret

class Harmonic(object):

    def __init__(self, frequency, weight=1.0, decay=np.inf):
        self.frequency = frequency
        self.weight = weight
        self.decay = decay

    def waveform(self, time):
        wf = self.weight*np.cos(2*np.pi*self.frequency*time)
        if not np.isinf(self.decay):
            wf *= np.exp(-time/self.decay)
        return wf
        
    def __repr__(self):
        return "{}".format([self.frequency, self.weight, self.decay])


class GuitarString(object):

    def __init__(self, string=6):
        self.string = string
        
        self.length = 90.0/100.0
        self.pluck_distance = 15.0/100.0
        self.pluck_height = 0.5/100.0
        self.damping = 1.7
        self.stiffness = 0.008
        self.max_mode = 25
        
        # Get the frequency of the string, string 5 (A) = 110.0 Hz
        f5 = 110.0
        self.f_open = f5
        if string == 6:
            self.f_open = f5*pow(2.0, -7.0/12.0)
        elif string == 4:
            self.f_open = f5*pow(2.0, 5.0/12.0)
        elif string == 3:
            self.f_open = f5*pow(2.0, 10.0/12.0)
        elif string == 2:
            self.f_open = f5*pow(2.0, 14.0/12.0)
        elif string == 1:
            self.f_open = f5*pow(2.0, 19.0/12.0)
            
        self.velocity = 2.0*self.length*self.f_open
                
    def pluck(self, fret):
#        """
#        
#        >>> E = GuitarString(6)
#        >>> E.pluck(0) # doctest: +NORMALIZE_WHITESPACE
#        [[82.002623958017352, 0.36475626111241594, 100.0], 
#         [164.02099065668395, 0.1579440941563911, 50.0], 
#         [246.07083780082513, 0.08105694691387022, 33.333333333333336], 
#         [328.16789303038161, 0.039486023539097782, 25.0], 
#         [410.32786890485522, 0.014590250444496636, 20.0]]
#        """
        
        harmonics = []
        
        d = self.pluck_distance
        h = self.pluck_height
        b = self.stiffness

        # Find the length of the string to this fret
        L = self.length/pow(2.0, fret/12.0)

        # Need to convert from string/fret to f1
        f1 = self.velocity/(2.0*L)
        
        for i in range(self.max_mode):
            n = i+1
            
            frequency = n*f1*np.sqrt(1 + b*b*n*n)
            weight = 2*h*L*L/(np.pi*np.pi*n*n*d*(L-d))*np.sin(np.pi*n*d/L)
#            weight = 1.0
            decay = np.inf if (self.damping == 0.0) else 1.0/(n*self.damping)
            
            harmonics.append(Harmonic(frequency, weight, decay))

        return harmonics

    def plot(self, fret):
        """
        
        >>> E = GuitarString(6)
        >>> E.plot(0)
        
        """
        harmonics = self.pluck(fret)
        t_max = 32.0/harmonics[0].frequency
        t_step = t_max/1024.0
        t = np.arange(0, t_max, t_step)

        norm = 0
        for h in harmonics:
            norm += h.weight

        y = harmonics[0].waveform(t)
        for h in harmonics:
            if not h == harmonics[0]:
                y += h.waveform(t)
        
#        y /= norm
        print y

        pylab.plt.subplot(3,1,1)
        for h in harmonics:
            pylab.plt.plot(t,h.waveform(t))
        pylab.plt.xlabel('Time')
        pylab.plt.ylabel('Amplitude')
        pylab.plt.subplot(3,1,2)
        pylab.plt.plot(t,y)
        pylab.plt.xlabel('Time')
        pylab.plt.ylabel('Amplitude')
        pylab.plt.subplot(3,1,3)
        plotSpectrum(y,1.0/t_step)
        pylab.plt.show()


class Guitar(object):

    def __init__(self):
        self.E = GuitarString(6)
        self.A = GuitarString(5)
        self.D = GuitarString(4)
        self.G = GuitarString(3)
        self.B = GuitarString(2)
        self.e = GuitarString(1)

    def major_chord(self, note):
        """
        
        >>> g = Guitar()
        >>> g.major_chord('E')
        
        """
    
        notes = []
        
        if note == 'A':
            notes.append(self.A.pluck(0))
            notes.append(self.D.pluck(2))
            notes.append(self.G.pluck(2))
            notes.append(self.B.pluck(2))
            notes.append(self.e.pluck(0))

        if note == 'A#':
            notes.append(self.A.pluck(1))
            notes.append(self.D.pluck(3))
            notes.append(self.G.pluck(3))
            notes.append(self.B.pluck(3))
            notes.append(self.e.pluck(1))

        if note == 'B':
            notes.append(self.A.pluck(2))
            notes.append(self.D.pluck(4))
            notes.append(self.G.pluck(4))
            notes.append(self.B.pluck(4))
            notes.append(self.e.pluck(2))

        if note == 'C':
            notes.append(self.A.pluck(3))
            notes.append(self.D.pluck(5))
            notes.append(self.G.pluck(5))
            notes.append(self.B.pluck(5))
            notes.append(self.e.pluck(3))

        if note == 'C#':
            notes.append(self.A.pluck(4))
            notes.append(self.D.pluck(6))
            notes.append(self.G.pluck(6))
            notes.append(self.B.pluck(6))
            notes.append(self.e.pluck(4))

        if note == 'D':
            notes.append(self.A.pluck(5))
            notes.append(self.D.pluck(7))
            notes.append(self.G.pluck(7))
            notes.append(self.B.pluck(7))
            notes.append(self.e.pluck(5))

        if note == 'D#':
            notes.append(self.A.pluck(6))
            notes.append(self.D.pluck(8))
            notes.append(self.G.pluck(8))
            notes.append(self.B.pluck(8))
            notes.append(self.e.pluck(6))

        if note == 'E':
            notes.append(self.E.pluck(0))
            notes.append(self.A.pluck(2))
            notes.append(self.D.pluck(2))
            notes.append(self.G.pluck(1))
            notes.append(self.B.pluck(0))
            notes.append(self.e.pluck(0))

        if note == 'F':
            notes.append(self.E.pluck(1))
            notes.append(self.A.pluck(3))
            notes.append(self.D.pluck(3))
            notes.append(self.G.pluck(2))
            notes.append(self.B.pluck(1))
            notes.append(self.e.pluck(1))

        if note == 'F#':
            notes.append(self.E.pluck(2))
            notes.append(self.A.pluck(4))
            notes.append(self.D.pluck(4))
            notes.append(self.G.pluck(3))
            notes.append(self.B.pluck(2))
            notes.append(self.e.pluck(2))

        if note == 'G':
            notes.append(self.E.pluck(3))
            notes.append(self.A.pluck(5))
            notes.append(self.D.pluck(5))
            notes.append(self.G.pluck(4))
            notes.append(self.B.pluck(3))
            notes.append(self.e.pluck(3))

        if note == 'G#':
            notes.append(self.E.pluck(4))
            notes.append(self.A.pluck(6))
            notes.append(self.D.pluck(6))
            notes.append(self.G.pluck(5))
            notes.append(self.B.pluck(4))
            notes.append(self.e.pluck(4))

        return notes

if __name__ == '__main__':
    import doctest
    doctest.testmod()


#    root = 110.0
    
#    Fs = 5000.0;  # sampling rate
#    Ts = 1.0/Fs; # sampling interval
#    t = np.arange(0,1,Ts) # time vector

#    just = False
#    ff = root;   # frequency of the signal
#    y = (1.0/4.0)*( 
#          np.sin(2*np.pi*ff*t) 
##        + np.sin(2*np.pi*minor_2nd(ff,just)*t) 
##        + np.sin(2*np.pi*major_2nd(ff,just)*t) 
##        + np.sin(2*np.pi*minor_3rd(ff,just)*t) 
#        + np.sin(2*np.pi*major_3rd(ff,just)*t) 
##        + np.sin(2*np.pi*perfect_4th(ff,just)*t)
##        + np.sin(2*np.pi*diminished_5th(ff,just)*t)
#        + np.sin(2*np.pi*perfect_5th(ff,just)*t)
##        + np.sin(2*np.pi*minor_6th(ff,just)*t) 
##        + np.sin(2*np.pi*major_6th(ff,just)*t) 
#        + np.sin(2*np.pi*minor_7th(ff,just)*t) 
##        + np.sin(2*np.pi*major_7th(ff,just)*t) 
##        + np.sin(2*np.pi*perfect_8th(ff,just)*t)
#        )

#    y = harmonic_waveform(t)

#    pylab.plt.subplot(2,1,1)
#    pylab.plt.plot(t,y)
#    pylab.plt.xlabel('Time')
#    pylab.plt.ylabel('Amplitude')
#    pylab.plt.subplot(2,1,2)
#    plotSpectrum(y,Fs)
#    pylab.plt.show()
    
    
#    interval = minor_7th(root, False)
    
#    print root, interval
#    draw([root, interval])
    
#    print major_chord(root, True)
#    draw(major_chord(root, True))    
#    print major_chord(root, False)
#    draw(major_chord(root, False))

#    print minor_chord(root, True)
#    draw(minor_chord(root, True))    
#    print minor_chord(root, False)
#    draw(minor_chord(root, False))
    
#    print diminished_chord(root, True)
#    draw(diminished_chord(root, True))    
#    print diminished_chord(root, False)
#    draw(diminished_chord(root, False))
    
    