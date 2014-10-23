#!/usr/bin/env python

import numpy as np

from music import get_octave_and_note, note_as_str, get_index, calc_pitch_freq
from music import CHORDS

#===============================================================================
#
# Guitar specific stuff.
#
#===============================================================================

# Standard 6-string guitar tuning, from high to low 
#  64 = E_5
#  59 = B_4
#  55 = G_4
#  50 = D_4
#  45 = A_3
#  40 = E_3
STANDARD_TUNING = [64, 59, 55, 50, 45, 40]
NFRET = 20
MIN_INDEX = STANDARD_TUNING[-1]
MAX_INDEX = STANDARD_TUNING[0] + NFRET

def get_guitar_indices(note):
    """
    Get all octaves of note on guitar.
    
    Args:
        note (int): The note
        
    Returns:
        List of MIDI indices
        
    >>> get_guitar_indices(0)
    [48, 60, 72, 84]
    >>> get_guitar_indices(4)
    [40, 52, 64, 76]
    
    """
    indices = []
    for octave in range(11):
        index = get_index(note, octave)
        if (index >= MIN_INDEX) and (index <= MAX_INDEX):
            indices.append(index)
    
    return indices

def get_guitar_chordes(root, quality='maj'):
    """
    Get all possible 3-note chords
    
    >>> get_guitar_chordes(4)
    [[40, 44, 47], [52, 56, 59], [64, 68, 71], [76, 80, 83]]
    >>> get_guitar_chordes(9)
    [[45, 49, 52], [57, 61, 64], [69, 73, 76]]
    >>> get_guitar_chordes(2)
    [[50, 54, 57], [62, 66, 69], [74, 78, 81]]
    >>> get_guitar_chordes(7)
    [[43, 47, 50], [55, 59, 62], [67, 71, 74]]
    >>> get_guitar_chordes(11)
    [[47, 51, 54], [59, 63, 66], [71, 75, 78]]

    """
    sig = CHORDS[quality]
    
    good_chords = []
    max_interval = CHORDS[quality][-1]
    for i in get_guitar_indices(root):
        if (i+max_interval) <= MAX_INDEX:
            ch = [(j + i) for j in sig]
        
            good_chords.append(ch)
    
    
    return good_chords

def get_fret_index(string, fret):
    """
    Convert a string and fret into a unique index. Fret 0 corresponds to 
    the open string.
        
    Args:
        string (int): The string [0, nstring-1]
        fret (int): The fret [0, nfret]
        
    Returns:
        (int) The unique index [0, nfret+(nfret+1)*(nstring-1)]
        
    Get the fret index of all open strings, from Low to High
    >>> [get_fret_index(i,0) for i in [5, 4, 3, 2, 1, 0]]
    [0, 21, 42, 63, 84, 105]
    """
    return fret + (NFRET+1)*(len(STANDARD_TUNING)-1-string)
        
def get_string_and_fret(fret_id):
    """
    Convert an index into string and fret pair. Fret 0 corresponds to 
    the open string.
        
    Args:
        fret_id (int): The index [0, nfret+(nfret+1)*(nstring-1)]
        
    Returns:
        (int, int) The string, fret pair
        
    Get string, fret pairs for indices corresponding to open strings
    >>> [get_string_and_fret(i) for i in [0, 21, 42, 63, 84, 105]]
    [(5, 0), (4, 0), (3, 0), (2, 0), (1, 0), (0, 0)]
    """
    s,f = divmod(fret_id, NFRET+1)
    return (len(STANDARD_TUNING)-1-s), f

def get_note_id(string, fret):
    """
    Get the MIDI note id at the string-fret position.
    
    Args:
        string (int): Stings numbered 0-len(STANDARD_TUNING), Hi-Low
        fret (int): Frets numbered 0=Open, 1-NFRET from nut to bridge
        
    Returns:
        The note id
        
    >>> get_note_id(0, 0)
    64
    """
    return STANDARD_TUNING[string] + fret

def print_fretboard(mode='fret_ids', mask=None, only=None):
    """
    Print a representation of a fretboard.
    
    >>> print_fretboard()
       Frets:    0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19   20
    String 1:  105  106  107  108  109  110  111  112  113  114  115  116  117  118  119  120  121  122  123  124  125
    String 2:   84   85   86   87   88   89   90   91   92   93   94   95   96   97   98   99  100  101  102  103  104
    String 3:   63   64   65   66   67   68   69   70   71   72   73   74   75   76   77   78   79   80   81   82   83
    String 4:   42   43   44   45   46   47   48   49   50   51   52   53   54   55   56   57   58   59   60   61   62
    String 5:   21   22   23   24   25   26   27   28   29   30   31   32   33   34   35   36   37   38   39   40   41
    String 6:    0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19   20

    >>> print_fretboard(mode='note_ids')
       Frets:    0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19   20
    String 1:   64   65   66   67   68   69   70   71   72   73   74   75   76   77   78   79   80   81   82   83   84
    String 2:   59   60   61   62   63   64   65   66   67   68   69   70   71   72   73   74   75   76   77   78   79
    String 3:   55   56   57   58   59   60   61   62   63   64   65   66   67   68   69   70   71   72   73   74   75
    String 4:   50   51   52   53   54   55   56   57   58   59   60   61   62   63   64   65   66   67   68   69   70
    String 5:   45   46   47   48   49   50   51   52   53   54   55   56   57   58   59   60   61   62   63   64   65
    String 6:   40   41   42   43   44   45   46   47   48   49   50   51   52   53   54   55   56   57   58   59   60

    >>> print_fretboard(mode='notes')
       Frets:    0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19   20
    String 1:    E    F   F#    G   G#    A   A#    B    C   C#    D   D#    E    F   F#    G   G#    A   A#    B    C
    String 2:    B    C   C#    D   D#    E    F   F#    G   G#    A   A#    B    C   C#    D   D#    E    F   F#    G
    String 3:    G   G#    A   A#    B    C   C#    D   D#    E    F   F#    G   G#    A   A#    B    C   C#    D   D#
    String 4:    D   D#    E    F   F#    G   G#    A   A#    B    C   C#    D   D#    E    F   F#    G   G#    A   A#
    String 5:    A   A#    B    C   C#    D   D#    E    F   F#    G   G#    A   A#    B    C   C#    D   D#    E    F
    String 6:    E    F   F#    G   G#    A   A#    B    C   C#    D   D#    E    F   F#    G   G#    A   A#    B    C

    >>> print_fretboard(mode='octaves')
       Frets:    0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19   20
    String 1:    5    5    5    5    5    5    5    5    6    6    6    6    6    6    6    6    6    6    6    6    7
    String 2:    4    5    5    5    5    5    5    5    5    5    5    5    5    6    6    6    6    6    6    6    6
    String 3:    4    4    4    4    4    5    5    5    5    5    5    5    5    5    5    5    5    6    6    6    6
    String 4:    4    4    4    4    4    4    4    4    4    4    5    5    5    5    5    5    5    5    5    5    5
    String 5:    3    3    3    4    4    4    4    4    4    4    4    4    4    4    4    5    5    5    5    5    5
    String 6:    3    3    3    3    3    3    3    3    4    4    4    4    4    4    4    4    4    4    4    4    5

    """
    frets = "   Frets: " + " ".join('{:>4}'.format(n) for n in range(NFRET+1))
    print(frets)

    for i in range(len(STANDARD_TUNING)):
        things = []
        for j in range(NFRET+1):
            fret_id = get_fret_index(i, j)
            note_id = get_note_id(i, j)
            octave, note = get_octave_and_note(get_note_id(i, j))
            
            if mode == 'fret_ids':
                if mask and fret_id in mask:
                    things.append(' ')
                elif only and fret_id not in only:
                    things.append(' ')
                else:
                    things.append(fret_id)
                            
            elif mode == 'note_ids':
                if mask and note_id in mask:
                    things.append(' ')
                elif only and note_id not in only:
                    things.append(' ')
                else:
                    things.append(note_id)
            
            elif mode == 'notes':
                if mask and note in mask:
                    things.append(' ')
                elif only and note not in only:
                    things.append(' ')
                else:
                    things.append(note_as_str(note))

            elif mode == 'octaves':
                if mask and octave in mask:
                    things.append(' ')
                elif only and octave not in only:
                    things.append(' ')
                else:
                    things.append(octave)

        ids = "String {}: ".format(i+1) + " ".join('{:>4}'.format(n) for n in things)
        print(ids)

class Harmonic(object):
    """
    Class to represent a member of a harmonic series produced by plucking
    a string.
    """

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

    LENGTH = 90.0/100.0           # 90 cm
    PLUCK_DIST = 15.0/100.0       # 15 cm
    PLUCK_HEIGHT = 5.0/1000.0     #  5 mm

    def __init__(self, open_note_id, length=LENGTH, damping=0.0, stiffness=0.0):
        self.open_freq = calc_pitch_freq(open_note_id)
        self.length = length
        self.damping = damping
        self.stiffness = stiffness
        
        self.velocity = 2.0*self.length*self.open_freq
    
    def get_overtone(self, fret, mode, pluck_distance=PLUCK_DIST, pluck_height=PLUCK_HEIGHT):
        """
        Get the nth overtone of the note at a given fret.
        
        Args:
            fret (int): The fret of the played note (0 = open string)
            mode (int): The mode of the overtone (>0)
            
        Kwargs:
            pluck_distance (float): The distance between the bridge and the pick (in meters)
            pluck_height (float): The distance the string is plucked (in meters)
            
        Returns:
            A list containing the frequency, the harmonic weight, and the decay constant
        
        The lowest mode of each string of an ideal guitar
        >>> gs = GuitarString(40)
        >>> gs.get_overtone(0, 1)
        [82.406889228217494, 0.00364756261112416, inf]
        >>> gs = GuitarString(45)
        >>> gs.get_overtone(0, 1)
        [110.0, 0.00364756261112416, inf]
        >>> gs = GuitarString(50)
        >>> gs.get_overtone(0, 1)
        [146.83238395870379, 0.00364756261112416, inf]
        >>> gs = GuitarString(55)
        >>> gs.get_overtone(0, 1)
        [195.99771799087463, 0.00364756261112416, inf]
        >>> gs = GuitarString(59)
        >>> gs.get_overtone(0, 1)
        [246.94165062806206, 0.00364756261112416, inf]
        >>> gs = GuitarString(64)
        >>> gs.get_overtone(0, 1)
        [329.62755691286992, 0.00364756261112416, inf]

        The 4th mode of E, which should have the same frequency as e
        >>> gs = GuitarString(40)
        >>> gs.get_overtone(0, 4)
        [329.62755691286998, 0.00039486023539097789, inf]

        The 4th mode of E, for a non-ideal string
        >>> gs = GuitarString(40, damping=1.7, stiffness=0.008)
        >>> gs.get_overtone(0, 4)
        [329.79628303917298, 0.00039486023539097789, 0.14705882352941177]

        """

        # Find the length of the fretted string        
        L = self.length/pow(2.0, fret/12.0)
    
        # Find the pitch of this fret
        f1 = self.velocity/(2.0*L)

        # Shift the frequency due to string stiffness
        delta = np.sqrt(1.0 + pow(mode*self.stiffness, 2.0))
        frequency = mode*f1*delta
        
        # Find the weight of this mode (fourier coefficient)
        d = pluck_distance
        h = pluck_height
        weight = 2*h*L*L/(np.pi*np.pi*mode*mode*d*(L-d))*np.sin(np.pi*mode*d/L)

        # Find the time decay constant of this mode
        decay = np.inf if (self.damping == 0.0) else 1.0/(mode*self.damping)

        return [frequency, weight, decay]
    
    def get_overtones(self, fret, max_mode, pluck_distance=PLUCK_DIST, pluck_height=PLUCK_HEIGHT):
        """
        Get the first n harmonics of the note at a given fret (including the note itself).
        
        Args:
            fret (int): The fret of the played note (0 = open string)
            max_mode (int): The mode of the highest overtone to return (>0)
            
        Kwargs:
            pluck_distance (float): The distance between the bridge and the pick (in meters)
            pluck_height (float): The distance the string is plucked (in meters)
            
        Returns:
            A list of lists containing the frequency, the harmonic weight, and the decay constant
        
        The lowest 4 modes of an ideal E string
        >>> gs = GuitarString(40)
        >>> gs.get_overtones(0, 4)  # doctest: +NORMALIZE_WHITESPACE
        [[82.406889228217494, 0.00364756261112416, inf], 
         [164.81377845643499, 0.0015794409415639111, inf], 
         [247.22066768465248, 0.0008105694691387023, inf], 
         [329.62755691286998, 0.00039486023539097789, inf]]        
         
        The lowest 4 modes of real E string
        >>> gs = GuitarString(40, damping=1.7, stiffness=0.008)
        >>> gs.get_overtones(0, 4)  # doctest: +NORMALIZE_WHITESPACE
        [[82.409526206481829, 0.00364756261112416, 0.5882352941176471], 
         [164.83487327009573, 0.0015794409415639111, 0.29411764705882354], 
         [247.29185698716185, 0.0008105694691387023, 0.19607843137254904], 
         [329.79628303917298, 0.00039486023539097789, 0.14705882352941177]]
         
        """
        ts = []
        for n in range(max_mode):
            mode = n+1
            ts.append(self.get_overtone(fret, mode, pluck_distance, pluck_height))
            
        return ts
    
    
class Guitar(object):
    """
    Guitar Class.
    
    TODO: make string numbering consistent
        string_and_fret_to_index and index_to_string_and_fret work with [0, nstring-1]
        get_note works with [1, nstring]
    
    """


    def __init__(self, tuning=STANDARD_TUNING, nfret=20):
        """
        Create an instance of a Guitar.
        
        Args:
            tuning (list of ints): list of pitch indices for strings low-to-high
            nfret (int): the number of frets
        """
        
        self.tuning = tuning
        self.nfret = nfret
        
        self.nstring = len(self.tuning)

        # Experimental shifting from [40-84] to [4-48] or [0-44]
        #   [4-48] keeps order of notes the same
        #   [0-44] requires [0,1,2,3,4,...,11] goes to [8,9,10,11,0,1,...,7]
        #          so increment by 8 or decrement by 4
        low_index = self.tuning[-1]
        octave, note = get_octave_and_note(low_index)

        # [4-48]
        self.index_shift = low_index - note
        self.note_perm = [i for i in range(12)]
        
        # [0-44]
#        self.index_shift = low_index
#        self.note_perm = [decrement_note(i,note) for i in range(12)]

        self.shifted_tuning = [i-self.index_shift for i in self.tuning]

#        print("Shifted Tuning = {:>2}".format(self.shifted_tuning))
#        print("Note Permutation = {:>2}".format(self.note_perm))


        self._initialize_maps()
#        print("Note Map = {}".format(self.note_map))
#        print("Pos Map = {}".format(self.pos_map))

    def _initialize_maps(self):

        self.note_map = {}
        self.pos_map = {}
        for i in range(self.nstring):
            for j in range(self.nfret+1):
                pos = get_fret_index(i, j)
                note = self.tuning[i] + j
                
                self.note_map[pos] = note
                
                if note in self.pos_map:
                    self.pos_map[note].append(pos)
                else:
                    self.pos_map[note] = [pos]

    def get_note_id(self, string, fret):
        """
        Get the note at a given string-fret position in current tuning.
        
        Args:
            string (int): Guitar string, numbering is Hi=1 to Lo=nstring
            fret (int): Guitar fret, numbering is Nut=0 to Bridge=nfret
        
        Returns:
            The note (str)

        Get the notes of all open strings and [E-5,A-5,D-5,G-4,B-5,E-5]
        >>> guitar = Guitar()
        >>> [guitar.get_note_id(i,j) for i,j in [(6, 0), (5, 0), (4, 0), (3, 0), (2, 0), (1, 0)]]
        [40, 45, 50, 55, 59, 64]
        >>> [guitar.get_note_id(i,j) for i,j in [(6, 5), (5, 5), (4, 5), (3, 4), (2, 5), (1, 5)]]
        [45, 50, 55, 59, 64, 69]
        
        """
        if (string <= 0) or (string > self.nstring):
            raise Exception('String range exception')
        elif (fret < 0) or (fret > self.nfret):
            raise Exception('Fret range exception')

        pos = get_fret_index(string-1, fret)
        try:
            return self.note_map[pos]
        except:
            print("Position %s not in map!!!".format(pos))
            raise
    
    
    def get_positions(self, note_id):
        """
        Get the list of positions for a note.
        
        Args:
            note_id (int): MIDI note id
            
        Returns:
            List of positions by fret_id (int)

        Get the positions of all 'A3' notes
        >>> guitar = Guitar()
        >>> sorted(guitar.get_positions(40+5))
        [5, 21]
        
        Get the positions of all 'C5' notes
        >>> sorted(guitar.get_positions(36+12+12))
        [20, 36, 52, 68, 85]
        """
        try:
            return self.pos_map[note_id]
        except:
            print("Note %s not in map!!!".format(note_id))
            raise
            
            
    
    
    def print_fretboard(self, notes_to_show=None):
        frets = [str(i).ljust(2) for i in range(self.nfret+1)]
        print("      Frets: {}".format(frets))

        use_strings = True
        if notes_to_show and type(notes_to_show[0]) is int:
            use_strings = False        

        for i in range(self.nstring):
            string_notes = []
            for j in range(self.nfret+1):
                note = self.get_note(i+1, j)
                if use_strings:
                    note = note_as_str(note)
                if notes_to_show and note not in notes_to_show:
                    note = '  '
                str_note = str(note).ljust(2)
                    
                string_notes.append(str_note)        

            print("   String {}: {}".format(i+1, string_notes))

            
        
 
def midi():

    print("TESTING: midi pitch calculations")

    open_string_freqs   = [82.41, 110.00, 146.83, 196.00, 246.94, 329.63]
    open_string_indices = [40, 45, 50, 55, 59, 64]

    for f,i in zip(open_string_freqs, open_string_indices):
        n, r = calc_pitch_index(f)
        print("   String {}: {:>6.2f} = ({:>3}, {:5.4f}) should be {:>3}".format(1+open_string_freqs.index(f), f, n, r, i))
    
    for f,i in zip(open_string_freqs, open_string_indices):
        f2 = calc_pitch_freq(i)
        print("   String {}: ({:>3}, {:5.4f}) = {:>6.2f} should be {:>3}".format(1+open_string_indices.index(i), i, 0.0, f2, f))
    
    for i in range(127):
        o, n = get_octave_and_note(i)
        f = calc_pitch_freq(i)
        print("   Index {:>3}: {:>2}-{:<2} at {:>8.2f} Hz".format(i, note_as_str(n), o, f))

    freqs =[83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0]
    for f in freqs:
        n, r = calc_pitch_index(f)
        f2 = calc_pitch_freq(n, r)
        print("   Freq {} Hz = ({:>3}, {:5.4f}) CHECK-> {}".format(f, n, r, f2))

def notes():

    print("TESTING: note convertions")
    note_strings = [convert_int_to_str(i) for i in range(12)]
    print("   Standard notes: {}".format(note_strings))
    note_ints = [convert_str_to_int(i) for i in note_strings]
    print("   Standard nums: {}".format(note_ints))
    
    print("TESTING: note shifting")
    shift_up_note_strings = [increment_note(i,1) for i in note_strings]
    print("   Shifted up notes: {}".format(shift_up_note_strings))
    shift_up_note_ints = [increment_note(i,1) for i in note_ints]
    print("   Shifted up nums: {}".format(shift_up_note_ints))
    shift_dn_note_strings = [decrement_note(i,1) for i in note_strings]
    print("   Shifted down notes: {}".format(shift_dn_note_strings))
    shift_dn_note_ints = [decrement_note(i,1) for i in note_ints]
    print("   Shifted down nums: {}".format(shift_dn_note_ints))
    

def guitar_old():

    print("TESTING: guitar")
    guitar = Guitar()
    guitar.print_fretboard(['B','C'])
    guitar.print_fretboard([11, 0])
    guitar.print_fretboard([0,1,2,3,4,5,6,7,8,9,10,11])

    for i in sorted(note_int_to_str_dict):
        positions = guitar.get_positions(i)
        print("   {:>2} Positions: {}".format(note_as_str(i), positions))


def chords_and_scales():

    as_str = True
    print("Major Scale Chords:")
    print("C  Chords = {}".format(scale_chords(0, as_str)))
    print("Db Chords = {}".format(scale_chords(1, as_str)))
    print("D  Chords = {}".format(scale_chords(2, as_str)))
    print("Eb Chords = {}".format(scale_chords(3, as_str)))
    print("E  Chords = {}".format(scale_chords(4, as_str)))
    print("F  Chords = {}".format(scale_chords(5, as_str)))
    print("Fs Chords = {}".format(scale_chords(6, as_str)))
    print("Gb Chords = {}".format(scale_chords(6, as_str)))
    print("G  Chords = {}".format(scale_chords(7, as_str)))
    print("Ab Chords = {}".format(scale_chords(8, as_str)))
    print("A  Chords = {}".format(scale_chords(9, as_str)))
    print("Bb Chords = {}".format(scale_chords(10, as_str)))
    print("B  Chords = {}".format(scale_chords(11, as_str)))

    print("Chords by Note:")
    for i in range(12):
        print("{} Chords = {} {} {} {}".format(note_as_str(i).ljust(2), 
            chord(i, 'maj', as_str), chord(i, 'min', as_str), chord(i, 'dim', as_str), chord(i, 'aug', as_str)))

 
            
if __name__ == '__main__':
    import doctest
    doctest.testmod()
    
#    for i in range(127):
#        o, n = get_octave_and_note(i)
#        f = calc_pitch_freq(i)
#        print("   Index {:>3}: {:>2}-{:<2} at {:>8.2f} Hz".format(i, note_as_str(n), o, f))

    
   