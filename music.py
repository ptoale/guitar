#!/usr/bin/env python
"""
.. module:: music
    :synopsis: Collection of functions for dealing with music.
    
.. moduleauthor:: Patrick Toale

This module provides functions and classes for musical notes. 
It assumes 12-TET tuning, referenced to A440.

"""
import math

"""
Mapping between standard string representation and integer mod 12
"""
note_str_to_int_dict = { 'C'  : 0,
                         'C#' : 1,
                         'D'  : 2,
                         'D#' : 3,
                         'E'  : 4,
                         'F'  : 5,
                         'F#' : 6,
                         'G'  : 7,
                         'G#' : 8,
                         'A'  : 9,
                         'A#' : 10,
                         'B'  : 11 }
note_int_to_str_dict = {y:x for x,y in note_str_to_int_dict.iteritems()}

def convert_str_to_int(string_note):
    """
    Convert a note from string to integer representation.
    
    Args:
        string_note (str): The note as a string
    
    Returns:
        (int) The note as an integer
    
    Throws:
        Exception if string_note is not a string or if its unknown
    
    Produce a list of integers for strings A-G#:
    >>> [convert_str_to_int(i) for i in sorted(note_str_to_int_dict)]
    [9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    
    Unknown notes will throw an exception:
    >>> convert_str_to_int('H')
    Traceback (most recent call last):
       ...
    Exception: Unknown note exception

    """
    if type(string_note) is not str:
        raise Exception('Note type exception')
    elif string_note not in note_str_to_int_dict:
        raise Exception('Unknown note exception')
    else:
        return note_str_to_int_dict[string_note]    
    
def convert_int_to_str(int_note):
    """
    Convert a note from integer to string representation.

    Args:
        int_note (int): The note as an integer
    
    Returns:
        (str) The note as a string

    Throws:
        Exception if int_note is not an integer or if its unknown
    
    Produce a list of strings for integers 0-11:
    >>> [convert_int_to_str(i) for i in sorted(note_int_to_str_dict)]
    ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    Unknown notes will throw an exception:
    >>> convert_int_to_str(42)
    Traceback (most recent call last):
       ...
    Exception: Unknown note exception

    """
    if type(int_note) is not int:
        raise Exception('Note type exception')
    elif int_note not in note_int_to_str_dict:
        raise Exception('Unknown note exception')
    else:
        return note_int_to_str_dict[int_note]    

def note_as_int(note):
    """
    Return the note as an integer.
    
    Args:
        note (int or str): The input note, either string or integer
        
    Returns:
        (int) The note as an integer

    Produce a list of integers for either strings or ints
    >>> [note_as_int(i) for i in sorted(note_str_to_int_dict)]
    [9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    >>> [note_as_int(i) for i in sorted(note_int_to_str_dict)]
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    >>> note_as_int(sorted(note_str_to_int_dict))
    [9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    """
    if type(note) is int:
        return note
    elif type(note) is list:
        return [note_as_int(n) for n in note]
    else:
        return convert_str_to_int(note)

def note_as_str(note):
    """
    Return the note as a string.
    
    Args:
        note (int or str): The input note, either string or integer
        
    Returns:
        (str) The note as a string

    Produce a list of strings for either strings or ints
    >>> [note_as_str(i) for i in sorted(note_str_to_int_dict)]
    ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
    >>> [note_as_str(i) for i in sorted(note_int_to_str_dict)]
    ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    >>> note_as_str(sorted(note_int_to_str_dict))
    ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    """
    if type(note) is str:
        return note
    elif type(note) is list:
        return [note_as_str(n) for n in note]
    else:
        return convert_int_to_str(note)

def increment_note(note, n_semitones):
    """
    Increment a note by n_semitones.
    
    Args:
        note (str or int): The base note
        n_semitones (int): The number of semitones to increment by.
        
    Returns:
        The new note as either a str or int, based on the input type.
        
    Increment by zero returns the same list
    >>> [increment_note(i, 0) for i in sorted(note_str_to_int_dict)]
    ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']

    Increment by nine maps to relative minor keys
    >>> [increment_note(i, 9) for i in sorted(note_str_to_int_dict)]
    ['F#', 'G', 'G#', 'A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F']

    """
    inc = (note_as_int(note) + n_semitones) % 12
    if type(note) is int:
        return inc
    else:
        return convert_int_to_str(inc)

def decrement_note(note, n_semitones):
    """
    Decrement a note by n_semitones.
    
    Args:
        note (str or int): The base note
        n_semitones (int): The number of semitones to decrement by.
        
    Returns:
        The new note as either a str or int, based on the input type.

    Decrement by zero returns the same list
    >>> [decrement_note(i, 0) for i in sorted(note_str_to_int_dict)]
    ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']

    Decrement by three maps to relative minor keys
    >>> [decrement_note(i, 3) for i in sorted(note_str_to_int_dict)]
    ['F#', 'G', 'G#', 'A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F']

    """
    dec = (note_as_int(note) - n_semitones) % 12
    if type(note) is int:
        return dec
    else:
        return convert_int_to_str(dec)

#===============================================================================
#
# Pitch calculations and MIDI representation.
#
#===============================================================================

# MIDI pitch indexing
CONCERT_PITCH = 440.0 # A4 = 440.0 Hz
CONCERT_INDEX = 69    # A4 = 69

def interval_in_cents(freq1, freq2):
    """
    Calculate the interval between frequencies f1 and f2, in cents.
    
    The interval is defined logarithmically and set so that
    one octave, or doubling in frequency, is defined as 1200 cents.
    
    Args:
        freq1 (float): First frequency
        freq2 (float): Second frequency
        
    Returns:
        (float) The interval in cents
        
    Produce intervals in cents for all semitones in first octave above the reference
    >>> [round(interval_in_cents(math.pow(2,i/12.0)*440.0, 440.0), 2) for i in range(13)]
    [0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0]

    Produce intervals in cents for all semitones in first octave below the reference
    >>> [round(interval_in_cents(440.0, math.pow(2,i/12.0)*440.0), 2) for i in range(13)]
    [0.0, -100.0, -200.0, -300.0, -400.0, -500.0, -600.0, -700.0, -800.0, -900.0, -1000.0, -1100.0, -1200.0]
    """
    return 1200.0*math.log(float(freq1)/freq2, 2)

def interval_in_semitones(freq1, freq2):
    """
    Calculate the interval between frequencies f1 and f2, in semitones.
    
    The interval is defined logarithmically and set so that
    one octave, or doubling in frequency, is defined as 12 semitones.
    
    Args:
        freq1 (float): First frequency
        freq2 (float): Second frequency
        
    Returns:
        (float) The interval in semitones

    Produce intervals in semitones for all semitones in first octave above the reference
    >>> [round(interval_in_semitones(math.pow(2,i/12.0)*440.0, 440.0), 2) for i in range(13)]
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

    Produce intervals in semitones for all semitones in first octave below the reference
    >>> [round(interval_in_semitones(440.0, math.pow(2,i/12.0)*440.0), 2) for i in range(13)]
    [0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0]
    """
    return interval_in_cents(freq1, freq2)/100.0

def frequency_from_cents(freq1, interval):
    """
    Calculate the frequency separated from freq1 by interval in cents.
    
    Args:
        freq1 (float): The base frequency
        interval (float): The interval with respect to the base
        
    Returns:
        (float) The frequency
        
    Produce frequencies every 100 cents in first octave above the reference
    >>> [round(frequency_from_cents(440.0, i*100), 2) for i in range(13)]
    [440.0, 466.16, 493.88, 523.25, 554.37, 587.33, 622.25, 659.26, 698.46, 739.99, 783.99, 830.61, 880.0]

    Produce frequencies every 100 cents in first octave below the reference
    >>> [round(frequency_from_cents(440.0, -i*100), 2) for i in range(13)]
    [440.0, 415.3, 392.0, 369.99, 349.23, 329.63, 311.13, 293.66, 277.18, 261.63, 246.94, 233.08, 220.0]
    """
    return freq1*math.pow(2, interval/1200.0)

def frequency_from_semitones(freq1, interval):
    """
    Calculate the frequency separated from freq1 by interval in semitones.
    
    Args:
        freq1 (float): The base frequency
        interval (float): The interval with respect to the base
        
    Returns:
        (float) The frequency

    Produce frequencies of all semitones in first octave above the reference
    >>> [round(frequency_from_semitones(440.0, i), 2) for i in range(13)]
    [440.0, 466.16, 493.88, 523.25, 554.37, 587.33, 622.25, 659.26, 698.46, 739.99, 783.99, 830.61, 880.0]

    Produce frequencies of all semitones in first octave below the reference
    >>> [round(frequency_from_semitones(440.0, -i), 2) for i in range(13)]
    [440.0, 415.3, 392.0, 369.99, 349.23, 329.63, 311.13, 293.66, 277.18, 261.63, 246.94, 233.08, 220.0]
    """
    return frequency_from_cents(freq1, 100.0*interval)


def calc_pitch_index(freq):
    """
    Calculate the MIDI note index and fractional remainder.

    The index is defined so that CONCERT_PITCH coincides with
    CONCERT_INDEX and that the octave is divided evenly into 12
    parts. 
    
    The fractional remainder is defined as the interval above the
    index frequency divided by 100 cents (1 semitone).
    
    By convention the defining pitch is A4=440 Hz at index 69.
    
    Args:
        freq (float): The frequency of the pitch
        
    Returns:
        (index (int), frac (float)): The index and fraction above the index.

    Produce index and fraction for several harmonics of A
    >>> [calc_pitch_index(i) for i in [55.0, 110.0, 220.0, 440.0, 880.0, 1760.0]]
    [(33, 0.0), (45, 0.0), (57, 0.0), (69, 0.0), (81, 0.0), (93, 0.0)]

    Produce index and fraction for a frequency slightly higher than middle C
    >>> calc_pitch_index(264)
    (60, 0.15641287000552745)
    """

    # Get the distance (in semitones) between the frequency and the reference
    n_semitones = interval_in_semitones(freq, CONCERT_PITCH)
    
    # Add/subtract this distance to/from the reference index
    index = int(math.floor(CONCERT_INDEX + n_semitones))

    # Check that we're in the allowed range
    if index < 0:
        raise Exception('Frequency less than {}'.format(calc_pitch_freq(0)))
    elif index > 127:
        raise Exception('Frequncy greater than {}'.format(calc_pitch_freq(127)))

    # Now calculate the fraction above the index
    #   This is defined as the fraction of 100 cents (1 semitone)
    frac = interval_in_semitones(freq, calc_pitch_freq(index))
    
    return index, frac

def calc_pitch_freq(index, frac=0.0):
    """
    Calculate the frequency for a give index and fractional remainder.
    
    Args:
        index (int): MIDI note index
        frac (float): MIDI note fraction (or None)
        
    Returns:
        (float) The frequency
    
    Produce frequencies of open guitar strings in standard tuning
    >>> [round(calc_pitch_freq(i), 2) for i in [40, 45, 50, 55, 59, 64]]
    [82.41, 110.0, 146.83, 196.0, 246.94, 329.63]

    Do the same, but using fraction=1.0 (whole semitone)
    >>> [round(calc_pitch_freq(i-1, 1.0), 2) for i in [40, 45, 50, 55, 59, 64]]
    [82.41, 110.0, 146.83, 196.0, 246.94, 329.63]
    """
    pitch_index = index + frac

    # Check that the index is in the allowed range
    if pitch_index < 0:
        raise Exception('Index less than 0')
    elif pitch_index > 127:
        raise Exception('Index greater than 127')

    # Get the distance (in semitones) between the index and the reference
    n_semitones = pitch_index - CONCERT_INDEX

    # Calculate the frequency of this pitch
    return frequency_from_semitones(CONCERT_PITCH, n_semitones)


def get_index(note, octave):
    """
    Get the index [0-127] from the note and octave
    
    Args:
        note (int): The note number [0-11]
        octave (int) : The note octave [0-10]
        
    Returns:
        (int) The MIDI index
    
    Get indices of lowest note, middle C, A440, and highest note
    >>> [get_index(i, j) for i,j in [('C',0), ('C',5), ('A',5), ('G',10)]]
    [0, 60, 69, 127]
    
    """
    return note_as_int(note) + octave*12

def get_octave_and_note(index):
    """
    Get the note [0-12] and octave [0-10] from the index [0-127]
    
    Args:
        index (int): The MIDI index [0-127]
        
    Returns:
        (int, int) The octave and note

    Get octave and note of several important indices
    >>> [get_octave_and_note(i) for i in [0, 60, 69, 127]]
    [(0, 0), (5, 0), (5, 9), (10, 7)]
    
    """
    return divmod(index, 12)


"""
Note patterns
"""

# Diatonic scale modes
DIATONIC_IONIAN     = [0, 2, 4, 5, 7, 9, 11]
DIATONIC_DORIAN     = [0, 2, 3, 5, 7, 9, 10]
DIATONIC_PHRYGIAN   = [0, 1, 3, 5, 7, 8, 10]
DIATONIC_LYDIAN     = [0, 2, 4, 6, 7, 9, 11]
DIATONIC_MIXOLYDIAN = [0, 2, 4, 5, 7, 9, 10]
DIATONIC_AEOLIAN    = [0, 2, 3, 5, 7, 8, 10]
DIATONIC_LOCRIAN    = [0, 1, 3, 5, 6, 8, 10]

DIATONIC_SCALE = [DIATONIC_IONIAN, DIATONIC_DORIAN, DIATONIC_PHRYGIAN, DIATONIC_LYDIAN,
                  DIATONIC_MIXOLYDIAN, DIATONIC_AEOLIAN, DIATONIC_LOCRIAN]

MAJOR_SCALE = DIATONIC_IONIAN
MINOR_SCALE = DIATONIC_AEOLIAN

# Pentatonic scale modes
PENTATONIC_MAJOR     = [0, 2, 4, 7,  9]
PENTATONIC_SUSPENDED = [0, 2, 5, 7, 10]
PENTATONIC_MAN_GONG  = [0, 3, 5, 8, 10]
PENTATONIC_RITUSEN   = [0, 2, 5, 7,  9]
PENTATONIC_MINOR     = [0, 3, 5, 7, 10]

PENTATONIC_SCALE = [PENTATONIC_MAJOR, PENTATONIC_SUSPENDED, PENTATONIC_MAN_GONG,
                    PENTATONIC_RITUSEN, PENTATONIC_MINOR]

# Whole-tone scale
WHOLE_TONE = [0, 2, 4, 6, 8, 10]
WHOLE_TONE_SCALE = [WHOLE_TONE]

# Whole-half-tone scales
DIMINISHED_SCALE = [0, 2, 3, 5, 6, 8, 9, 11]
OCTATONIC_SCALE  = [0, 1, 3, 4, 6, 7, 9, 10]

WHOLE_HALF_TONE_SCALE = [DIMINISHED_SCALE, OCTATONIC_SCALE]

# CHORDS

MAJOR_TRIAD      = [0, 4, 7]
MINOR_TRIAD      = [0, 3, 7]
DIMINISHED_TRIAD = [0, 3, 6]
AUGMENTED_TRIAD  = [0, 4, 8]
SUSPENDED_TRIAD  = [0, 5, 7]

MAJOR_SIXTH                = [0, 4, 7,  9]
MINOR_SIXTH                = [0, 3, 7,  9]
MAJOR_SEVENTH              = [0, 4, 7, 11]
MINOR_SEVENTH              = [0, 3, 7, 11]
DOMINANT_SEVENTH           = [0, 4, 7, 10]
HALF_DIMINISHED_SEVENTH    = [0, 3, 6, 10]
DIMINISHED_SEVENTH         = [0, 3, 6,  9]
AUGMENTED_SEVENTH          = [0, 4, 8, 10]
SUSPENDED_DOMINANT_SEVENTH = [0, 5, 7, 10]
MINOR_MAJOR_SEVENTH        = [0, 5, 7, 11]

MAJOR_NINTH    = [0, 4, 7, 11, 14]
MINOR_NINTH    = [0, 3, 7, 11, 14]
DOMINANT_NINTH = [0, 4, 7, 10, 14]

MAJOR_ELEVENTH    = [0, 4, 7, 11, 14, 17]
MINOR_ELEVENTH    = [0, 3, 7, 11, 14, 17]
DOMINANT_ELEVENTH = [0, 4, 7, 10, 14, 17]

MAJOR_THIRTEENTH    = [0, 4, 7, 11, 14, 17, 21]
MINOR_THIRTEENTH    = [0, 3, 7, 11, 14, 17, 21]
DOMINANT_THIRTEENTH = [0, 4, 7, 10, 14, 17, 21]


CHORDS = { 'maj': MAJOR_TRIAD,
           'min': MINOR_TRIAD,
           'dim': DIMINISHED_TRIAD,
           'aug': AUGMENTED_TRIAD,
           'maj6': MAJOR_SIXTH,
           'min6': MINOR_SIXTH,
           'maj7': MAJOR_SEVENTH,
           'min7': MINOR_SEVENTH,
           'dom7': DOMINANT_SEVENTH,
           'dim7': DIMINISHED_SEVENTH,
           'hdim7': HALF_DIMINISHED_SEVENTH,
           'aug7': AUGMENTED_SEVENTH,
           'sus2': SUSPENDED_DOMINANT_SEVENTH,
           'sus4': MINOR_MAJOR_SEVENTH,
           'maj9': MAJOR_NINTH,
           'min9': MINOR_NINTH,
           'dom9': DOMINANT_NINTH,
           'maj11': MAJOR_ELEVENTH,
           'min11': MINOR_ELEVENTH,
           'dom11': DOMINANT_ELEVENTH,
           'maj13': MAJOR_THIRTEENTH,
           'min13': MINOR_THIRTEENTH,
           'dom13': DOMINANT_THIRTEENTH
            }

SCALES = { 'diatonic': DIATONIC_SCALE,
           'pentatonic': PENTATONIC_SCALE}

MAJOR_SCALE_DEGREE = ['maj', 'min', 'min', 'maj', 'maj', 'min', 'dim']

def scale(root, stype='diatonic', mode=0, as_strings=False):
    """
    Generate a scale.
    
    Args:
        root (int): The root note of the scale
        stype (str): The type of scale [diatonic]
        mode (int): The mode of the scale
        as_strings (bool): Return notes as strings [False]
        
    Returns:
        The scale as a list of notes
        
    >>> scale(0)
    [0, 2, 4, 5, 7, 9, 11]
    >>> scale(0, mode=5)
    [0, 2, 3, 5, 7, 8, 10]
    >>> scale(7)
    [7, 9, 11, 0, 2, 4, 6]
    >>> scale(7, mode=5)
    [7, 9, 10, 0, 2, 3, 5]

    """

    signature = DIATONIC_IONIAN
    if stype in SCALES and mode < len(SCALES[stype]):
        signature = SCALES[stype][mode]

    if as_strings:    
        notes = [note_as_str((root+i)%12) for i in signature]
    else:
        notes = [(root+i)%12 for i in signature]
    
    return notes    
    

def chord(root, quality='maj', as_strings=False):
    """
    Generate a chord.
    
    Args:
        root (int): The root note of the chord
        quality (str): The type of chord [maj, min, dim, aug]
        as_strings (bool): Return notes as strings [False]
        
    Returns:
        The chord as a list of notes
        
    >>> note_as_str(chord(7))
    ['G', 'B', 'D']
    >>> note_as_str(chord(7, 'min'))
    ['G', 'A#', 'D']
    >>> chord(7, 'dim')
    [7, 10, 1]
    >>> chord(7, 'aug')
    [7, 11, 3]
    >>> chord(7, 'maj7')
    [7, 11, 2, 6]
    >>> chord(7, 'min7')
    [7, 10, 2, 6]
    >>> chord(7, 'dom7')
    [7, 11, 2, 5]
    >>> chord(7, 'dim7')
    [7, 10, 1, 4]
    >>> chord(7, 'hdim7')
    [7, 10, 1, 5]
    >>> chord(7, 'aug7')
    [7, 11, 3, 5]
    >>> chord(7, 'sus2')
    [7, 0, 2, 5]
    >>> chord(7, 'sus4')
    [7, 0, 2, 6]

    """
    
    signature = []
    if quality in CHORDS:
        signature = CHORDS[quality]
    
    if as_strings:    
        notes = [note_as_str((root+i)%12) for i in signature]
    else:
        notes = [(root+i)%12 for i in signature]
    
    return notes    

def scale_chords(root, as_strings=False):
    """
    Generate all chords of a major scale.
    
    Pattern is: I ii iii IV V vi viid
    
    Args:
        root (int): Root of scale
        
    Returns:
        List of chords [[n,n,n], [n,n,n],...]
        
    >>> scale_chords(7)
    [[7, 11, 2], [9, 0, 4], [11, 2, 6], [0, 4, 7], [2, 6, 9], [4, 7, 11], [6, 9, 0]]
    >>> scale_chords(7, as_strings=True)
    [['G', 'B', 'D'], ['A', 'C', 'E'], ['B', 'D', 'F#'], ['C', 'E', 'G'], ['D', 'F#', 'A'], ['E', 'G', 'B'], ['F#', 'A', 'C']]
    
    """ 
    pattern = MAJOR_SCALE_DEGREE
        
    return [chord(scale(root)[i], pattern[i], as_strings) for i in range(7)]
        
class PitchSet(object):

    def __init__(self, list_of_pitches):
        self.pitches = list_of_pitches
        self.npitch = len(self.pitches)
        
    def rotate(self, steps):
        """
        Rotate the list. Positive steps = counter-clockwise rotation.
        
        Args:
            steps (int): number of steps to rotate
            
        Returns:
            A new, rotated PitchSet
            
        >>> ps = PitchSet([0,2,4,5,7,9,11])
        >>> ps.rotate(1)
        [2, 4, 5, 7, 9, 11, 0]
        >>> ps.rotate(1+7)
        [2, 4, 5, 7, 9, 11, 0]
        >>> ps.rotate(1+8)
        [4, 5, 7, 9, 11, 0, 2]
        >>> ps.rotate(-1)
        [11, 0, 2, 4, 5, 7, 9]

        """
        steps = steps % self.npitch
        return PitchSet(self.pitches[steps:] + self.pitches[:steps])
                
    def sharpen(self, index):
        """
        Raise one pitch by one semitone.
        
        Args:
            index (int): the index of the pitch to raise
            
        Returns:
            A new, slightly sharpened PitchSet
            
        >>> ps = PitchSet([0,2,4,5,7,9,11])
        >>> ps.sharpen(3)
        [0, 2, 4, 6, 7, 9, 11]
        >>> ps.sharpen(3+7)
        [0, 2, 4, 6, 7, 9, 11]
        >>> ps.sharpen(3-7)
        [0, 2, 4, 6, 7, 9, 11]

        """
        return PitchSet( [(self.pitches[i] + (1 if i==(index%self.npitch) else 0))%12
                          for i in range(self.npitch)]
                        )
                
    def flatten(self, index):
        """
        Lower one pitch by one semitone.
        
        Args:
            index (int): the index of the pitch to lower
            
        Returns:
            A new, slightly flattened PitchSet

        >>> ps = PitchSet([0,2,4,5,7,9,11])
        >>> ps.flatten(6)
        [0, 2, 4, 5, 7, 9, 10]
        >>> ps.flatten(6+7)
        [0, 2, 4, 5, 7, 9, 10]
        >>> ps.flatten(6-7)
        [0, 2, 4, 5, 7, 9, 10]

        """
        return PitchSet( [(self.pitches[i] - (1 if i==(index%self.npitch) else 0))%12
                          for i in range(self.npitch)]
                        )
                

    def __str__(self):
        return repr(self)
        
    def __repr__(self):
        return str(self.pitches)
