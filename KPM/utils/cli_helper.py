def str_to_tuple(s):
    '''Converts a string-form tuple to a Python tuple.
    
    Takes a string in the form '(x, y)' (where x and y are floats)
    and returns the tuple (x, y).

    Used within CLI argument parsing.
    '''
    s = s[1:-1] # Remove brackets.
    t = tuple(map(float, s.split(', ')))

    return t