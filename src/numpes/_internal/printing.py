"""Module containing extra print functionality for descriptive printing"""

from typing import TYPE_CHECKING

import numpy as np

from numpes._config import CFG

if TYPE_CHECKING:
    from typing import Any, Callable, Literal, Optional


# [untested/unverified]
def sym_replace(arr: str, char: str = CFG.sym_char) -> str:
    """Replace the (assumed to be symmetric) numbers located in the upper-triangular part of an 
    2d square NumPy array with `char`. This is to be used for arrays which are symmetric, and
    where the upper-triangular values can be inferred from the lower part. The function does
    not check if the array provided is 2d  or square, and undefined behavior might follow
    otherwise.

    The function works for all size `n >= 1`, for integers and floating point, decimal and scientific notation, and preserves leading/trailing spaces and alignment due to negative values and dropped trailing zeros.

    Parameters
    ----------
    arr : str
        String representation of an array, presumable `str(A)`, where `A` is a NumPy array of size `(n, n)`
    char : str
        Character for which to replace the upper-triangular part. It is assumed to be of length 1
        (a single character), but this is not enforced.

    Warns
    -----
    RuntimeWarning
        If the maximum number length in the array cannot be determined. If raised, the original array
        without any modifications is returned.

    Warnings
    --------
    This feature has not been rigorously tested, and relies on two key assumptions (see source code) about NumPy string
    formatting which might fail for certain edge cases, or may change in the future.

    Examples
    --------
    >>> Q = np.array([[ 1, -1],
                      [-1,  1])
    >>> print(sym_replace(str(Q)))
    [[ 1, * ],
     [-1,  1])

    Truncation is also supported.
     
    >>> A = np.arange(10_000).reshape(100, 100) * 1E-2
    >>> Q = A + A.T
    >>> with np.printoptions(precision=2, edgeitems=2):
    >>>     print(sym_replace(str(Q), char='•'))
    [[  0.     •    ...   •      •   ]
     [  1.01   2.02 ...   •      •   ]
     ...
     [ 98.98  99.99 ... 197.96   •   ]
     [ 99.99 101.   ... 198.97 199.98]]
    """
    # NOTE: This function works under the assumption that the 'footprint' of each number in a NumPy
    # string is consistent across all numbers, and that the footprint is determined by the
    # longest number (in terms of characters) in the array.

    lines = arr.splitlines()
    try:
        idx_trunc = lines.index(' ...')
    except ValueError as _:
        idx_trunc = None

    # The num_length is equal to the length of the first line, minus the three characters '[[' and ']', minus the ' ' characters in between the numbers, all divided by the number of columns. If there is truncation, we must subtract the three '...' characters and divide by one fewer column
    if idx_trunc is None:
        num_length = int((len(lines[0]) - 3 - (len(lines) - 1)) / len(lines))
    else:
        num_length = int((len(lines[0]) - 3 - (len(lines) - 1) - 3) / (len(lines) - 1))

    # NOTE: This adds a '█' character at the end of all lines except the last, to make them equal length
    str_arr = np.array([list(line)
                        if line[-2:] == ']]'
                        else list(line) + ['█']
                        for line in lines if line != ' ...'], dtype=str)

    for i in range(str_arr.shape[0]):
        j, idx_start = 0, 2
        while idx_start + num_length < str_arr[i, :].size:
            if ''.join(str_arr[i, idx_start:(idx_start + 3)].tolist()) == '...':
                idx_start += 4
            else:
                if j > i + (0 if (idx_trunc is None or i < idx_trunc) else 1):
                    str_arr[i, idx_start:(idx_start + num_length)] = list(f" {char}" if (num_length == 2 and '.' not in arr) else f"{char:^{num_length}}")
                idx_start += num_length + 1
            j += 1

    arr_list = [''.join(elem).strip('█') for (i, elem) in enumerate(str_arr.tolist())]
    if idx_trunc is not None:
        arr_list.insert(idx_trunc, ' ...')

    return '\n'.join(arr_list)


# [untested/unverified]
def format_as_set(elements: list[str], edgeitems: Optional[int] = None) -> str:
    """Format a list of 1-D vectors as elements of a set. It is assumed that `elements` contains
    column-vector representations of NumPy arrays, all of equal length `n`, although no verification is performed, and unexpected
    behaviour might follow for inputs not satisfying these assumptions.
    
    Parameters
    ----------
    elements : list[str]
        List of string representation of 1-D arrays, presumable `str(v)`, where `v` is a NumPy array of size `(n,)`
    edgeitems : int, default=None
        Number of edgeitems after which truncation will take place. Default is `None`, meaning all
        elements will be diplayed.

    Warnings
    --------
    This feature has not been rigorously tested, and undefined behaviour might follows for inputs
    not adhering to the assumptions on their sizes

    Examples
    --------
    >>> verts = np.array([[1, 0],
                          [0, 1],
                          [0, 0]])
    >>> print(format_as_set([str(np.atleast_2d(vert).T) for vert in verts]))
    /[[1]  [[0]  [[0] \\
    \\ [0]], [1]], [0]]/

    Setting `edgeitems` to an integer causes truncation when the number of elements exceeds this.
     
    >>> verts = np.arange(100).reshape(20, 5)
    >>> print(format_as_set([str(np.atleast_2d(vert).T) for vert in verts], edgeitems=3))
    /[[0]  [[5]  [[10]       [[85]  [[90]  [[95] \\
    | [1]   [6]   [11]        [86]   [91]   [96] |
    < [2] , [7] , [12] , ..., [87] , [92] , [97] >
    | [3]   [8]   [13]        [88]   [93]   [98] |
    \\ [4]]  [9]]  [14]]       [89]]  [94]]  [99]]/
    """
    elem_lines = [[pad(elem, len(line.splitlines()[-1])) for elem in line.splitlines()] for line in elements]
    nlines = len(elem_lines[0])
    try:
        idx_trunc = elem_lines[0].index(' ...')
    except ValueError as _:
        idx_trunc = None
    idx_text = nlines // 2
    if edgeitems is not None and len(elem_lines) > (edgeitems * 2):
        elem_lines = elem_lines[:edgeitems] + [['    ' if idx != idx_text else ' ...' for idx in range(nlines)]] + elem_lines[-edgeitems:]

    match nlines:
        case 1:
            left_brackets = ['<']
            right_brackets = ['>']
        case 2:
            left_brackets = ['/', '\\']
            right_brackets = ['\\', '/']
        case _:
            left_brackets, right_brackets =[], []
            for idx in range(nlines):
                if idx == 0:
                    left_brackets.append('/')
                    right_brackets.append('\\')
                elif idx == nlines - 1:
                    left_brackets.append('\\')
                    right_brackets.append('/')
                elif (idx_trunc is not None and idx == idx_trunc) or (idx == nlines // 2):
                    left_brackets.append('<')
                    right_brackets.append('>')
                else:
                    left_brackets.append('|')
                    right_brackets.append('|')

    spacer = ([' ' if idx != idx_text else ',' for idx in range(nlines)]
              if nlines != 1
              else [' ' if idx != idx_text else ', ' for idx in range(nlines)])
    comb = '\n'.join([''.join(line) for line in zip(left_brackets,
                                                    *[item for elem in elem_lines for item in (elem, spacer)][:-1],
                                                    right_brackets)])
    if nlines == 1:
        # FIXME: Maybe replace the brackets entirely?
        comb = comb.replace('[[', '').replace(']]', '').replace('<', '{').replace('>', '}')

    return comb



# [untested/unverified]
def pad(text: str, length: int, char: str = " ") -> str:
    """Pad a string that is shorter than a certain length by appending copies of `char` (to the end).
    Leaves the string unchanged if the string is of greater or equal lenght.
    
    Parameters
    ----------
    test : str
        Text string to be padded
    length : int
        Length up to which the string must be padded
    char : str, default=" "
        Character with which to pad the array. Must be of length 1. Default character is a space.
    
    Raises
    ------
    ValueError
        If `char` is not of length 1

    Examples
    --------
    >>> text = "Sample text"
    print(pad(text, 20, char="_"))
    Sample text_________

    By default, a space is used as a padding character.

    >>> text_1, text_2 = "First part", "Second part"
    print(pad(text_1, 30) + text_2)
    First part                    Second part

    If the input text is longer then `length`, the original string is returned.
    >>> text = "Once upon a time"
    print(pad(text, 5, char= "?"))
    Once upon a time
    """
    if len(char) != 1:
        raise ValueError(f"The length of `char` must be 1, recieved '{char}' of length {len(char)}")
    return (text
            if len(text) >= length
            else text + ''.join([' '] * (length - len(text))))


def repr_items(obj: object,
               compact_ndarray: bool = False,
               ) -> list[tuple[str, str]]:
    """Return (attribute, formatted-value) pairs used by repr formatting. Flattens NumPy arrays by removing any `'\\n'` characters in its representation."""

    def fmt_repr_value(value: Any, pretty_ndarray: bool) -> str:
        if isinstance(value, np.ndarray):
            if not pretty_ndarray:
                return " ".join([elem.strip() for elem in repr(value).splitlines()])
            return f"NDArray[shape={value.shape}, dtype={value.dtype}]"
        if isinstance(value, tuple):
            inner = ", ".join(fmt_repr_value(item, pretty_ndarray) for item in value)
            if len(value) == 1:
                inner += ","
            return f"({inner})"
        return repr(value)

    return [(key, fmt_repr_value(value, compact_ndarray)) for key, value in obj.__dict__.items()]



def format_spec_to_opts(format_spec: str,
                        valid_repr: set[str] = {''},
                        ) -> tuple[Literal['', 'i', 'r', '#'] | None,
                                   str | None,
                                   Literal['float', 'int'] | None,
                                   int | None,
                                   dict[str, Callable[[Any], str]] | None,
                                   Literal['-', '+', ' '] | None,
                                   ]:
    if not format_spec:
        raise AssertionError("Expected 'format_spec' to be non-empty, but received ''")
    if valid_repr.intersection({'i', 'r', '#', 'd', 'f', 'e', 'E'}):
        raise AssertionError(f"'valid_repr' cannot contain 'i', 'r', '#', 'd', 'f', 'e', or 'E', but received valid_repr={valid_repr}")

    token = format_spec
    which_debug = None
    which_repr = ''
    to_dtype = None
    edgeitems = None
    formatter = None
    sign = None

    if token == 'r' or token == '#':
        which_debug = token
        token = token[1:]
    if 'r' in token or '#' in token:
        raise ValueError(f"Format specifiers 'r' and '#' do not except any additional symbols, received '{format_spec}'")
    if token.startswith('i'):
        which_debug = 'i'
        token = token[1:]
    if len(token) == 0:
        return which_debug, None, None, None, None, None
    if token[0] in valid_repr:
        which_repr = token[0]
        token = token[1:]
    if token and token[0] in {' ', '+', '-'}:
        sign = token[0]
        token = token[1:]
    if token and (char := token[0]) in {'f', 'e', 'E'}:
        sign = sign if sign is not None else ' '
        formatter = {
            'float': lambda value: f'{value:{sign}{'f' if char == 'f' else 'e'}}',
        }
        to_dtype = 'float'
        token = token[1:]
    if token.startswith('.'):
        digits, idx = '', 1
        while idx < len(token) and token[idx].isdigit():
            digits += token[idx]
            idx += 1
        if not digits:
            raise ValueError(f"Invalid format '{format_spec}': '.' character must be followed by at least one digit, received '{token}'")
        token = token[idx:]
        if token and (char := token[0]) in {'f', 'e', 'E'}:
            sign = sign if sign is not None else ' '
            if char == 'f' and digits == '0':
                formatter = {
                    'float': lambda value: f"{value:{sign}.0f}.",
                }
            else:
                formatter = {
                    'float': lambda value: f"{value:{sign}.{int(digits)}{char}}",
                }
            to_dtype = 'float'
            token = token[1:]
        elif token and token[0] == 'd':
            raise ValueError(f"Invalid format '{format_spec}': precision format specifier '.*' cannot be followed by 'd' as integer type does not take precision")
        elif not token:
            raise ValueError(f"Invalid format '{format_spec}': precision format specifier '.*' must be followed by 'f', 'e', or 'E', but was not followed by any character")
        else:
            raise ValueError(f"Invalid format '{format_spec}': precision format specifier '.*' must be followed by 'f', 'e', or 'E', received '{token[0]}'")
    elif token.startswith('d'):
        to_dtype = 'int'
        token = token[1:]
    if token.startswith('~'):
        digits = token[1:]
        if not digits.isdigit():
            raise ValueError(f"Invalid format '{format_spec}': edgeitems modifier '~*' must be the final element and followed by a positive integer, received '{token}'")
        edgeitems = int(digits)
        token = ''
    if token != '':
        raise ValueError(f"Invalid format '{format_spec}': substring not recognized starting with '{token}'")

    return which_debug, which_repr, to_dtype, edgeitems, formatter, sign
