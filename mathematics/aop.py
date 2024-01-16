'''
Array operations
----------------

This library was designed to apply list operations
'''

def amult(lst1, lst2):
    '''
    Multiply each i element of list
    '''
    rtn = []
    if not len(lst1) == len(lst2):
        raise Exception("List must be the same size")
    for index, value in enumerate(lst1):
        rtn.append(lst1[index]*lst2[index])

    return rtn

def apow(lst1, lst2):
    '''
    Pow each i element of list1 to i element of list2
    '''
    rtn = []
    if not len(lst1) == len(lst2):
        raise Exception("List must be the same size")
    for index, value in enumerate(lst1):
        rtn.append(lst1[index]**lst2[index])

    return rtn

def valpow(val, lst):

    return [val**x for x in lst]

def adiv(lst1, lst2):
    '''
    Divide each i element of list1 to i element of list2
    '''
    rtn = []
    if not len(lst1) == len(lst2):
        raise Exception("List must be the same size")
    for index, value in enumerate(lst1):
        rtn.append(lst1[index]/lst2[index])

    return rtn