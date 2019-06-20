'''
Fun example!
'''


def present_wrapper(func):
    '''
    This can wrap!
    '''
    def wrapper():
        print('Ice cream')
        func()
        print('Ice cream')
    return wrapper


@present_wrapper
def pets_are_great(mk,
                   here,
                   are,
                   multiple,
                   params):
    '''
    And even a comment!
    '''
    def _can_we_handle_func_within_func(yes):
        '''
        Good
        '''
        x = 1
        x

    y = 1
    y
