import ctypes
from ctypes import wintypes
import time


"""
    Borrowed from https://stackoverflow.com/questions/13564851/how-to-generate-keyboard-events
    Answered by: lucasg
    Edited by: Eryk Sun 

"""
user32 = ctypes.WinDLL('user32', use_last_error=True)

INPUT_MOUSE    = 0
INPUT_KEYBOARD = 1
INPUT_HARDWARE = 2

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP       = 0x0002
KEYEVENTF_UNICODE     = 0x0004
KEYEVENTF_SCANCODE    = 0x0008

MAPVK_VK_TO_VSC = 0

# msdn.microsoft.com/en-us/library/dd375731
VK_DOWN = 0x28
VK_UP = 0x26
VK_RIGHT = 0x27
VK_F4 = 0x73
VK_F5 = 0x74
VK_LCONTROL = 0xA2
VK_LSHIFT = 0xA0
VK_RETURN = 0x0D
VK_END = 0x23
VK_BACK = 0x08
VK_MENU = 0x12

S_key = 0x53
J_key = 0x4A
P_key = 0x50
G_key = 0x47
W_key = 0x57

# C struct definitions

wintypes.ULONG_PTR = wintypes.WPARAM

class MOUSEINPUT(ctypes.Structure):
    _fields_ = (("dx",          wintypes.LONG),
                ("dy",          wintypes.LONG),
                ("mouseData",   wintypes.DWORD),
                ("dwFlags",     wintypes.DWORD),
                ("time",        wintypes.DWORD),
                ("dwExtraInfo", wintypes.ULONG_PTR))

class KEYBDINPUT(ctypes.Structure):
    _fields_ = (("wVk",         wintypes.WORD),
                ("wScan",       wintypes.WORD),
                ("dwFlags",     wintypes.DWORD),
                ("time",        wintypes.DWORD),
                ("dwExtraInfo", wintypes.ULONG_PTR))

    def __init__(self, *args, **kwds):
        super(KEYBDINPUT, self).__init__(*args, **kwds)
        # some programs use the scan code even if KEYEVENTF_SCANCODE
        # isn't set in dwFflags, so attempt to map the correct code.
        if not self.dwFlags & KEYEVENTF_UNICODE:
            self.wScan = user32.MapVirtualKeyExW(self.wVk,
                                                 MAPVK_VK_TO_VSC, 0)

class HARDWAREINPUT(ctypes.Structure):
    _fields_ = (("uMsg",    wintypes.DWORD),
                ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD))

class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = (("ki", KEYBDINPUT),
                    ("mi", MOUSEINPUT),
                    ("hi", HARDWAREINPUT))
    _anonymous_ = ("_input",)
    _fields_ = (("type",   wintypes.DWORD),
                ("_input", _INPUT))

LPINPUT = ctypes.POINTER(INPUT)

def _check_count(result, func, args):
    if result == 0:
        raise ctypes.WinError(ctypes.get_last_error())
    return args

user32.SendInput.errcheck = _check_count
user32.SendInput.argtypes = (wintypes.UINT, # nInputs
                             LPINPUT,       # pInputs
                             ctypes.c_int)  # cbSize

# Functions

def PressKey(hexKeyCode):
    x = INPUT(type=INPUT_KEYBOARD,
              ki=KEYBDINPUT(wVk=hexKeyCode))
    user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    x = INPUT(type=INPUT_KEYBOARD,
              ki=KEYBDINPUT(wVk=hexKeyCode,
                            dwFlags=KEYEVENTF_KEYUP))
    user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))


def singlePress(hexKeyCode,n=1):
    for _ in range(n):
        PressKey(hexKeyCode)
        ReleaseKey(hexKeyCode)

def NavigateBCR(first):
    """Press Alt+Tab and hold Alt key for 2 seconds
    in order to see the overlay.
    """
    if(first):
        singlePress(VK_DOWN)
        singlePress(VK_UP)
        singlePress(VK_RETURN)
    else:
        singlePress(VK_DOWN,2)
        singlePress(VK_RETURN)

def saveToJpg():
    #save img
    PressKey(VK_LCONTROL)
    PressKey(S_key)
    time.sleep(1)
    ReleaseKey(VK_LCONTROL) 
    ReleaseKey(S_key)
    
    #jump to old extention and delete
    singlePress(VK_END)
    singlePress(VK_BACK,3)
    time.sleep(0.5)

    #Type jpg
    singlePress(J_key)
    singlePress(P_key)
    singlePress(G_key)
    singlePress(VK_RETURN)
    time.sleep(1)
    singlePress(VK_RETURN)
    time.sleep(1)
    

def closeBCRfile():
    #close down
    PressKey(VK_LCONTROL)
    singlePress(W_key)
    ReleaseKey(VK_LCONTROL)
    time.sleep(0.5)
    PressKey(VK_MENU)
    PressKey(VK_F4)
    ReleaseKey(VK_MENU)
    ReleaseKey(VK_F4)
    time.sleep(1)
    singlePress(VK_F5)

def closeFolder():
    #close down
    PressKey(VK_LCONTROL)
    singlePress(W_key)
    ReleaseKey(VK_LCONTROL)







