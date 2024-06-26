import ctypes, time
from pyWinActivate import win_activate

SendInput = ctypes.windll.user32.SendInput

PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functions

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


mapping_commands = {
    "up": 0x2C, # W
    "down": 0x1F, # S
    "stop": 0x01, #esc
    "left": 0x10, #Q
    "right": 0x20, #D
    "go": 0x1C, # enter
    "yes": 0x26, # l
    "no": 0x32, # m
}

stoppers = {
    "up": ["down"],
    "right": ["left"],
    "left": ["right"],
    "down": ["up"],
}
stoppers = {k: v + ["stop"] for k,v in stoppers.items()}

def KeyPress(command: str, last_command: str):
    hexKeyCode = mapping_commands.get(command)
    if not hexKeyCode:
        return
    time.sleep(0.1)
    PressKey(hexKeyCode) # press Q
    if command in ["stop", "yes", "no", "go"] or stoppers[command] == last_command :
        time.sleep(0.1)
        ReleaseKey(hexKeyCode)
    else :
        time.sleep(.7)
        ReleaseKey(hexKeyCode)
    return command

def focus(title):
    win_activate(window_title=title, partial_match=True)