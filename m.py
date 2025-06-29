from typing import *
import math
import wave
import dataclasses

FRAMES_PER_SECOND = 44100
INV_FRAMES_PER_SECOND = 1 / FRAMES_PER_SECOND

class Trigger(int):
    def __new__(cls, frame, value, counter=0):
        obj = super().__new__(cls, value)
        obj.frame   = frame
        obj.counter = counter
        return obj

    def __eq__(self, other):
        if isinstance(other, Trigger):
            return int(self) == int(other) and self.frame == other.frame
        return int(self) == other

    
def clip(x):
    if x >= 256:
        print(x)
        return 255
    if x < 0:
        print(x)
        return 0
    return x

MIDI_INACCURATE = [8, 9, 9, 10, 10, 11, 12, 12, 13, 14, 15, 15, 16, 17, 18, 19, 21, 22, 23, 24, 26, 28, 29, 31, 33, 35, 37, 39, 41, 44, 46, 49, 52, 55, 58, 62, 65, 69, 73, 78, 82, 87, 92, 98, 104, 110, 117, 123, 131, 139, 147, 156, 165, 175, 185, 196, 208, 220, 233, 247, 262, 277, 294, 311, 330, 349, 370, 392, 415, 440, 466, 494, 523, 554, 587, 622, 659, 698, 740, 784, 831, 880, 932, 988, 1047, 1109, 1175, 1245, 1319, 1397, 1480, 1568, 1661, 1760, 1865, 1976, 2093, 2217, 2349, 2489, 2637, 2794, 2960, 3136, 3322, 3520, 3729, 3951, 4186, 4435, 4699, 4978, 5274, 5588, 5920, 6272, 6645, 7040, 7459, 7902, 8372, 8870, 9397, 9956, 10548, 11175, 11840, 12544]

PIPI = 2 * math.pi


def float_or_int(s: str) -> bool:
    # BUG: es. "123.456.678"
    return s.replace(".", "").isnumeric()
    
# =========================================================== #
# =========================================================== #

def sine_wave(frame: int, frequency: float, amplitude: float):
    # time = frame
    time = frame * INV_FRAMES_PER_SECOND
    # w    = math.sin(2 * math.pi * frequency * time)
    w    = math.sin(6.28 * frequency * time)
    return clip(round(amplitude * (w+1) / 2))


def sine_wave_optcheck(tokens: list[Any]) -> tuple[Any, int]:
    if len(tokens) < 3:
        return None, 1
    
    freq, amplitude, funcname, *rest = tokens

    if funcname == "sine":
        if isinstance(freq, str) and isinstance(amplitude, str):
            if freq.isnumeric() and amplitude.isnumeric():
                f = lambda fr,env,stack: stack.append(sine_wave(
                    fr, int(freq), int(amplitude)
                ))
                return f, 3
    return None, 1

# =========================================================== #

def noise(_frame: int, amplitude: float):
    return round(random.random() * amplitude)

def metro(frame: int, bps: int, unique_id: int, env: dict):
    if frame % (FRAMES_PER_SECOND // bps) == 0:

        trigger = env.triggers.get(unique_id, None)

        if trigger is None:
            trigger = Trigger(frame, 1)
        else:
            trigger.frame = frame
            trigger.counter += 1
            
        env.triggers[unique_id] = trigger
        return trigger

    if (t := env.triggers.get(unique_id, None)) is not None:
        if t != 0:
            t = Trigger(frame, 0)
            env.triggers[unique_id] = t
        return t

    return Trigger(-1, 0)


def gate(frame: int, frame_len: int, trigger: Trigger):
    if trigger == 1:
        return 1
    if frame < trigger.frame + frame_len:
        return 1
    return 0

def count_to_old(frame: int, every_x_frames: int, up_to: int):
    return (frame // every_x_frames) % up_to

def count_to(frame: int, trigger: Trigger, start: int, end: int):
    return start + (trigger.counter % (end - start + 1)) 

# =========================================================== #

def mtf(_frame: int, midi: int):
    return MIDI_INACCURATE[midi]
    # return 440 * 2 ** ((midi - 69) / 12)

def mtf_optcheck(tokens: list[Any]) -> tuple[Any, int]:
    if len(tokens) < 2:
        return None, 1
    
    arg, funcname, *rest = tokens
    if funcname == "mtf":
        if isinstance(arg, str) and arg.isnumeric():
            arg = int(arg)
            return str(mtf(None, arg)), 2
            # def inner(fr, env, stack, arg=arg):
            #     stack.append(mtf(None, arg))
            # return inner, 2
    return None, 1

# =========================================================== #

# TODO: sequenze di numeri e/o operazioni artimetiche vengono compresse
def math_take_until(tokens) -> list:
    ...
    
def generic_math_optcheck(tokens: list[Any]) -> tuple[Any, int]:
    operators = ["+", "-", "*", "/"]
    if not (tokens[0] in operators or float_or_int(tokens[0])):
        return None, 1
    ...

# =========================================================== #

def env_asr(frame: int, trigger: Trigger, atk: int, sus: float, rel: int):
    start = trigger.frame

    if frame > start + atk + rel:
        return 0

    if frame < start + atk:
        return sus * ((frame - start) / atk)

    return sus * (1 - (frame - start - atk) / rel)

def env_asr_optcheck(tokens: list[Any]) -> tuple[Any, int]:
    if len(tokens) < 4:
        return None, 1
    
    atk, sus, rel, funcname, *rest = tokens
    if funcname == "env_asr":
        if isinstance(atk, str) and isinstance(sus, str) and isinstance(rel, str):
            if atk.isnumeric() and sus.replace(".", "").isnumeric() and rel.isnumeric():
                atk, sus, rel = int(atk), float(sus), int(rel)
                def inner(fr, env, stack, atk=atk, sus=sus, rel=rel):
                    stack.append(
                        env_asr(fr, stack.pop(), atk, sus, rel)
                    )
                return inner, 4
    return None, 1


# =========================================================== #
# =========================================================== #

s = """
;; Esempio di variable
60 'myConst set

;; Esempio di variabile di tipo lista
[ 69 70 'myConst 71 70 ] 'myList set

;; Analogo a range(0, 5) in python, cambia ogni 1/8 di secondo
8 5 countTo 'myIdx set

;; Genera sine con pitch preso dalla lista
( 'myList get
  'myIdx get nth ) mtf 96 sine

;; Genera seconda sine
440 ( 4 8820 gate ) 64 * sine


;; Genera terza sine con inviluppo 
;;  5 metro
;;    1000 1 200 linenv
;; 'myEnvelope set

;; 440 ( 'myEnvelope get ) 96 * sine

;; Mixa le due sine
+ 2 /"""

s = """
;; Esempio di variable
60 'myConst set

;; Esempio di variabile di tipo lista
[ 69 70 'myConst 71 70 ] 'myList set

;; Analogo a range(0, 5) in python, cambia ogni 1/8 di secondo
8 5 countTo 'myIdx set

;; Genera sine con pitch preso dalla lista
( 'myList get
  'myIdx get nth ) mtf 96 sine

;; Genera seconda sine
440 ( 4 8820 gate ) 64 * sine

+ 2 /
"""

s = """
;; [ 57 62 64 ] 'tripletList set
;; [ 71 66 61 57 ] 'quadList set

3 3 countTo 'myIdx1 set
;; 4 4 countTo 'myIdx2 set

( 'tripletList get
  'myIdx1 get nth ) mtf 96 sine
 2 / 
;; ( 'quadList get
;;  'myIdx2 get nth ) 12 + mtf 96 sine

;; + 2 /

"""


s = """
3 metro 'myTrigger set
4 metro 'myTrigger2 set
5 metro 'myTrigger3 set

'myTrigger get
  64 1 4410 env_asr
  'myEnv set

'myTrigger2 get
  64 1 4410 env_asr
  'myEnv2 set

 'myTrigger3 get
   64 1 4410 env_asr
   'myEnv3 set

;; print_trig

880 ( 128 'myEnv get * ) sine

550 ( 255 'myEnv2 get * ) sine

1440 ( 128 'myEnv3 get * ) sine

+ + 3 /
"""

s = """
;; Esempio di variable
60 'myConst set

;; Esempio di variabile di tipo lista
[ 69 70 71 72 73 ] 'myList set

;; Analogo a range(0, 5) in python, cambia ogni 1/8 di secondo
8 5 countTo 'myIdx set  

;; Genera sine con pitch preso dalla lista
( 'myList get
  'myIdx get nth ) mtf 96 sine


;; Genera seconda sine
50 mtf ( 4 metro 64 1 4410 env_asr ) 96 * sine

+ 2 /
"""

s = """
;; Esempio di variable
70 'myConst set

4 metro dup

'myConst get 73 count_to
  'pitch set

61 1 4410 env_asr
  'myEnv set

'pitch get mtf ( 'myEnv get 128 * ) sine



5 metro dup
'myConst get 73 count_to 16 +
  'pitch2 set
1000 1 500 env_asr
  'myEnv2 set
'pitch2 get mtf ( 'myEnv2 get 128 * ) sine

+ 2 /

"""

# s = """
# 440 128 sine
# 449 128 sine
# + 2 /
# """


def tokenize(program: str):
    tokens = list()
    
    for line in program.split("\n"):
        if line.startswith(";;"):
            continue
        for token in line.split(" "):
            if token in "()":
                continue
            tokens.append(token)

    return tokens


def parse_list(tokens: list, env: dict):
    l = list()
    while (token := tokens.pop(0)) != "]":
        if token.isnumeric():
            try:
                l.append(int(token))
            except ValueError:
                l.append(float(token))
        else:
            raise Exception(f"Non valido per lista: {token}")
    return l


def optimize_tokens(tokens: list):
    list_optimizers = [
        mtf_optcheck,
        env_asr_optcheck, sine_wave_optcheck
    ]

    for optimizer in list_optimizers:
        out_tokens = list()
        i = 0

        while i < len(tokens):
            special_sub, increment = optimizer(tokens[i:])
            if special_sub is None:
                out_tokens.append(tokens[i])
                i += 1
            else:
                out_tokens.append(special_sub)
                i += increment

        tokens = out_tokens

    return out_tokens

    # =========================================================== #
    
    tokens = out_tokens
    out_tokens = list()
    i = 0

    while i < len(tokens):
        special_sub, increment = env_asr_optcheck(tokens[i:])
        if special_sub is None:
            out_tokens.append(tokens[i])
            i += 1
        else:
            out_tokens.append(special_sub)
            i += increment
        
    return out_tokens

@dataclasses.dataclass
class Env:
    triggers: dict[int, Trigger]
    variables: list[Any]
    

def compiler(program: str, env: dict):
    tokens_raw = tokenize(program)
    tokens = optimize_tokens(tokens_raw)

    print(tokens)
    
    symbol_to_id = dict()
    
    test_out = list()
    unique_id = 0
    
    while len(tokens) > 0:
        token = tokens.pop(0)
        unique_id += 1

        if isinstance(token, Callable):
            test_out.append(token)
            
        elif token.isnumeric():
            v = int(token)
            def inner(frame, env, stack, val=v):
                stack.append(val)
            test_out.append(inner)

        elif token.replace(".", "").isnumeric():
            v = float(token)
            def inner(frame, env, stack, val=v):
                stack.append(val)
            test_out.append(inner)

        elif token.startswith("'"):
            v = token
            enlarge_varlist = False
            
            if v in symbol_to_id:
                v = symbol_to_id[v]
            else:
                symbol_to_id[v] = len(symbol_to_id)
                v = symbol_to_id[v]
                enlarge_varlist = True
                
            def inner(frame, env, stack, val=v, enlarge=enlarge_varlist):
                stack.append(val)
                if enlarge:
                    env.variables.append(None)
            test_out.append(inner)

        elif token == "sine":
            def custom_sine(frame, env, stack):
                amplitude = stack.pop()
                frequency = stack.pop()
                r = sine_wave(frame, frequency, amplitude)
                stack.append(r)
            test_out.append(custom_sine)

        elif token == "rand":
            def inner(frame, env, stack):
                stack.append(random.random())
            test_out.append(inner)

        elif token == "print_trig":
            def inner(frame, env, stack):
                print(env.triggers)
            test_out.append(inner)

        elif token == "[":
            # BUG: in caso di lettura di variabili ovviamente non va bene!
            parsed_list = parse_list(tokens, env)
            def inner(frame, env, stack, val=parsed_list):
                stack.append(val)
            test_out.append(inner)
            
        elif token == "+":
            test_out.append(lambda frame, env, stack: stack.append(stack.pop() + stack.pop()))

        elif token == "-":
            def inner(frame, env, stack):
                b = stack.pop()
                a = stack.pop()
                stack.append(a - b)
            test_out.append(inner)
            
        elif token == "*":
            test_out.append(lambda frame, env, stack: stack.append(stack.pop() * stack.pop()))
            
        elif token == "/":
            def inner(frame, env, stack):    
                div = stack.pop()
                stack.append(round(stack.pop() / div))
            test_out.append(inner)
            
        elif token == "gate":
            def inner(frame, env, stack):    
                frame_len = int(stack.pop())
                trigger: Trigger = stack.pop()
                stack.append(gate(
                    frame, frame_len, trigger
                ))
            test_out.append(inner)
            
        elif token == "set":
            def inner(frame, env, stack):    
                name_id: int = stack.pop()
                value        = stack.pop()
                env.variables[name_id] = value
            test_out.append(inner)
            
        elif token == "get":
            def inner(frame, env, stack):    
                name_id: int = stack.pop()
                stack.append(env.variables[name_id])
            test_out.append(inner)

        elif token == "count_to":
            def inner(frame, env, stack):    
                up_to = stack.pop()
                start = stack.pop()
                trigger = stack.pop()
                if trigger == 1:
                    print(f"counter {trigger} {start} {up_to}")
                stack.append(count_to(frame, trigger, start, up_to))
            test_out.append(inner)

        elif token == "dup":
            def inner(frame, env, stack):
                stack.append(stack[-1])
            test_out.append(inner)

        elif token == "mtf":
            def inner(frame, env, stack):    
                stack.append(mtf(frame, stack.pop()))
            test_out.append(inner)

        elif token == "nth":
            def inner(frame, env, stack):    
                idx = round(stack.pop())
                lst = stack.pop() # TODO: pop o get?
                stack.append(lst[idx])
            test_out.append(inner)

        elif token == "metro":
            def inner(frame, env, stack, unique_id=unique_id):
                bps = stack.pop()
                stack.append(metro(frame, bps, unique_id, env))
            test_out.append(inner)

        elif token == "env_asr":
            def inner(frame, env, stack):
                rel = stack.pop()
                sus = float(stack.pop())
                atk = stack.pop()
                trigger = stack.pop()
                stack.append(env_asr(frame, trigger, atk, sus, rel))
            test_out.append(inner)

        elif token == "print":
            def inner(frame, env, stack):
                print(f"F{frame}", stack)
            test_out.append(inner)

        else:
            raise Exception(f"not found: {token}")

    return test_out
    # return stack.pop()

def interpreter(functions, frame, env):
    stack = list()
    for func in functions:
        func(frame, env, stack)
    r = int(stack.pop())
    # print(r)
    return r

def generate_frames(program: str, seconds: int):
    env = Env({}, [])
    
    import time
    start = time.time()
    functions = compiler(program, env)    
    for frame in range(0, int(FRAMES_PER_SECOND * seconds)):
        yield interpreter(functions, frame, env)
    print(time.time() - start)

# if __name__ == "__main__":        
if 1 == 1:
    with wave.open("output.wav", mode="wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(1)
        wav_file.setframerate(FRAMES_PER_SECOND)
        # wav_file.writeframes(bytes(sound_wave(440, 2.5)))
        wav_file.writeframes(bytes(generate_frames(s, 4)))

    import subprocess
    subprocess.run("aplay output.wav", shell=True)
