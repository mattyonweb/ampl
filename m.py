from typing import *
import math
import wave

FRAMES_PER_SECOND = 44100

def clip(x):
    if x >= 256:
        print(x)
        return 255
    if x < 0:
        print(x)
        return 0
    return x

def sine_wave(frame: int, frequency: float, amplitude: float):
    time = frame / FRAMES_PER_SECOND
    w    = math.sin(2 * math.pi * frequency * time)
    return clip(round(amplitude * (w+1) / 2))

def noise(_frame: int, amplitude: float):
    return round(random.random() * amplitude)

def metro(frame: int, bps: int):
    if frame % (FRAMES_PER_SECOND // bps) == 0:
        return 1
    return 0

def gate(frame: int, every_x_frame: int, frame_len: int):
    rem = frame % every_x_frame
    if rem < frame_len:
        return 1
    return 0

def count_to(frame: int, every_x_frames: int, up_to: int):
    return (frame // every_x_frames) % up_to
    
def mtf(_frame: int, midi: int):
    return round(440 * 2 ** ((midi - 69) / 12))


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
10 24 countTo 'myIdx set

71 ( 'myIdx get ) - mtf 96 sine

4 /
"""
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
            l.append(float(token))
        elif token.startswith("'"):
            l.append(env["vars"].get(token, 0))
        else:
            raise Exception(f"Non valido per lista: {token}")
    return l

class Const:
    def __init__(self, funcall, *args):
        self.funcall = funcall
        self.args = args

def optimize_tokens(tokens: list):
    mask = ("<num>", "<num>", "sine")
    out_tokens = list()
    i = 0
    
    while i <= len(tokens) - 3:
        if tokens[i].isnumeric() and tokens[i+1].isnumeric() and tokens[i+2] == "sine":
            out_tokens.append(
                Const(sine_wave, float(tokens[i]), float(tokens[i+1]))
            )
            i = i + 3
        else:
            out_tokens.append(tokens[i])
            i = i + 1

    return out_tokens


def compiler(program: str, env: dict):
    tokens_raw = tokenize(program)
    tokens = optimize_tokens(tokens_raw)
    # tokens = tokens_raw
    test_out = list()
    
    while len(tokens) > 0:
        token = tokens.pop(0)

        
        if isinstance(token, Const):
            
            funcall = token.funcall
            args    = token.args
            def inner(frame, env, stack, funcall=funcall, args=args):
                stack.append(funcall(frame, *args))
            test_out.append(inner)
            
        if token.isnumeric():
            v = float(token)
            def inner(frame, env, stack, val=v):
                stack.append(val)
            test_out.append(inner)

        elif token in "()":
            continue    

        elif token.startswith("'"):
            # stack.append(token)
            v = token
            def inner(frame, env, stack, val=v):
                stack.append(val)
            test_out.append(inner)

        elif token == "sine":
            def custom_sine(frame, env, stack):
                amplitude = stack.pop()
                frequency = stack.pop()
                r = sine_wave(frame, frequency, amplitude)
                stack.append(r)
            test_out.append(custom_sine)

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
                bps = int(stack.pop())
                stack.append(gate(
                    frame, FRAMES_PER_SECOND // bps, frame_len
                ))
            test_out.append(inner)
            
        elif token == "set":
            def inner(frame, env, stack):    
                name  = stack.pop()
                value = stack.pop()
                env["vars"][name] = value
            test_out.append(inner)
            
        elif token == "get":
            def inner(frame, env, stack):    
                name = stack.pop()
                stack.append(env["vars"].get(name, 0))
            test_out.append(inner)
            
        elif token == "countTo":
            def inner(frame, env, stack):    
                up_to = stack.pop()
                bps   = stack.pop()
                stack.append(count_to(frame, FRAMES_PER_SECOND // bps , up_to))
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
        else:
            raise Exception(f"not found: {token}")

    return test_out
    # return stack.pop()

def interpreter(functions, frame, env):
    stack = list()
    for func in functions:
        func(frame, env, stack)
    r = stack.pop()
    return r

def generate_frames(program: str, seconds: int):
    env = {
        "vars" : {},
        "triggers": {},
        "tracks": {}
    }
    import time
    start = time.time()
    functions = compiler(program, env)    
    for frame in range(0, int(FRAMES_PER_SECOND * seconds)):
        yield interpreter(functions, frame, env)
    print(time.time() - start)

        
with wave.open("output.wav", mode="wb") as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(1)
    wav_file.setframerate(FRAMES_PER_SECOND)
    # wav_file.writeframes(bytes(sound_wave(440, 2.5)))
    wav_file.writeframes(bytes(generate_frames(s, 8)))

import subprocess
subprocess.run("aplay output.wav", shell=True)
