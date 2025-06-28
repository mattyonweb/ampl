from typing import *
import math
import wave
import dataclasses

FRAMES_PER_SECOND = 44100


class Trigger(int):
    def __new__(cls, frame, value):
        obj = super().__new__(cls, value)
        obj.frame = frame
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

# =========================================================== #
# =========================================================== #

def sine_wave(frame: int, frequency: float, amplitude: float):
    time = frame / FRAMES_PER_SECOND
    w    = math.sin(2 * math.pi * frequency * time)
    return clip(round(amplitude * (w+1) / 2))


def sine_wave_optcheck(tokens: list[Any]) -> tuple[Any, int]:
    if len(tokens) < 3:
        return None, 1
    
    freq, amplitude, funcname, *rest = tokens

    if funcname == "sine":
        if isinstance(freq, str) and isinstance(amplitude, str):
            if freq.isnumeric() and amplitude.isnumeric():
                f = lambda fr,env,stack: sine_wave(
                    fr, int(freq), int(amplitude)
                )
                return f, 3
    return None, 1

# =========================================================== #

def noise(_frame: int, amplitude: float):
    return round(random.random() * amplitude)

def metro(frame: int, bps: int, unique_id: int, env: dict):
    if frame % (FRAMES_PER_SECOND // bps) == 0:

        trigger = Trigger(frame, 1)
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

def count_to(frame: int, every_x_frames: int, up_to: int):
    return (frame // every_x_frames) % up_to
    
def mtf(_frame: int, midi: int):
    return round(440 * 2 ** ((midi - 69) / 12))

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
            if atk.isnumeric() and sus.isnumeric() and rel.isnumeric():
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
10 24 countTo 'myIdx set

71 ( 'myIdx get ) - mtf 96 sine

4 /
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


def optimize_tokens(tokens: list):
    out_tokens = list()
    i = 0

    while i < len(tokens):
        special_sub, increment = sine_wave_optcheck(tokens[i:])
        if special_sub is None:
            out_tokens.append(tokens[i])
            i += 1
        else:
            out_tokens.append(special_sub)
            i += increment

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
        
        if isinstance(token, Const):
            funcall = token.funcall
            args    = token.args
            
            def inner(frame, env, stack, funcall=funcall, args=args):
                stack.append(funcall(frame, *args))
            test_out.append(inner)

        elif isinstance(token, Callable):
            test_out.append(token)
            
        elif token.isnumeric():
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
