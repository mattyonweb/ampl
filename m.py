from typing import *
import math
import wave

FRAMES_PER_SECOND = 44100

def sine_wave(frame: int, frequency: float, amplitude: float):
    time = frame / FRAMES_PER_SECOND
    w    = math.sin(2 * math.pi * frequency * time)
    return round(amplitude * (w+1) / 2)

def noise(_frame: int, amplitude: float):
    return round(random.random() * amplitude)

def metro(frame: int, bps: int):
    if frame % (FRAMES_PER_SECOND // bps) == 0:
        return 1
    return 0

def gate(frame: int, bps: int, frame_len: int, click_avoid=0):
    rem = frame % (FRAMES_PER_SECOND // bps)
    if rem < frame_len + click_avoid:
        if rem < 64:
            # return 1 / (64-rem)
            return 0.5
        if rem > frame_len - 64:
            return 0.5 #1 / (frame_len-rem)
        return 1
    return 0

def count_to(frame: int, bps: int, up_to: int):
    return 1 + ((frame // (FRAMES_PER_SECOND // bps)) % up_to)
    
def mtf(_frame: int, midi: int):
    return 440 * 2 ** ((midi - 69) / 12)
    
s = """
4 metro 'myTrigger set


  'myCounter get
  'myTrigger get
   +
  'myCounter set

  'myCounter get
   2
   == 1 'otherTrigger set

'myTrigger get
  2 triggerEvery 'myTriggerDouble set

440 ( 4 8820 gate ) 64 * sine
550 ( 3 8820 gate ) 100 * sine
+ 2 /
"""

s = """
440 ( 4 8820 gate ) 64 * sine
( 6 5 countTo 73 + mtf ) 96 sine
+ 2 /
"""

def tokenize(program: str):
    tokens = s.replace("\n", " ").split(" ")
    return tokens


def interpreter(program: str, frame: int, env: dict):
    stack = list()
    tokens = tokenize(program)
    for token in tokenize(program):
        if token.isnumeric():
            stack.append(float(token))
        elif token in "()":
            continue
        elif token.startswith("'"):
            stack.append(token)
        elif token == "sine":
            amplitude = stack.pop()
            frequency = stack.pop()
            stack.append(sine_wave(frame, frequency, amplitude))
        elif token == "+":
            stack.append(stack.pop() + stack.pop())
        elif token == "*":
            stack.append(round(stack.pop() * stack.pop()))
        elif token == "/":
            div = stack.pop()
            stack.append(round(stack.pop() / div))
        elif token == "gate":
            frame_len = int(stack.pop())
            bps = int(stack.pop())
            stack.append(gate(frame, bps, frame_len))
        elif token == "set":
            name  = stack.pop()
            value = stack.pop()
            env["vars"][name] = value
        elif token == "get":
            name = stack.pop()
            stack.append(env["vars"].get(name, 0))
        elif token == "countTo":
            up_to = stack.pop()
            bps   = stack.pop()
            stack.append(count_to(frame, bps, up_to))
        elif token == "mtf":
            stack.append(mtf(frame, stack.pop()))
        else:
            raise Exception(f"not found: {token}")
            
    return stack.pop()


def generate_frames(program: str, seconds: int):
    env = {
        "vars" : {},
        "triggers": {}
    }
    for frame in range(0, int(FRAMES_PER_SECOND * seconds)):
        yield interpreter(program, frame, env)
    
# def sound_wave(frequency, num_seconds):
#     for frame in range(round(num_seconds * FRAMES_PER_SECOND)):
#         time = frame / FRAMES_PER_SECOND
#         amplitude = math.sin(2 * math.pi * frequency * time)
#         yield round((amplitude + 1) / 2 * 255)

        
with wave.open("output.wav", mode="wb") as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(1)
    wav_file.setframerate(FRAMES_PER_SECOND)
    # wav_file.writeframes(bytes(sound_wave(440, 2.5)))
    wav_file.writeframes(bytes(generate_frames(s, 8)))

import subprocess
subprocess.run("aplay output.wav", shell=True)
