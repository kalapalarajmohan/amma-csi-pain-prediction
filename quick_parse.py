import sys, ast, math
for line in sys.stdin:
    parts = line.strip().split(' ', 1)
    if len(parts) ==2:
        t, raw = parts
        b = eval(raw)
        ints = list(b)
        amps = [math.sqrt(ints[i]**2 + ints[i+1]**2) for i in range(0, len(ints)-1, 2)]
        print(t, amps[:5])
