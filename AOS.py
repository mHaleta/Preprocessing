import numpy as np
import pandas as pd
import re

class ProcessorImitation:
    def __init__(self, n_bits):
        self.d = {
            'IR': None,
            'R1': self.twos_complement(np.random.randint(-1000, 1000), n_bits),
            'R2': self.twos_complement(np.random.randint(-1000, 1000), n_bits),
            'R3': self.twos_complement(np.random.randint(-1000, 1000), n_bits),
            'PC': 0,
            'TC': 0,
            'PS': 0
        }
        self.n_bits = n_bits
        self.commands = {
            'mov': self.mov,
            'add': self.add,
            'mod': self.mod
        }
        
    def execute_command(self, command, *args):
        self.commands[command](*args)
        
    def twos_complement(self, n, bits):
        mask = (1 << bits) - 1
        if n < 0:
            n = ((abs(n) ^ mask) + 1)
            return format(n & mask, '#0{}b'.format(bits+2))
        else:
            return format(n & mask, '#0{}b'.format(bits+2))
    
    def output(self):
        frame = pd.DataFrame.from_dict(self.d, orient='index', columns=[''])
        print(frame)
    
    def set_ir(self, ir):
        self.d['IR'] = ir
        self.d['PC'] = self.d['PC'] + 1
        self.d['TC'] = self.d['TC'] % 2 + 1
        self.output()
    
    def mov(self, reg, value):
        self.set_ir('mov {}, {}'.format(reg, value))
        self.d[reg] = self.twos_complement(int(value), self.n_bits)
        self.d['TC'] = self.d['TC'] % 2 + 1
        self.d['PS'] = self.d[reg][2]
        self.output()
        
    def add(self, reg1, reg2, reg3='R3'):
        self.set_ir('add {}, {}'.format(reg1, reg2))
        self.d[reg3] = self.twos_complement(int(self.d[reg1], 2) + int(self.d[reg2], 2), self.n_bits)
        self.d['TC'] = self.d['TC'] % 2 + 1
        self.d['PS'] = self.d[reg3][2]
        self.output()
        
    def mod(self, reg1, reg2, reg3='R3'):
        self.set_ir('mod {}, {}'.format(reg1, reg2))
        self.d[reg3] = self.twos_complement(divmod(int(self.d[reg1], 2), int(self.d[reg2], 2))[1], self.n_bits)
        self.d['TC'] = self.d['TC'] % 2 + 1
        self.d['PS'] = self.d[reg3][2]
        self.output()
        
with open("Commands.txt", 'r') as f:
    commands = []
    for line in f:
        commands.append(line.replace('\n', ''))

proc_imit = ProcessorImitation(24)
for command in commands:
    proc_imit.execute_command(*re.split(' |, ', command))
