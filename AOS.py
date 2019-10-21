import numpy as np
import pandas as pd

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
        print()
    
    def set_ir(self, ir):
        self.d['IR'] = ir
        self.d['PC'] = self.d['PC'] + 1
        self.d['TC'] = self.d['TC'] % 2 +1
        self.output()
    
    def mov(self, reg, value):
        self.d[reg] = self.twos_complement(value, self.n_bits)
        self.d['TC'] = self.d['TC'] % 2 + 1
        self.d['PS'] = self.d[reg][2]
        self.output()
        
    def add(self, reg1, reg2, reg3):
        self.d[reg3] = self.twos_complement(bin(int(self.d[reg1], 2)) + bin(int(self.d[reg2], 2)), self.n_bits)
        self.d['TC'] = self.d['TC'] % 2 + 1
        self.d['PS'] = self.d[reg3][2]
        self.output()
        
    def mod(self, reg1, reg2, reg3):
        self.d[reg3] = self.twos_complement(divmod(int(self.d[reg1], 2), int(self.d[reg2], 2))[1], self.n_bits)
        self.d['TC'] = self.d['TC'] % 2 + 1
        self.d['PS'] = self.d[reg3][2]
        self.output()

proc_imit = ProcessorImitation(24)
proc_imit.set_ir('mov R1, 1256')
proc_imit.mov('R1', 1256)
proc_imit.set_ir('mov R2, 317')
proc_imit.mov('R2', 317)
proc_imit.set_ir('R1 mod R2')
proc_imit.mod('R1', 'R2', 'R3')
