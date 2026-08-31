#!/usr/bin/env python3
"""WP2 - independent branch-point mathematics for Du & Olhoff Eq. (4), (4a), (4b).
Computed from the source definitions, not from Phase-2D's numbers.  READ-ONLY."""
import numpy as np, math
from fractions import Fraction

def report(name, glow, dlow, ghigh=lambda x: x, dhigh=lambda x: 1.0):
    x0 = 0.1
    nb = np.nextafter(x0, 0.0); na = np.nextafter(x0, 1.0)
    print(f'--- {name} ---')
    print(f'  left limit  g(0.1-)  = {glow(nb)!r}')
    print(f'  value       g(0.1)   = {glow(x0)!r}     (branch condition x <= 0.1 is closed below)')
    print(f'  right limit g(0.1+)  = {ghigh(na)!r}')
    jump = ghigh(na) - glow(nb)
    print(f'  absolute jump        = {jump:.6e}')
    print(f'  multiplicative jump  = {ghigh(na)/glow(nb):.6e}')
    print(f'  left  derivative     = {dlow(x0):.10g}')
    print(f'  right derivative     = {dhigh(x0):.10g}')
    print(f'  C0 continuous        = {abs(jump) < 1e-15}')
    print(f'  C1 continuous        = {abs(dlow(x0)-dhigh(x0)) < 1e-12}')

# exact rational arithmetic at the branch point, free of floating point
F = Fraction
x0 = F(1, 10)
print('=== exact rational arithmetic at rho_e = 1/10 ===')
print(f'  Eq.(4)  low  : (1/10)^6                        = {x0**6}          = {float(x0**6):.6e}')
print(f'  Eq.(4)  high : 1/10                            = {x0}             = {float(x0):.6e}')
print(f'  Eq.(4)  jump ratio                             = {F(x0, x0**6)}   = {float(F(x0,x0**6)):.6e}')
c0 = F(100000)
print(f'  Eq.(4a) low  : 1e5*(1/10)^6                    = {c0*x0**6}       -> exactly {float(c0*x0**6)}')
print(f'  Eq.(4a) C0 residual (exact)                    = {c0*x0**6 - x0}')
c1, c2 = F(600000), F(-5000000)
print(f'  Eq.(4b) low  : 6e5*(1/10)^6 + (-5e6)*(1/10)^7  = {c1*x0**6 + c2*x0**7}')
print(f'  Eq.(4b) C0 residual (exact)                    = {c1*x0**6 + c2*x0**7 - x0}')
print(f'  Eq.(4b) slope: 6*6e5*(1/10)^5 + 7*(-5e6)*(1/10)^6 = {6*c1*x0**5 + 7*c2*x0**6}')
print()

print('=== IEEE-754 double evaluation ===')
report('Eq. (4)   g(x) = x^6            (x <= 0.1)', lambda x: x**6,       lambda x: 6*x**5)
report('Eq. (4a)  g(x) = 1e5 * x^6      (x <= 0.1)', lambda x: 1e5*x**6,   lambda x: 6*1e5*x**5)
report('Eq. (4b)  g(x) = 6e5x^6 - 5e6x^7(x <= 0.1)', lambda x: 6e5*x**6 - 5e6*x**7,
                                                     lambda x: 6*6e5*x**5 - 7*5e6*x**6)

print('\n=== global Lipschitz constants on [0,1] (the property that governs stability) ===')
xs = np.linspace(0, 0.1, 2_000_001)
for nm, d in (('Eq. (4) ', lambda x: 6*x**5),
              ('Eq. (4a)', lambda x: 6e5*x**5),
              ('Eq. (4b)', lambda x: 36e5*x**5 - 35e6*x**6)):
    L = float(np.max(np.abs(d(xs))))
    print(f'  {nm}: sup|g\'| on the low branch = {L:.6f}   (high branch = 1)'
          + ('   -- but g is DISCONTINUOUS, so no finite Lipschitz constant exists globally'
             if nm.strip() == 'Eq. (4)' else f'   -> global Lipschitz constant {max(L,1.0):.6f}'))

print('\n=== monotonicity and positivity of the amended low branch ===')
for nm, g in (('Eq. (4a)', lambda x: 1e5*x**6), ('Eq. (4b)', lambda x: 6e5*x**6 - 5e6*x**7)):
    v = g(xs)
    print(f'  {nm}: min={v.min():.6e}  max={v.max():.6e}  monotone increasing={bool(np.all(np.diff(v)>=0))}')

print('\n=== what the branch means for the E3 void floor (x clamped to 1e-3) ===')
for nm, g in (('Eq. (4) ', lambda x: x**6), ('Eq. (4a)', lambda x: 1e5*x**6)):
    print(f'  {nm}: g(1e-3) = {g(1e-3):.6e}   solid/void mass ratio = {1.0/g(1e-3):.3e}')
print('  E3 stiffness at 1e-3 = 1e7*(1e-3)^3 = 1.000000e-02 ; solid = 1e7')
for nm, m in (('Eq. (4) ', 1e-18), ('Eq. (4a)', 1e-13)):
    print(f'  {nm}: void element sqrt(K/M) ~ {math.sqrt(1e-2/m):.3e}  vs solid ~ {math.sqrt(1e7/1.0):.3e}'
          f'  (ratio {math.sqrt(1e-2/m)/math.sqrt(1e7):.2f}x)')
