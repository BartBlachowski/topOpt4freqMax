function v = mac_mass(a, b, M)
%MAC_MASS  Mass-weighted modal assurance criterion (Eq. (9) of the paper):
%   (a'Mb)^2 / ((a'Ma)(b'Mb)).
v = (a'*(M*b))^2 / ((a'*(M*a)) * (b'*(M*b)));
end
