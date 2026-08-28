function h = sha256_hex(data)
%SHA256_HEX  Lowercase hex SHA-256 of a char string or a numeric/logical array.
%
%   h = SHA256_HEX(txt)  hashes the UTF-8 bytes of a char row vector or string.
%   h = SHA256_HEX(A)    hashes the RAW IEEE-754 bytes of a numeric array,
%                        column-major, exactly as MATLAB stores them.
%
%   The numeric form is a bit-exactness test, not a similarity test: two
%   density fields hash equal only if every one of their doubles is
%   bit-identical, including the sign of zero and the payload of any NaN.  That
%   is the property the equivalence harness needs -- "final omega agrees to
%   four figures" is precisely what it must not accept.
%
%   Logical arrays are hashed as uint8 0/1.  Class is folded into the hash via
%   a short prefix, so a double 1 and an int8 1 do not collide.
%
%   See also REPRO2007_NORMALIZED_CONFIG, VERIFY_REPRO2007_BENCHMARK_EQUIVALENCE.

if isstring(data)
    if ~isscalar(data)
        error('sha256_hex:NonScalarString', 'string input must be scalar.');
    end
    data = char(data);
end

if ischar(data)
    bytes = uint8(unicode2native(data(:).', 'UTF-8'));
    prefix = uint8(unicode2native('char:', 'UTF-8'));
elseif islogical(data)
    bytes = uint8(data(:));
    prefix = uint8(unicode2native('logical:', 'UTF-8'));
elseif isnumeric(data)
    bytes = typecast(data(:), 'uint8');
    prefix = uint8(unicode2native([class(data) ':'], 'UTF-8'));
else
    error('sha256_hex:UnsupportedType', ...
        'Cannot hash a value of class %s.', class(data));
end

md = java.security.MessageDigest.getInstance('SHA-256');
md.update(prefix);
if ~isempty(bytes)
    md.update(bytes);
end
digest = typecast(md.digest(), 'uint8');
h = lower(reshape(dec2hex(digest, 2).', 1, []));
end
