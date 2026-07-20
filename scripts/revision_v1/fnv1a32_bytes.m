function hash = fnv1a32_bytes(bytes)
%FNV1A32_BYTES  Deterministic wrapping FNV-1a hash of a byte sequence.
%
%   hash = FNV1A32_BYTES(bytes) returns a uint32. Multiplication is performed
%   in uint64 and explicitly masked to 32 bits. This is required in MATLAB:
%   direct uint32 multiplication saturates at UINT32_MAX instead of wrapping.

bytes = uint8(bytes(:));
hash = uint32(2166136261);          % FNV-1a 32-bit offset basis, 0x811c9dc5
prime = uint64(16777619);           % FNV-1a 32-bit prime,        0x01000193
mask = uint64(4294967295);          % 2^32 - 1

for k = 1:numel(bytes)
    mixed = bitxor(hash, uint32(bytes(k)));
    hash = uint32(bitand(uint64(mixed) * prime, mask));
end
end
