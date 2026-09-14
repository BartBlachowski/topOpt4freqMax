function h = trajectory_fingerprint(X)
%TRAJECTORY_FINGERPRINT SHA-256 over shape and lossless double bytes.
md=java.security.MessageDigest.getInstance('SHA-256');
md.update(typecast(uint64(size(X)),'uint8'));md.update(typecast(double(X(:)),'uint8'));
h=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2).',1,[]));
end
