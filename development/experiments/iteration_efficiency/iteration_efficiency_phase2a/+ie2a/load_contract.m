function contract = load_contract(path)
%LOAD_CONTRACT Decode the frozen experiment contract.
if nargin < 1 || isempty(path)
    path = ie2a.paths().contract;
end
contract = jsondecode(fileread(path));
end
