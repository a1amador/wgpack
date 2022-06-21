function adcpr = wrap_rdradcp(filepath_in,filepath_out,varargin)

if nargin>2
    NFIRST  = varargin{1};
    NEND    = varargin{2};
    adcpr = rdradcp(filepath_in,1,[NFIRST, NEND]);
else
    adcpr = rdradcp(filepath_in);
end

if filepath_out
    save(filepath_out,'adcpr','-v7.3')
end

