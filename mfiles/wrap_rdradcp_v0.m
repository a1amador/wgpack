function adcpr = wrap_rdradcp_v0(filepath_in,filepath_out)

adcpr = rdradcp(filepath_in);

if filepath_out
    save(filepath_out,'adcpr','-v7.3')
end

