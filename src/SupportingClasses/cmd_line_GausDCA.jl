# cmd_line_GausDCA.jl

try
	using GaussDCA
catch
	Pkg.clone("https://github.com/carlobaldassi/GaussDCA.jl")
	using GaussDCA
end

FNR = gDCA(ARGS[1]);

printrank(ARGS[2], FNR)
