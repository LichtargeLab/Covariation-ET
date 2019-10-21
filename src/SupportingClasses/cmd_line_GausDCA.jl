# cmd_line_GausDCA.jl

try
	using GaussDCA
catch
    using Pkg
	Pkg.clone("https://github.com/carlobaldassi/GaussDCA.jl")
	cd(joinpath(Pkg.devdir(), "GaussDCA"))
	run(`git pull origin master`)
	using GaussDCA
end

FNR = gDCA(ARGS[1]);

printrank(ARGS[2], FNR)
